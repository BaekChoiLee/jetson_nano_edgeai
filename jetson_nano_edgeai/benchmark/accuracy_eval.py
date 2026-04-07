#!/usr/bin/env python3
"""Accuracy evaluation across runtimes (top-1, top-5) including TRT and ncnn.
Supports classification (ImageNet) and detection (COCO) models."""

import argparse
import os
import csv
import json
import numpy as np
from PIL import Image

import torch
import torchvision.models as models
import torchvision.models.detection as det_models
import torchvision.transforms as transforms


MODEL_FACTORIES = {
    "mobilenetv3_small": models.mobilenet_v3_small,
    "resnet50": models.resnet50,
    "shufflenet_v2": models.shufflenet_v2_x1_0,
    "ssd_mobilenet_v2": det_models.ssdlite320_mobilenet_v3_large,
}

DETECTION_MODELS = {"ssd_mobilenet_v2"}

# Standard ImageNet preprocessing (C, H, W)
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# COCO detection preprocessing (for SSD 320x320)
COCO_TRANSFORM = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

# COCO 91-class to 80-class contiguous mapping
COCO_91_TO_80 = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19,
    21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27,
    32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35,
    40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43,
    49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51,
    57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59,
    65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67,
    77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75,
    86: 76, 87: 77, 88: 78, 89: 79, 90: 80,
}


def load_imagenet_val(data_dir, max_images=500):
    """Load ImageNet validation images (subdirs or flat)."""
    images = []
    labels = []

    # Format 1: subdirectory per class (ImageNet-1K format)
    val_dir = os.path.join(data_dir, "val")
    if os.path.isdir(val_dir):
        classes = sorted(os.listdir(val_dir))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        count = 0
        for cls in classes:
            cls_dir = os.path.join(val_dir, cls)
            if not os.path.isdir(cls_dir): continue
            
            for fname in sorted(os.listdir(cls_dir)):
                if count >= max_images:
                    break
                fpath = os.path.join(cls_dir, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                    images.append(IMAGENET_TRANSFORM(img))
                    labels.append(class_to_idx[cls])
                    count += 1
                except Exception:
                    continue
            if count >= max_images:
                break
        return images, labels

    # Format 2: flat images + labels file
    labels_file = os.path.join(data_dir, "val.txt")
    img_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(img_dir):
        img_dir = data_dir

    if os.path.exists(labels_file):
        with open(labels_file) as f:
            for i, line in enumerate(f):
                if i >= max_images:
                    break
                parts = line.strip().split()
                fname, label = parts[0], int(parts[1])
                fpath = os.path.join(img_dir, fname)
                if os.path.exists(fpath):
                    try:
                        img = Image.open(fpath).convert("RGB")
                        images.append(IMAGENET_TRANSFORM(img))
                        labels.append(label)
                    except Exception:
                        continue
        return images, labels

    print(f"[WARN] Could not find ImageNet validation data in {data_dir}")
    return [], []


def load_coco_val(data_dir, max_images=200):
    """Load COCO validation images and annotations for detection.

    Expected structure:
      data_dir/
        images/       (or val2017/)
        annotations/  (instances_val2017.json)
    """
    images = []
    annotations = {}  # image_id -> list of {bbox, category_id}

    # Find images directory
    img_dir = None
    for candidate in ["val2017", "images"]:
        cand_path = os.path.join(data_dir, candidate)
        if os.path.isdir(cand_path):
            img_dir = cand_path
            break
    if img_dir is None:
        print(f"[WARN] COCO images not found in {data_dir}")
        return [], {}

    # Find annotations
    ann_path = None
    for candidate in [
        os.path.join(data_dir, "annotations", "instances_val2017.json"),
        os.path.join(data_dir, "instances_val2017.json"),
    ]:
        if os.path.exists(candidate):
            ann_path = candidate
            break

    if ann_path is None:
        print(f"[WARN] COCO annotations not found in {data_dir}")
        return [], {}

    print(f"  Loading COCO annotations from {ann_path}...")
    with open(ann_path) as f:
        coco_data = json.load(f)

    # Build annotations dict
    ann_by_image = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append({
            "bbox": ann["bbox"],  # [x, y, w, h] in COCO format
            "category_id": ann["category_id"],
        })

    # Load images
    img_entries = sorted(coco_data.get("images", []), key=lambda x: x["id"])
    count = 0
    for img_info in img_entries:
        if count >= max_images:
            break
        img_id = img_info["id"]
        fname = img_info["file_name"]
        fpath = os.path.join(img_dir, fname)
        if not os.path.exists(fpath):
            continue

        try:
            img = Image.open(fpath).convert("RGB")
            w, h = img.size
            img_tensor = COCO_TRANSFORM(img)
            images.append({
                "tensor": img_tensor,
                "image_id": img_id,
                "orig_size": (w, h),
            })
            # Ground truth: convert COCO [x,y,w,h] to [x1,y1,x2,y2] normalized
            gt_boxes = []
            for ann in ann_by_image.get(img_id, []):
                x, y, bw, bh = ann["bbox"]
                gt_boxes.append({
                    "bbox": [x, y, x + bw, y + bh],
                    "category_id": ann["category_id"],
                })
            annotations[img_id] = gt_boxes
            count += 1
        except Exception:
            continue

    print(f"  Loaded {len(images)} COCO images.")
    return images, annotations


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compute_map(all_predictions, all_ground_truths, iou_threshold=0.5):
    """Compute mAP@IoU for detection results.

    all_predictions: list of dicts with boxes, labels, scores per image
    all_ground_truths: list of dicts with gt_boxes, gt_labels per image
    """
    # Collect all predictions by class
    class_preds = {}
    class_gts = {}

    for img_idx, (preds, gts) in enumerate(zip(all_predictions, all_ground_truths)):
        for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
            label = int(label)
            if label not in class_preds:
                class_preds[label] = []
            class_preds[label].append({
                "img_idx": img_idx,
                "box": box,
                "score": float(score),
            })
        for box, label in zip(gts["boxes"], gts["labels"]):
            label = int(label)
            if label not in class_gts:
                class_gts[label] = {}
            if img_idx not in class_gts[label]:
                class_gts[label][img_idx] = []
            class_gts[label][img_idx].append(box)

    # Compute AP per class
    aps = []
    for cls in class_gts:
        if cls not in class_preds:
            aps.append(0.0)
            continue

        preds = sorted(class_preds[cls], key=lambda x: x["score"], reverse=True)
        gt_by_img = class_gts[cls]
        n_gt = sum(len(v) for v in gt_by_img.values())

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched = {img_idx: set() for img_idx in gt_by_img}

        for i, pred in enumerate(preds):
            img_idx = pred["img_idx"]
            pred_box = pred["box"]

            if img_idx not in gt_by_img:
                fp[i] = 1
                continue

            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_by_img[img_idx]):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx not in matched[img_idx]:
                tp[i] = 1
                matched[img_idx].add(best_gt_idx)
            else:
                fp[i] = 1

        # Compute precision/recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / n_gt if n_gt > 0 else tp_cumsum
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        # AP via 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            p = precision[recall >= t]
            ap += max(p) / 11.0 if len(p) > 0 else 0
        aps.append(ap)

    return round(float(np.mean(aps)) * 100, 2) if aps else 0.0


def eval_pytorch(model_name, images, labels, device="cpu"):
    """Evaluate PyTorch model accuracy. Fixed CPU 0.1% Bug."""
    if device == "cpu":
        torch.backends.mkldnn.enabled = False

    model = MODEL_FACTORIES[model_name](pretrained=True)
    model = model.eval().to(device)
    correct_top1, correct_top5 = 0, 0

    with torch.no_grad():
        for img, label in zip(images, labels):
            output = model(img.unsqueeze(0).to(device))
            _, top5_idx = output.topk(5, dim=1)
            top5_idx = top5_idx.cpu().numpy()[0]
            if top5_idx[0] == label:
                correct_top1 += 1
            if label in top5_idx:
                correct_top5 += 1

    n = len(labels)
    return {"top1": round(correct_top1/n*100, 2) if n else 0, "top5": round(correct_top5/n*100, 2) if n else 0, "n": n}


def eval_pytorch_detection(model_name, coco_images, coco_annotations, device="cpu"):
    """Evaluate PyTorch detection model using COCO data. Returns mAP@0.5."""
    model = MODEL_FACTORIES[model_name](pretrained=True)
    model = model.eval().to(device)

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for img_data in coco_images:
            img_tensor = img_data["tensor"].to(device)
            img_id = img_data["image_id"]
            orig_w, orig_h = img_data["orig_size"]

            output = model([img_tensor])
            det = output[0]

            # Scale boxes back to original image coords
            boxes = det["boxes"].cpu().numpy()
            labels = det["labels"].cpu().numpy()
            scores = det["scores"].cpu().numpy()

            # Filter low-confidence
            mask = scores > 0.3
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]

            # Scale from 320×320 to original size
            boxes[:, [0, 2]] *= orig_w / 320.0
            boxes[:, [1, 3]] *= orig_h / 320.0

            all_preds.append({
                "boxes": boxes.tolist(),
                "labels": labels.tolist(),
                "scores": scores.tolist(),
            })

            # Ground truth
            gt_anns = coco_annotations.get(img_id, [])
            gt_boxes = [a["bbox"] for a in gt_anns]
            gt_labels = [a["category_id"] for a in gt_anns]
            all_gts.append({"boxes": gt_boxes, "labels": gt_labels})

    map50 = compute_map(all_preds, all_gts, iou_threshold=0.5)
    return {"mAP@0.5": map50, "n": len(coco_images)}


def eval_onnxrt(onnx_path, images, labels, provider="CUDAExecutionProvider"):
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=[provider])
    input_name = session.get_inputs()[0].name
    correct_top1, correct_top5 = 0, 0

    for img, label in zip(images, labels):
        input_data = img.unsqueeze(0).numpy()
        output = session.run(None, {input_name: input_data})[0]
        top5_idx = np.argsort(output[0])[-5:][::-1]
        if top5_idx[0] == label:
            correct_top1 += 1
        if label in top5_idx:
            correct_top5 += 1

    n = len(labels)
    return {"top1": round(correct_top1/n*100, 2) if n else 0, "top5": round(correct_top5/n*100, 2) if n else 0, "n": n}


def eval_tflite(tflite_path, images, labels):
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=tflite_path)
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=tflite_path)

    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    input_dtype = interpreter.get_input_details()[0]["dtype"]
    
    correct_top1, correct_top5 = 0, 0
    for img, label in zip(images, labels):
        input_data = img.unsqueeze(0).numpy().astype(input_dtype)
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)[0]
        top5_idx = np.argsort(output)[-5:][::-1]
        if top5_idx[0] == label:
            correct_top1 += 1
        if label in top5_idx:
            correct_top5 += 1

    n = len(labels)
    return {"top1": round(correct_top1/n*100, 2) if n else 0, "top5": round(correct_top5/n*100, 2) if n else 0, "n": n}


def eval_tensorrt(engine_path, images, labels):
    """Evaluate TensorRT Engine accuracy using pycuda and python bindings."""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
        
    context = engine.create_execution_context()
    
    # Memory allocation
    h_input = np.empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = np.empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    
    correct_top1, correct_top5 = 0, 0
    for img, label in zip(images, labels):
        np.copyto(h_input, img.numpy().ravel())
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        
        top5_idx = np.argsort(h_output)[-5:][::-1]
        if top5_idx[0] == label:
            correct_top1 += 1
        if label in top5_idx:
            correct_top5 += 1
            
    n = len(labels)
    return {"top1": round(correct_top1/n*100, 2) if n else 0, "top5": round(correct_top5/n*100, 2) if n else 0, "n": n}


def eval_ncnn(param_path, bin_path, images, labels):
    """Evaluate NCNN model accuracy via pyncnn."""
    import ncnn
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False # CPU evaluation for accuracy isolation
    net.load_param(param_path)
    net.load_model(bin_path)

    input_name = net.input_names()[0]
    output_name = net.output_names()[0]
    
    correct_top1, correct_top5 = 0, 0
    for img, label in zip(images, labels):
        ex = net.create_extractor()
        mat_in = ncnn.Mat(img.numpy().astype(np.float32))
        ex.input(input_name, mat_in)
        _, mat_out = ex.extract(output_name)
        
        out_np = np.array(mat_out)
        top5_idx = np.argsort(out_np)[-5:][::-1]
        if top5_idx[0] == label:
            correct_top1 += 1
        if label in top5_idx:
            correct_top5 += 1
            
    n = len(labels)
    return {"top1": round(correct_top1/n*100, 2) if n else 0, "top5": round(correct_top5/n*100, 2) if n else 0, "n": n}


def main():
    parser = argparse.ArgumentParser(description="Multi-Runtime Accuracy Evaluator")
    parser.add_argument("--model", required=True, choices=list(MODEL_FACTORIES.keys()))
    parser.add_argument("--data-dir", required=True, help="ImageNet or COCO validation data dir")
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()

    is_detection = args.model in DETECTION_MODELS

    if is_detection:
        # ===== Detection Model Evaluation (COCO mAP) =====
        print(f"[Detection] Loading COCO validation data from {args.data_dir}...")
        coco_images, coco_annotations = load_coco_val(args.data_dir, args.max_images)
        if not coco_images:
            print("[ERROR] No COCO images loaded.")
            return

        results = []

        # PyTorch CPU
        print("[PyTorch CPU - Detection]")
        r = eval_pytorch_detection(args.model, coco_images, coco_annotations, "cpu")
        r["runtime"] = "pytorch_cpu"
        results.append(r)
        print(f"  mAP@0.5: {r['mAP@0.5']}%")

        # PyTorch CUDA
        if torch.cuda.is_available():
            print("[PyTorch CUDA - Detection]")
            r = eval_pytorch_detection(args.model, coco_images, coco_annotations, "cuda")
            r["runtime"] = "pytorch_cuda"
            results.append(r)
            print(f"  mAP@0.5: {r['mAP@0.5']}%")

        # Save detection results
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, f"{args.model}_accuracy_complete.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["runtime", "mAP@0.5", "n"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[DONE] Detection Accuracy Saved to: {csv_path}")

    else:
        # ===== Classification Model Evaluation (Top-1/Top-5) =====
        print(f"Loading {args.max_images} validation images from {args.data_dir}...")
        images, labels = load_imagenet_val(args.data_dir, args.max_images)
        if not images:
            print("[ERROR] No images loaded.")
            return
        print(f"Loaded {len(images)} images.\n")
    
        results = []

        # 1. PyTorch CPU
        print("[PyTorch CPU]")
        r = eval_pytorch(args.model, images, labels, "cpu")
        r["runtime"] = "pytorch_cpu"
        results.append(r)
        print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")

        # 2. PyTorch CUDA
        if torch.cuda.is_available():
            print("[PyTorch CUDA]")
            r = eval_pytorch(args.model, images, labels, "cuda")
            r["runtime"] = "pytorch_cuda"
            results.append(r)
            print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")

        # 3. ONNX Runtime
        onnx_path = os.path.join(args.model_dir, f"{args.model}.onnx")
        if os.path.exists(onnx_path):
            try:
                print("[ONNX Runtime CUDA]")
                r = eval_onnxrt(onnx_path, images, labels)
                r["runtime"] = "onnxrt_cuda"
                results.append(r)
                print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")
            except Exception as e: print(f"  [SKIP] {e}")

        # 4. TFLite
        tflite_path = os.path.join(args.model_dir, f"{args.model}.tflite")
        if os.path.exists(tflite_path):
            try:
                print("[TFLite CPU]")
                r = eval_tflite(tflite_path, images, labels)
                r["runtime"] = "tflite_cpu"
                results.append(r)
                print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")
            except Exception as e: print(f"  [SKIP] {e}")

        # 5. TensorRT (FP16 & INT8)
        trt_fp16_path = os.path.join(args.model_dir, f"{args.model}_fp16.engine")
        if os.path.exists(trt_fp16_path):
            try:
                print("[TensorRT FP16]")
                r = eval_tensorrt(trt_fp16_path, images, labels)
                r["runtime"] = "tensorrt_fp16"
                results.append(r)
                print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")
            except Exception as e: print(f"  [SKIP] {e}")

        trt_int8_path = os.path.join(args.model_dir, f"{args.model}_int8.engine")
        if os.path.exists(trt_int8_path):
            try:
                print("[TensorRT INT8]")
                r = eval_tensorrt(trt_int8_path, images, labels)
                r["runtime"] = "tensorrt_int8"
                results.append(r)
                print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")
            except Exception as e: print(f"  [SKIP] {e}")

        # 6. ncnn
        ncnn_param = os.path.join(args.model_dir, f"{args.model}.param")
        ncnn_bin = os.path.join(args.model_dir, f"{args.model}.bin")
        if os.path.exists(ncnn_param) and os.path.exists(ncnn_bin):
            try:
                print("[ncnn Python]")
                r = eval_ncnn(ncnn_param, ncnn_bin, images, labels)
                r["runtime"] = "ncnn"
                results.append(r)
                print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")
            except Exception as e: print(f"  [SKIP] {e}")

        # Summary Output
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, f"{args.model}_accuracy_complete.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["runtime", "top1", "top5", "n"])
            writer.writeheader()
            writer.writerows(results)
    
        print(f"\n[DONE] All Runtimes Accuracy Saved to: {csv_path}")

if __name__ == "__main__":
    main()
