#!/usr/bin/env python3
# =============================================================================
# accuracy_eval.py — 멀티 런타임 정확도 평가기 (공통 전처리/평가 적용)
# =============================================================================

import argparse
import os
import csv
import json
import numpy as np
import torch
import torchvision.models as models
from PIL import Image

MODEL_FACTORIES = {
    "mobilenetv3_small": models.mobilenet_v3_small,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
    "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
}

# ===== 공통 함수 (전 런타임 대상) =====
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_imagenet(img_path, input_size=224):
    """Resize(짧은변 256) -> CenterCrop(224) -> Normalize -> NCHW float32 np.ndarray"""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = 256.0 / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)

    left = (nw - input_size) // 2
    top  = (nh - input_size) // 2
    img = img.crop((left, top, left + input_size, top + input_size))

    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))              # HWC -> CHW
    x = np.expand_dims(x, axis=0).astype(np.float32)  # NCHW
    return x

def topk_from_logits(logits, gt_label, k=(1,5)):
    """
    logits: np.ndarray [N,C] 또는 [C]
    gt_label: int (0-based)
    """
    logits = np.asarray(logits)
    if logits.ndim == 1:
        logits = logits[None, :]
    assert logits.ndim == 2, f"Expected [N,C], got {logits.shape}"
    assert 0 <= gt_label < logits.shape[1], "gt_label index mismatch (0-based 확인)"

    maxk = max(k)
    idx = np.argsort(-logits, axis=1)[:, :maxk]   # 내림차순 topk index
    res = {}
    for kk in k:
        hit = (idx[:, :kk] == gt_label).any(axis=1).astype(np.float32)
        res[f"top{kk}"] = float(hit.mean() * 100.0)
    return res, idx[0, 0], idx[0, :5]


def load_imagenet_val(data_dir, max_images=500):
    images = []
    labels = []
    val_dir = os.path.join(data_dir, "val")
    if os.path.isdir(val_dir):
        classes = sorted(os.listdir(val_dir))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        count = 0
        for cls in classes:
            cls_dir = os.path.join(val_dir, cls)
            if not os.path.isdir(cls_dir): continue
            for fname in sorted(os.listdir(cls_dir)):
                if count >= max_images: break
                fpath = os.path.join(cls_dir, fname)
                try:
                    images.append(preprocess_imagenet(fpath))
                    labels.append(class_to_idx[cls])
                    count += 1
                except Exception:
                    continue
            if count >= max_images: break
        return images, labels

    labels_file = os.path.join(data_dir, "val.txt")
    img_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(img_dir): img_dir = data_dir

    if os.path.exists(labels_file):
        with open(labels_file) as f:
            for i, line in enumerate(f):
                if i >= max_images: break
                parts = line.strip().split()
                fname, label = parts[0], int(parts[1])
                fpath = os.path.join(img_dir, fname)
                if os.path.exists(fpath):
                    try:
                        images.append(preprocess_imagenet(fpath))
                        labels.append(label)
                    except Exception:
                        continue
        return images, labels
    print(f"[WARN] Could not find ImageNet in {data_dir}")
    return [], []

def collect_acc(n_top1_hits, n_top5_hits, n):
    return {"top1": round(n_top1_hits/n*100, 2) if n else 0, "top5": round(n_top5_hits/n*100, 2) if n else 0, "n": n}


def apply_accuracy_gates(results):
    """가짜 성공 패턴을 status/status_reason으로 태깅한다."""
    by_runtime = {r.get("runtime"): r for r in results}
    cpu = by_runtime.get("pytorch_cpu")
    cuda = by_runtime.get("pytorch_cuda")

    for r in results:
        r["status"] = "measured"
        r["status_reason"] = "ok"
        r["error"] = r.get("error", "")
        top1 = r.get("top1")
        top5 = r.get("top5")
        n = r.get("n")
        try:
            top1 = float(top1) if top1 is not None else None
            top5 = float(top5) if top5 is not None else None
        except (TypeError, ValueError):
            top1, top5 = None, None
        try:
            n = int(n) if n is not None else None
        except (TypeError, ValueError):
            n = None

        if top1 is None or top5 is None:
            r["status"] = "invalid"
            r["status_reason"] = "missing_topk"
            r["error"] = r["error"] or "missing top1/top5"
            continue
        if top5 < top1:
            r["status"] = "invalid"
            r["status_reason"] = "top5_lt_top1"
            r["error"] = r["error"] or "top5 < top1"
            continue
        if n is not None and n >= 50 and top1 <= 0.2 and top5 <= 0.2:
            r["status"] = "invalid"
            r["status_reason"] = "suspicious_dummy_accuracy"
            r["error"] = r["error"] or "top1/top5 fixed near 0.2"

    # CPU/CUDA 차이가 지나치게 큰 경우 CPU 결과를 invalid로 표기
    if cpu and cuda:
        try:
            cpu_top1 = float(cpu.get("top1"))
            cuda_top1 = float(cuda.get("top1"))
            if cuda_top1 - cpu_top1 > 10.0:
                cpu["status"] = "invalid"
                cpu["status_reason"] = "cpu_cuda_gap_gt_10pp"
                cpu["error"] = cpu.get("error") or "top1 gap > 10pp vs pytorch_cuda"
        except (TypeError, ValueError):
            pass

def eval_pytorch(model_name, images, labels, device="cpu"):
    if device == "cpu":
        torch.backends.mkldnn.enabled = False
        torch.set_num_threads(1)  # aarch64 NEON SIMD stability fix

    try:
        weights = getattr(models, f"{model_name.title().replace('v', 'V').replace('Net', 'Net_')}_Weights", None)
        if weights:
            model = MODEL_FACTORIES[model_name](weights=weights.DEFAULT)
        else:
            model = MODEL_FACTORIES[model_name](pretrained=True)
    except Exception:
        model = MODEL_FACTORIES[model_name](pretrained=True)

    model = model.eval().to(device)
    correct_top1, correct_top5 = 0, 0

    with torch.no_grad():
        for img_np, label in zip(images, labels):
            output = model(torch.from_numpy(img_np).to(device)).cpu().numpy()
            res, _, _ = topk_from_logits(output, label)
            if res["top1"] == 100.0: correct_top1 += 1
            if res["top5"] == 100.0: correct_top5 += 1

    return collect_acc(correct_top1, correct_top5, len(labels))

def eval_onnxrt(onnx_path, images, labels, provider="CUDAExecutionProvider"):
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=[provider])
    input_name = session.get_inputs()[0].name
    correct_top1, correct_top5 = 0, 0

    for img_np, label in zip(images, labels):
        output = session.run(None, {input_name: img_np})[0]
        res, _, _ = topk_from_logits(output, label)
        if res["top1"] == 100.0: correct_top1 += 1
        if res["top5"] == 100.0: correct_top5 += 1

    return collect_acc(correct_top1, correct_top5, len(labels))

def eval_tflite(tflite_path, images, labels, use_gpu=False):
    delegates = []
    InterpreterClass = None
    if use_gpu:
        try:
            import tflite_runtime.interpreter as tflite
            delegates.append(tflite.load_delegate("libdelegate_gpu.so"))
            InterpreterClass = tflite.Interpreter
        except Exception:
            try:
                import tensorflow as tf
                try:
                    delegates.append(tf.lite.experimental.load_delegate("libdelegate_gpu.so"))
                except Exception:
                    delegates = []
                InterpreterClass = tf.lite.Interpreter
            except ImportError:
                import tflite_runtime.interpreter as tflite
                InterpreterClass = tflite.Interpreter
    else:
        try:
            import tflite_runtime.interpreter as tflite
            InterpreterClass = tflite.Interpreter
        except ImportError:
            import tensorflow as tf
            InterpreterClass = tf.lite.Interpreter

    interpreter = InterpreterClass(
        model_path=tflite_path,
        experimental_delegates=delegates if delegates else None,
    )
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    input_dtype = interpreter.get_input_details()[0]["dtype"]

    correct_top1, correct_top5 = 0, 0
    for img_np, label in zip(images, labels):
        input_data = img_np.astype(input_dtype)
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)[0]
        res, _, _ = topk_from_logits(output, label)
        if res["top1"] == 100.0: correct_top1 += 1
        if res["top5"] == 100.0: correct_top5 += 1

    return collect_acc(correct_top1, correct_top5, len(labels))

def eval_tensorrt(engine_path, images, labels):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    h_input = np.empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = np.empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    correct_top1, correct_top5 = 0, 0
    for img_np, label in zip(images, labels):
        np.copyto(h_input, img_np.ravel())
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        res, _, _ = topk_from_logits(h_output.reshape(1, -1), label)
        if res["top1"] == 100.0: correct_top1 += 1
        if res["top5"] == 100.0: correct_top5 += 1

    return collect_acc(correct_top1, correct_top5, len(labels))

def eval_ncnn(param_path, bin_path, images, labels, use_vulkan=False):
    import ncnn
    net = ncnn.Net()
    net.opt.use_vulkan_compute = use_vulkan
    net.load_param(param_path)
    net.load_model(bin_path)
    input_name = net.input_names()[0]
    output_name = net.output_names()[0]

    correct_top1, correct_top5 = 0, 0
    for img_np, label in zip(images, labels):
        ex = net.create_extractor()
        # img_np is [1, C, H, W]; ncnn.Mat expects (W, H, C) layout.
        # Create Mat with explicit dims and copy channel planes to avoid
        # shape misinterpretation from direct ndarray construction.
        chw = img_np[0].astype(np.float32)  # [C, H, W]
        c, h, w = chw.shape
        mat_in = ncnn.Mat(w, h, c)
        mat_arr = np.array(mat_in, copy=False)
        if mat_arr.shape == chw.shape:
            mat_arr[...] = chw
        else:
            for ci in range(c):
                ch = mat_in.channel(ci)
                np.array(ch, copy=False)[...] = chw[ci]
        ex.input(input_name, mat_in)
        _, mat_out = ex.extract(output_name)

        out_np = np.array(mat_out)
        res, _, _ = topk_from_logits(out_np.reshape(1, -1), label)
        if res["top1"] == 100.0: correct_top1 += 1
        if res["top5"] == 100.0: correct_top5 += 1

    return collect_acc(correct_top1, correct_top5, len(labels))

def main():
    parser = argparse.ArgumentParser(description="Multi-Runtime Accuracy Evaluator")
    parser.add_argument("--model", required=True, choices=list(MODEL_FACTORIES.keys()))
    parser.add_argument("--data-dir", required=True, help="ImageNet validation data dir")
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()

    print(f"Loading {args.max_images} validation images from {args.data_dir}...")
    images, labels = load_imagenet_val(args.data_dir, args.max_images)
    if not images:
        print("[ERROR] No images loaded.")
        return
    print(f"Loaded {len(images)} images.\n")

    results = []

    print("[PyTorch CPU]")
    r = eval_pytorch(args.model, images, labels, "cpu")
    r["runtime"] = "pytorch_cpu"
    results.append(r)
    print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")

    if torch.cuda.is_available():
        print("[PyTorch CUDA]")
        r = eval_pytorch(args.model, images, labels, "cuda")
        r["runtime"] = "pytorch_cuda"
        results.append(r)
        print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")

    onnx_path = os.path.join(args.model_dir, f"{args.model}.onnx")
    if os.path.exists(onnx_path):
        try:
            print("[ONNX Runtime CUDA]")
            r = eval_onnxrt(onnx_path, images, labels, provider="CUDAExecutionProvider")
            r["runtime"] = "onnxrt_cuda"
            results.append(r)
            print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")
        except Exception as e: print(f"  [SKIP] {e}")

        try:
            print("[ONNX Runtime TRT]")
            r = eval_onnxrt(onnx_path, images, labels, provider="TensorrtExecutionProvider")
            r["runtime"] = "onnxrt_trt"
            results.append(r)
            print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")
        except Exception as e: print(f"  [SKIP] {e}")

    tflite_path = os.path.join(args.model_dir, f"{args.model}.tflite")
    if os.path.exists(tflite_path):
        try:
            print("[TFLite CPU]")
            r = eval_tflite(tflite_path, images, labels, use_gpu=False)
            r["runtime"] = "tflite_cpu"
            results.append(r)
            print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")
        except Exception as e: print(f"  [SKIP] {e}")

        try:
            print("[TFLite GPU]")
            r = eval_tflite(tflite_path, images, labels, use_gpu=True)
            r["runtime"] = "tflite_gpu"
            results.append(r)
            print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")
        except Exception as e: print(f"  [SKIP] {e}")

    trt_fp32_path = os.path.join(args.model_dir, f"{args.model}_fp32.engine")
    if os.path.exists(trt_fp32_path):
        try:
            print("[TensorRT FP32]")
            r = eval_tensorrt(trt_fp32_path, images, labels)
            r["runtime"] = "tensorrt_fp32"
            results.append(r)
            print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")
        except Exception as e: print(f"  [SKIP] {e}")

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

    ncnn_param = os.path.join(args.model_dir, f"{args.model}.param")
    ncnn_bin = os.path.join(args.model_dir, f"{args.model}.bin")
    if os.path.exists(ncnn_param) and os.path.exists(ncnn_bin):
        try:
            print("[ncnn CPU]")
            r = eval_ncnn(ncnn_param, ncnn_bin, images, labels, use_vulkan=False)
            r["runtime"] = "ncnn_cpu"
            results.append(r)
            print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")
        except Exception as e: print(f"  [SKIP] {e}")

        try:
            print("[ncnn Vulkan]")
            r = eval_ncnn(ncnn_param, ncnn_bin, images, labels, use_vulkan=True)
            r["runtime"] = "ncnn_vulkan"
            results.append(r)
            print(f"  Top-1: {r['top1']}%  Top-5: {r['top5']}%")
        except Exception as e: print(f"  [SKIP] {e}")

    apply_accuracy_gates(results)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{args.model}_accuracy_complete.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["runtime", "top1", "top5", "n", "status", "status_reason", "error"],
        )
        writer.writeheader()
        writer.writerows(results)

    json_path = os.path.join(args.output_dir, f"{args.model}_classification.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] All Runtimes Accuracy Saved to: {csv_path} and {json_path}")

if __name__ == "__main__":
    main()
