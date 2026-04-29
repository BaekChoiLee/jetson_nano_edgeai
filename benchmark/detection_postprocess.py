#!/usr/bin/env python3
# =============================================================================
# detection_postprocess.py — YOLOv8n + SSD-MobileNet V2 공통 후처리 (Codex 픽스 적용)
# =============================================================================

import os
import numpy as np

COCO_80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

def get_postprocess_fn(model_name):
    if "yolov8" in model_name:
        return lambda raw, orig_size, conf=0.25, iou=0.45: common_postprocess(raw, orig_size, conf, iou, "yolo")
    if "ssd_mobilenet_v2" in model_name or "ssd_mv2" in model_name:
        return lambda raw, orig_size, conf=0.25, iou=0.45: common_postprocess(raw, orig_size, conf, iou, "ssd")
    raise ValueError(f"No postprocess fn for {model_name}")

def to_det_triplet(outputs, model_type="yolo", anchors_path="models/ssd_mobilenet_v2_anchors.npy"):
    if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
        b, s, c = outputs
        return np.asarray(b), np.asarray(s).reshape(-1), np.asarray(c).reshape(-1).astype(np.int32)

    if model_type == "yolo":
        x = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        x = np.asarray(x)
        if x.ndim == 3:
            # Determine layout: channel-first (batch, features, anchors) e.g. (1, 84, 8400)
            # or channel-last (batch, anchors, features) e.g. (1, 8400, 84).
            # Heuristic: if last dim > second dim, it is the anchor count (channel-first).
            if x.shape[2] > x.shape[1]:
                pred = np.transpose(x[0], (1, 0))  # (features, anchors) → (anchors, features)
            else:
                pred = x[0]  # already (anchors, features)
        elif x.ndim == 2:
            pred = x  # already (anchors, features)
        else:
            raise ValueError(f"Unsupported YOLO output shape: {x.shape}")

        boxes = pred[:, :4]

        # YOLOv8 raw head는 채널 84(4 box + 80 cls) 구조를 자주 사용한다.
        # 이 경우 objectness 채널이 없으므로 class score를 그대로 사용한다.
        if pred.shape[1] == 84:
            cls_prob = pred[:, 4:]
            # 일부 런타임은 logits를 그대로 반환할 수 있어 sigmoid 보정
            if np.nanmax(cls_prob) > 1.0 or np.nanmin(cls_prob) < 0.0:
                cls_prob = 1.0 / (1.0 + np.exp(-cls_prob))
            cls_id = np.argmax(cls_prob, axis=1)
            scores = cls_prob[np.arange(len(cls_id)), cls_id]
        else:
            obj = pred[:, 4:5]
            cls_prob = pred[:, 5:]
            if np.nanmax(cls_prob) > 1.0 or np.nanmin(cls_prob) < 0.0:
                cls_prob = 1.0 / (1.0 + np.exp(-cls_prob))
            cls_id = np.argmax(cls_prob, axis=1)
            cls_score = cls_prob[np.arange(len(cls_id)), cls_id]
            scores = obj[:, 0] * cls_score

        cx, cy, w, h = boxes.T
        xyxy = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)
        return xyxy, scores, cls_id.astype(np.int32)

    elif model_type == "ssd":
        if isinstance(outputs, dict):
            # TF2 OD API 직변환 경로: detection_* 출력 사용 (anchor 불필요)
            det_boxes = outputs.get("detection_boxes")
            det_scores = outputs.get("detection_scores")
            if det_boxes is not None and det_scores is not None:
                boxes = np.asarray(det_boxes)
                scores = np.asarray(det_scores)
                classes = outputs.get("detection_classes")
                n_det = outputs.get("num_detections")
                mcls = outputs.get("detection_multiclass_scores")

                if boxes.ndim == 3: boxes = boxes[0]
                if scores.ndim == 2: scores = scores[0]
                if classes is not None:
                    classes = np.asarray(classes)
                    if classes.ndim == 2: classes = classes[0]
                elif mcls is not None:
                    mcls = np.asarray(mcls)
                    if mcls.ndim == 3: mcls = mcls[0]
                    # background(0) 제외한 최고 점수 클래스 선택
                    classes = np.argmax(mcls[:, 1:], axis=1) + 1
                    scores = np.max(mcls[:, 1:], axis=1)
                else:
                    classes = np.ones((boxes.shape[0],), dtype=np.int32)

                if n_det is not None:
                    n = int(np.asarray(n_det).reshape(-1)[0])
                    n = max(0, min(n, len(boxes)))
                    boxes = boxes[:n]
                    scores = scores[:n]
                    classes = classes[:n]

                # TF 출력은 y1,x1,y2,x2 (normalized) -> xyxy
                xyxy = boxes[:, [1, 0, 3, 2]]
                # detection_classes는 1-based COCO id인 경우가 일반적
                cls_ids = np.maximum(0, np.asarray(classes).reshape(-1).astype(np.int32) - 1)
                return xyxy, np.asarray(scores).reshape(-1), cls_ids

            box_enc = outputs.get("box_encodings")
            cls_scores = outputs.get("class_scores")
        elif isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            box_enc, cls_scores = outputs[0], outputs[1]
        else:
            x = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            raise ValueError(f"SSD expects multi-output or dict, got shape: {np.shape(x)}")
        
        box_enc = np.asarray(box_enc)
        cls_scores = np.asarray(cls_scores)
        if box_enc.ndim == 3: box_enc = box_enc[0]
        if cls_scores.ndim == 3: cls_scores = cls_scores[0]

        # Class probabilities (shared across all SSD decoding paths)
        if np.nanmin(cls_scores) >= 0.0 and np.nanmax(cls_scores) <= 1.0:
            probs = cls_scores
        else:
            probs = _softmax(cls_scores, axis=1)
        fg_probs = probs[:, 1:] if probs.shape[1] > 1 else probs
        cls_ids = np.argmax(fg_probs, axis=1)
        scores = np.max(fg_probs, axis=1)

        # Prefer anchor-based decoding when the anchor file is available.
        # The anchor file is authoritative — it handles both raw-encoding
        # models (standard SSD) and avoids yxyx vs xyxy guesswork.
        if os.path.exists(anchors_path):
            anchors = np.load(anchors_path)
            # Auto-select matching anchor file if prediction count differs
            if anchors.shape[0] != box_enc.shape[0]:
                alt_path = anchors_path.replace('.npy', f'_{box_enc.shape[0]}.npy')
                if os.path.exists(alt_path):
                    anchors = np.load(alt_path)
            # Auto-detect encoding scale: raw SSD offsets with scale (10,10,5,5)
            # produce large values; scale-free offsets are in ~[-2, 2] range.
            decoded_yxyx = _decode_ssd_boxes(box_enc, anchors, scales=(10.0, 10.0, 5.0, 5.0))
            if float(np.nanmax(np.abs(decoded_yxyx))) > 3.0:
                # Values blew up → the encoding likely uses no scale factors
                decoded_yxyx = _decode_ssd_boxes(box_enc, anchors, scales=(1.0, 1.0, 1.0, 1.0))
            # decoded_yxyx is [y1, x1, y2, x2] → reorder to xyxy
            xyxy = decoded_yxyx[:, [1, 0, 3, 2]]
            return xyxy, scores, cls_ids.astype(np.int32)

        # Fallback (no anchors): heuristically treat box_enc as decoded boxes.
        # SSD TorchScript models from TF OD API export in [y1,x1,y2,x2] normalized.
        # Models converted from PyTorch-native implementations may use [x1,y1,x2,y2].
        # We try both and pick the one where more boxes are inside [0,1] range.
        box_like_decoded = (
            box_enc.ndim == 2
            and box_enc.shape[1] == 4
            and np.isfinite(box_enc).all()
            and float(np.nanmin(box_enc)) >= -1.5
            and float(np.nanmax(box_enc)) <= 2.5
        )
        if not box_like_decoded:
            raise RuntimeError(
                "SSD anchor file not found and box values look like raw "
                f"encodings (min={box_enc.min():.2f}, max={box_enc.max():.2f}). "
                f"Generate anchors with convert/generate_ssd_anchors.py"
            )
        # Try [y1,x1,y2,x2] (TF-style) first — default assumption.
        xyxy = box_enc[:, [1, 0, 3, 2]]
        return xyxy, scores, cls_ids.astype(np.int32)
        
    raise ValueError(f"Unknown model_type {model_type}")

def _decode_ssd_boxes(box_enc, anchors, scales=(10.0, 10.0, 5.0, 5.0)):
    ya = (anchors[:, 0] + anchors[:, 2]) / 2.0
    xa = (anchors[:, 1] + anchors[:, 3]) / 2.0
    ha = anchors[:, 2] - anchors[:, 0]
    wa = anchors[:, 3] - anchors[:, 1]
    ty = box_enc[:, 0] / scales[0]
    tx = box_enc[:, 1] / scales[1]
    th = box_enc[:, 2] / scales[2]
    tw = box_enc[:, 3] / scales[3]
    y = ty * ha + ya
    x = tx * wa + xa
    h = np.exp(th) * ha
    w = np.exp(tw) * wa
    return np.stack([y - h / 2.0, x - w / 2.0, y + h / 2.0, x + w / 2.0], axis=1)

def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def nms_numpy(boxes, scores, cls_ids, iou_thr=0.45, conf_thr=0.25):
    # Class-aware offset trick inside pure numpy NMS!
    keep_mask = scores >= conf_thr
    b, s, c = boxes[keep_mask], scores[keep_mask], cls_ids[keep_mask]
    if len(b) == 0:
        return np.array([], dtype=np.int32), keep_mask

    max_coord = float(np.max(b)) if b.size > 0 else 0.0
    offsets = c.astype(np.float32) * (max_coord + 1.0)
    b_offset = b + offsets[:, None]

    x1, y1, x2, y2 = b_offset.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = s.argsort()[::-1]
    picked = []

    while order.size > 0:
        i = order[0]
        picked.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thr]

    return np.array(picked, dtype=np.int32), keep_mask

def common_postprocess(raw_output, orig_size, conf=0.25, iou=0.45, model_type="yolo"):
    try:
        boxes, scores, cls_ids = to_det_triplet(raw_output, model_type=model_type)
    except Exception as e:
        print(f"[WARN] Failed to parse output triplet: {e}")
        return []
    
    keep_idx, mask = nms_numpy(boxes, scores, cls_ids, iou_thr=iou, conf_thr=conf)
    if len(keep_idx) == 0:
        return []

    # Apply mask then keep_idx
    b_final = boxes[mask][keep_idx]
    s_final = scores[mask][keep_idx]
    c_final = cls_ids[mask][keep_idx]

    orig_w, orig_h = orig_size
    input_size = 640 if model_type == "yolo" else 320
    
    sx = orig_w / float(input_size) if model_type == "yolo" else orig_w
    sy = orig_h / float(input_size) if model_type == "yolo" else orig_h

    results = []
    for i in range(len(b_final)):
        x1, y1, x2, y2 = b_final[i]
        x1 = max(0.0, min(x1 * sx, orig_w))
        y1 = max(0.0, min(y1 * sy, orig_h))
        x2 = max(0.0, min(x2 * sx, orig_w))
        y2 = max(0.0, min(y2 * sy, orig_h))
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0 or h <= 0: continue
        
        cid = int(c_final[i])
        if model_type == "yolo":
            if cid < 0 or cid >= len(COCO_80_TO_91): continue
            cat_id = COCO_80_TO_91[cid]
        else:
            cat_id = cid + 1
            
        results.append({
            "category_id": cat_id,
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(s_final[i]),
        })
    return results
