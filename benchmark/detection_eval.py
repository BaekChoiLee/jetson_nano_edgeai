#!/usr/bin/env python3
# =============================================================================
# detection_eval.py — 검출 모델 12 런타임 공통 COCO mAP 평가 코어
# =============================================================================
# 역할: `detection_infer.get_detection_infer_fn()` 로 얻은 raw infer 함수와
#       `detection_postprocess.get_postprocess_fn()` 로 얻은 NMS 함수를 결합하여
#       pycocotools 기반 mAP@0.5 / mAP@0.5:0.95 를 계산한다.
#
# 입력:
#   - model_name: 'yolov8n' | 'ssd_mobilenet_v2'
#   - runtime: 12 런타임 중 하나
#   - coco_val_dir: COCO val2017 이미지 디렉토리
#   - annotations_json: subset(seed 42, 500장) annotation json
#
# 출력:
#   {
#     "runtime": str, "model": str,
#     "map_50": float, "map_50_95": float,
#     "n": int, "latency_ms_mean": float (infer only),
#     "error": str? (실패 시)
#   }
#
# 설계 원칙:
#   - stdin/stdout 없이 순수 함수로 재사용 가능
#   - 이미지당 infer 시간 측정 (검출 지연 보조 지표)
#   - 실패한 런타임은 error 필드 채워서 반환 (파이프라인 중단 X)
# =============================================================================

"""COCO mAP evaluation for detection models across 12 runtimes."""

import os
import sys
import time

import numpy as np

# 현재 디렉토리를 import path 에 추가 (sibling 모듈 접근)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def eval_detection(model_name, runtime, coco_val_dir, annotations_json,
                   model_dir="models", conf=0.25, iou=0.45, max_images=None):
    """단일 (model, runtime) 조합에 대한 COCO mAP 평가.

    Args:
        model_name: 'yolov8n' | 'ssd_mobilenet_v2'
        runtime: 12 런타임 중 하나
        coco_val_dir: 'data/coco_val/images' 등 val 이미지 루트
        annotations_json: subset annotation 경로
        model_dir: 모델 아티팩트 루트
        conf/iou: 후처리 threshold
        max_images: 디버그용 샘플 수 제한 (None 이면 annotation 의 모든 img_id)

    Returns:
        결과 dict. 실패 시 'error' 키 포함.
    """
    # pycocotools 임포트 (lazy: 미설치 환경에서도 파일 import 는 가능)
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        return {
            "runtime": runtime, "model": model_name,
            "map_50": None, "map_50_95": None, "n": 0,
            "status": "error",
            "error": f"pycocotools not installed: {e}",
        }

    # sibling module import
    try:
        from detection_infer import get_detection_infer_fn, preprocess_image
        from detection_postprocess import get_postprocess_fn
    except ImportError as e:
        return {
            "runtime": runtime, "model": model_name,
            "status": "error",
            "error": f"sibling import failed: {e}",
        }

    if not os.path.exists(annotations_json):
        return {
            "runtime": runtime, "model": model_name,
            "status": "error",
            "error": f"annotations not found: {annotations_json}",
        }

    # 1) COCO GT 로드
    try:
        coco_gt = COCO(annotations_json)
    except Exception as e:
        return {"runtime": runtime, "model": model_name,
                "status": "error",
                "error": f"COCO load failed: {e}"}

    img_ids = sorted(coco_gt.getImgIds())
    if max_images:
        img_ids = img_ids[:max_images]
    if not img_ids:
        return {"runtime": runtime, "model": model_name,
                "status": "error",
                "error": "no image ids in subset"}

    # 2) infer + postprocess 함수 생성
    try:
        infer_fn = get_detection_infer_fn(model_name, runtime, model_dir)
        post_fn = get_postprocess_fn(model_name)
    except Exception as e:
        return {"runtime": runtime, "model": model_name,
                "status": "error",
                "error": f"factory failed: {e}"}

    # 3) 이미지 순회
    try:
        import cv2
    except ImportError as e:
        return {"runtime": runtime, "model": model_name,
                "status": "error",
                "error": f"opencv-python required: {e}"}

    predictions = []
    latencies = []
    n_ok = 0
    n_fail = 0
    missing_images = 0
    unreadable_images = 0
    infer_failures = 0
    postprocess_failures = 0

    for img_id in img_ids:
        img_info = coco_gt.loadImgs([img_id])[0]
        fname = img_info["file_name"]
        img_path = os.path.join(coco_val_dir, fname)
        if not os.path.exists(img_path):
            missing_images += 1
            n_fail += 1
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            unreadable_images += 1
            n_fail += 1
            continue

        orig_h, orig_w = bgr.shape[:2]
        input_nchw = preprocess_image(bgr, model_name)

        # infer 시간 측정 (전처리·후처리 제외)
        t0 = time.perf_counter()
        try:
            raw = infer_fn(input_nchw)
        except Exception as e:
            infer_failures += 1
            n_fail += 1
            if n_fail < 5:
                print(f"  [warn] infer failed on img {img_id}: {e}")
            continue
        latencies.append((time.perf_counter() - t0) * 1000.0)

        try:
            dets = post_fn(raw, (orig_w, orig_h), conf=conf, iou=iou)
        except Exception as e:
            postprocess_failures += 1
            n_fail += 1
            if n_fail < 5:
                print(f"  [warn] postprocess failed on img {img_id}: {e}")
            continue

        for d in dets:
            predictions.append({
                "image_id": int(img_id),
                "category_id": int(d["category_id"]),
                "bbox": [float(x) for x in d["bbox"]],
                "score": float(d["score"]),
            })
        n_ok += 1

    if not predictions:
        if missing_images == len(img_ids):
            error = (
                f"all annotation images missing under coco_val_dir={coco_val_dir} "
                f"(checked={len(img_ids)})"
            )
        elif n_ok == 0:
            error = (
                f"no images evaluated (missing_images={missing_images}, "
                f"unreadable_images={unreadable_images}, "
                f"infer_failures={infer_failures}, "
                f"postprocess_failures={postprocess_failures})"
            )
        else:
            error = (
                f"no detections (n_ok={n_ok}, n_fail={n_fail}, "
                f"missing_images={missing_images}, "
                f"unreadable_images={unreadable_images}, "
                f"infer_failures={infer_failures}, "
                f"postprocess_failures={postprocess_failures})"
            )
        return {
            "runtime": runtime, "model": model_name,
            "map_50": 0.0, "map_50_95": 0.0, "n": n_ok,
            "latency_ms_mean": float(np.mean(latencies)) if latencies else 0.0,
            "n_fail": n_fail,
            "n_predictions": 0,
            "missing_images": missing_images,
            "unreadable_images": unreadable_images,
            "infer_failures": infer_failures,
            "postprocess_failures": postprocess_failures,
            "status": "error",
            "error": error,
        }

    # 4) COCOeval
    try:
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = list(img_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # stats[0] = AP@[0.5:0.95], stats[1] = AP@0.50
        map_50_95 = float(coco_eval.stats[0])
        map_50 = float(coco_eval.stats[1])
    except Exception as e:
        return {
            "runtime": runtime, "model": model_name,
            "map_50": 0.0, "map_50_95": 0.0, "n": n_ok,
            "n_fail": n_fail,
            "n_predictions": len(predictions),
            "missing_images": missing_images,
            "unreadable_images": unreadable_images,
            "infer_failures": infer_failures,
            "postprocess_failures": postprocess_failures,
            "status": "error",
            "error": f"COCOeval failed: {e}",
        }

    return {
        "runtime": runtime,
        "model": model_name,
        "status": "ok",
        "map_50": round(map_50, 4),
        "map_50_95": round(map_50_95, 4),
        "n": n_ok,
        "n_fail": n_fail,
        "latency_ms_mean": round(float(np.mean(latencies)), 3) if latencies else 0.0,
        "n_predictions": len(predictions),
        "missing_images": missing_images,
        "unreadable_images": unreadable_images,
        "infer_failures": infer_failures,
        "postprocess_failures": postprocess_failures,
    }
