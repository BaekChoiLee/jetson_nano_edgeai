#!/usr/bin/env python3
# =============================================================================
# run_detection_accuracy.py — 검출 모델 11 런타임 COCO mAP CLI
# =============================================================================
# 역할: detection_eval.eval_detection() 을 11 런타임에 대해 일괄 호출하고
#       JSON 으로 저장. 단일 모델(yolov8n 또는 ssd_mobilenet_v2) 단위 실행.
#
# 출력 스키마:
#   results/detection_accuracy_<model>.json — list of dict
#   stdout: 사람이 읽기 쉬운 요약 라인
#
# 사용법:
#   # 모든 11 런타임
#   python benchmark/run_detection_accuracy.py --model yolov8n
#
#   # 일부 런타임만 (디버깅)
#   python benchmark/run_detection_accuracy.py --model ssd_mobilenet_v2 \
#     --runtimes pytorch_cpu onnxrt_cuda
#
#   # 100장으로 dry run
#   python benchmark/run_detection_accuracy.py --model yolov8n --max-images 100
# =============================================================================

"""CLI: COCO mAP for detection models across 11 runtimes."""

import argparse
import json
import os
import sys
import multiprocessing as mp

# sibling import path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# 11 런타임 (벤치마크 메인 ALL_RUNTIMES 와 동일 순서)
ALL_DETECTION_RUNTIMES = [
    "pytorch_cpu",
    "pytorch_cuda",
    "tensorrt_fp32",
    "tensorrt_fp16",
    "tensorrt_int8",
    "onnxrt_cuda",
    "onnxrt_trt",
    "tflite_cpu",
    "tflite_gpu",
    "ncnn_cpu",
    "ncnn_vulkan",
]


def _eval_detection_worker(kwargs, out_q):
    """Run eval_detection in a child process to enforce hard timeout."""
    try:
        # child process import to avoid side effects in parent
        from detection_eval import eval_detection
        res = eval_detection(**kwargs)
        out_q.put({"ok": True, "result": res})
    except Exception as e:
        out_q.put({"ok": False, "error": str(e)})


def _resolve_output_path(output_arg, model):
    """Return a concrete JSON output file path."""
    output_path = output_arg or os.path.join(
        "results", f"detection_accuracy_{model}.json"
    )
    if os.path.isdir(output_path) or output_path.endswith((os.sep, "/")):
        output_path = os.path.join(
            output_path, f"detection_accuracy_{model}.json"
        )
    return output_path


def _preflight_coco_inputs(coco_dir, annotations_json):
    """Fail early when the COCO image root does not match annotation file names."""
    if not os.path.isdir(coco_dir):
        return f"COCO image directory not found: {coco_dir}"
    if not os.path.exists(annotations_json):
        return f"annotations not found: {annotations_json}"

    try:
        with open(annotations_json) as f:
            data = json.load(f)
    except Exception as e:
        return f"annotations load failed: {e}"

    images = data.get("images") or []
    if not images:
        return f"no images listed in annotations: {annotations_json}"

    first_name = images[0].get("file_name")
    if not first_name:
        return "first annotation image has no file_name"

    first_path = os.path.join(coco_dir, first_name)
    if not os.path.exists(first_path):
        return (
            f"first annotation image not found under --coco-dir: {first_path}. "
            "Pass the actual image root, for example data/coco_val/images."
        )

    try:
        import cv2
        if cv2.imread(first_path) is None:
            return f"first annotation image could not be read by cv2: {first_path}"
    except ImportError:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Detection accuracy sweep across 11 runtimes"
    )
    parser.add_argument(
        "--model",
        choices=["yolov8n", "ssd_mobilenet_v2"],
        required=True,
        help="Detection model name",
    )
    parser.add_argument(
        "--runtimes",
        nargs="+",
        default=ALL_DETECTION_RUNTIMES,
        help=f"Runtimes to evaluate (default: all 11). "
             f"Choices: {ALL_DETECTION_RUNTIMES}",
    )
    parser.add_argument(
        "--coco-dir",
        default="data/coco_val/images",
        help="COCO val image root directory",
    )
    parser.add_argument(
        "--annotations",
        default="data/coco_val/annotations/instances_val2017_subset500.json",
        help="COCO subset annotation json",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Model artifact directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: results/detection_accuracy_<model>.json)",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images (debug). None = use all in subset.",
    )
    parser.add_argument(
        "--per-runtime-timeout-sec",
        type=int,
        default=1800,
        help="Hard timeout for each runtime eval (seconds). 0 disables timeout.",
    )
    args = parser.parse_args()

    output_path = _resolve_output_path(args.output, args.model)

    preflight_error = _preflight_coco_inputs(args.coco_dir, args.annotations)
    if preflight_error:
        print(f"[ERROR] {preflight_error}", file=sys.stderr)
        sys.exit(2)

    print(f"=" * 70)
    print(f"Detection accuracy sweep")
    print(f"  model:       {args.model}")
    print(f"  runtimes:    {len(args.runtimes)} ({args.runtimes})")
    print(f"  annotations: {args.annotations}")
    print(f"  coco-dir:    {args.coco_dir}")
    print(f"  conf/iou:    {args.conf} / {args.iou}")
    print(f"  max-images:  {args.max_images or 'all'}")
    print(f"=" * 70)

    all_results = []
    for rt in args.runtimes:
        print(f"\n--- {rt} ---")
        kwargs = {
            "model_name": args.model,
            "runtime": rt,
            "coco_val_dir": args.coco_dir,
            "annotations_json": args.annotations,
            "model_dir": args.model_dir,
            "conf": args.conf,
            "iou": args.iou,
            "max_images": args.max_images,
        }
        r = None
        timeout = int(args.per_runtime_timeout_sec or 0)
        if timeout > 0:
            q = mp.Queue()
            p = mp.Process(target=_eval_detection_worker, args=(kwargs, q), daemon=True)
            p.start()
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                r = {
                    "runtime": rt,
                    "model": args.model,
                    "status": "error",
                    "error": f"timeout after {timeout}s",
                }
            else:
                if q.empty():
                    r = {
                        "runtime": rt,
                        "model": args.model,
                        "status": "error",
                        "error": "worker exited without result",
                    }
                else:
                    msg = q.get()
                    if msg.get("ok"):
                        r = msg["result"]
                    else:
                        r = {
                            "runtime": rt,
                            "model": args.model,
                            "status": "error",
                            "error": f"unhandled exception: {msg.get('error')}",
                        }
        else:
            try:
                from detection_eval import eval_detection
                r = eval_detection(**kwargs)
            except Exception as e:
                r = {
                    "runtime": rt,
                    "model": args.model,
                    "status": "error",
                    "error": f"unhandled exception: {e}",
                }

        if "status" not in r:
            r["status"] = "error" if r.get("error") else "ok"

        if "error" in r and r.get("map_50") in (None, 0.0):
            print(f"  [FAIL] {r['error']}")
        else:
            print(
                f"  [OK] mAP@0.5 = {r['map_50']:.4f}  "
                f"mAP@[.5:.95] = {r['map_50_95']:.4f}  "
                f"n={r.get('n', 0)}  "
                f"latency={r.get('latency_ms_mean', 0):.2f} ms"
            )
        all_results.append(r)

    # 저장
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Saved {len(all_results)} runtime results → {output_path}")


if __name__ == "__main__":
    main()
