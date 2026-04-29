#!/usr/bin/env python3
# =============================================================================
# run_detection_latency.py — 검출 모델 12 런타임 지연 측정 CLI
# =============================================================================
# 역할: detection_infer.get_detection_infer_fn() 으로 raw infer 함수를 받아
#       워밍업 + 본 측정 루프를 돌리고 latency 통계를 생성한다.
#       전·후처리(Letterbox/NMS) 비용은 제외하여 모델 순전파 자체를 비교.
#
# 기존 분류 벤치마크 러너(run_pytorch.py 등)는 224×224 가정이라 그대로 못 쓴다.
# 이 파일은 640(YOLOv8n) / 320(SSD-MV2) 동적 입력을 지원하며,
# tegrastats 전력 로그도 옵션으로 함께 수집한다.
#
# 출력 스키마:
#   {
#     "runtime": str, "model": str,
#     "latency": {"mean", "std", "min", "max", "p50", "p95", "p99"},
#     "memory_mb": float, "model_size_mb": float,
#     "num_runs": int, "num_warmup": int,
#     "tegra_log": str?
#   }
#
# 사용법:
#   python benchmark/run_detection_latency.py \
#     --model yolov8n --runtime tensorrt_fp16 --num-warmup 20 --num-runs 200
# =============================================================================

"""Detection model latency benchmark for 12 runtimes."""

import argparse
import json
import os
import sys
import time

import numpy as np

# sibling import path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def benchmark_detection(model_name, runtime, num_warmup=10, num_runs=100,
                        model_dir="models", tegra_log=None):
    """단일 (model, runtime) 지연 벤치마크.

    Returns:
        결과 dict (latency 통계 포함). 실패 시 'error' 키.
    """
    from detection_infer import get_detection_infer_fn, get_input_hw

    # 1) infer fn 생성
    try:
        infer_fn = get_detection_infer_fn(model_name, runtime, model_dir)
    except Exception as e:
        return {
            "runtime": runtime, "model": model_name,
            "status": "error",
            "error": f"factory failed: {e}",
        }

    # 2) 더미 입력 (random NCHW)
    h, w = get_input_hw(model_name)
    dummy = np.random.rand(1, 3, h, w).astype(np.float32)

    # 3) tegrastats 로깅 시작 (옵션)
    tegra_logger = None
    tegra_summary = {}
    if tegra_log:
        try:
            from tegra_parser import TegraStatsLogger
            tegra_logger = TegraStatsLogger()
            tegra_logger.start(tegra_log, interval_ms=100)
        except Exception as e:
            print(f"[WARN] tegrastats logger init failed: {e}")
            tegra_logger = None

    # 4) 워밍업
    try:
        for _ in range(num_warmup):
            _ = infer_fn(dummy)
    except Exception as e:
        if tegra_logger:
            tegra_logger.stop()
            tegra_summary = tegra_logger.summary(tegra_log)
        return {
            "runtime": runtime, "model": model_name,
            "status": "error",
            "error": f"warmup failed: {e}",
            "tegrastats": tegra_summary,
        }

    # 5) 본 측정
    latencies = []
    try:
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = infer_fn(dummy)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    except Exception as e:
        if tegra_logger:
            tegra_logger.stop()
            tegra_summary = tegra_logger.summary(tegra_log)
        return {
            "runtime": runtime, "model": model_name,
            "status": "error",
            "error": f"benchmark loop failed: {e}",
            "tegrastats": tegra_summary,
        }

    if tegra_logger:
        tegra_logger.stop()
        tegra_summary = tegra_logger.summary(tegra_log)

    latencies = np.array(latencies)

    # 6) 모델 사이즈 (artifact 1순위 추정)
    model_size_mb = _estimate_model_size_mb(model_name, runtime, model_dir)

    ram_info = tegra_summary.get("ram_used_mb", {})
    ram_delta_mb = 0.0
    if isinstance(ram_info, dict):
        ram_delta_mb = max(0.0, float(ram_info.get("max", 0)) - float(ram_info.get("min", 0)))

    return {
        "runtime": runtime,
        "model": model_name,
        "status": "ok",
        "latency": {
            "mean": float(np.mean(latencies)),
            "std":  float(np.std(latencies)),
            "min":  float(np.min(latencies)),
            "max":  float(np.max(latencies)),
            "p50":  float(np.percentile(latencies, 50)),
            "p95":  float(np.percentile(latencies, 95)),
            "p99":  float(np.percentile(latencies, 99)),
        },
        "memory_mb": round(ram_delta_mb, 2),
        "model_size_mb": round(model_size_mb, 2),
        "num_runs": num_runs,
        "num_warmup": num_warmup,
        "input_hw": [h, w],
        "tegra_log": tegra_log,
        "tegrastats": tegra_summary,
    }


def _estimate_model_size_mb(model_name, runtime, model_dir):
    """런타임별 아티팩트 파일 크기 추정."""
    from detection_infer import (
        _resolve_ncnn_path,
        _resolve_onnx_path,
        _resolve_tflite_path,
        _resolve_trt_engine_path,
    )

    candidates = []
    if runtime in ("pytorch_cpu", "pytorch_cuda"):
        candidates.append(os.path.join(model_dir, f"{model_name}.torchscript"))
    elif runtime in ("tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8"):
        prec = runtime.split("_")[-1]
        candidates.append(_resolve_trt_engine_path(model_name, model_dir, prec))
    elif runtime in ("onnxrt_cuda", "onnxrt_trt"):
        candidates.append(_resolve_onnx_path(model_name, model_dir))
    elif runtime in ("tflite_cpu", "tflite_gpu"):
        candidates.append(_resolve_tflite_path(model_name, model_dir))
    elif runtime in ("ncnn_cpu", "ncnn_vulkan", "ncnn_vulkan_fixed"):
        _, ncnn_bin = _resolve_ncnn_path(model_name, model_dir)
        candidates.append(ncnn_bin)
        candidates.append(os.path.join(model_dir, f"{model_name}_ncnn_model", "model.bin"))
        candidates.append(os.path.join(model_dir, f"{model_name}.bin"))

    for c in candidates:
        if os.path.exists(c):
            return os.path.getsize(c) / (1024 * 1024)
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Detection latency benchmark (single model+runtime)"
    )
    parser.add_argument(
        "--model",
        choices=["yolov8n", "ssd_mobilenet_v2"],
        required=True,
    )
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument(
        "--tegra-log",
        default=None,
        help="Optional tegrastats log path",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = benchmark_detection(
        model_name=args.model,
        runtime=args.runtime,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        model_dir=args.model_dir,
        tegra_log=args.tegra_log,
    )

    print(json.dumps(result, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
