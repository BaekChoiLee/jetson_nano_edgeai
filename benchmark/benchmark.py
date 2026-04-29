#!/usr/bin/env python3
# ============================================================================
# benchmark.py - 메인 벤치마크 오케스트레이터
# ============================================================================
# 역할: 모든 런타임(PyTorch, TensorRT, ONNX Runtime, TFLite, ncnn)의
#       벤치마크를 조율하는 중앙 제어 파일.
#
# 동작 흐름:
#   1. CLI 인자 파싱 (모델, 런타임, 전력 모드, 반복 횟수 등)
#   2. ALL_RUNTIMES 리스트에서 지정된 런타임을 순차적으로 실행
#   3. 각 런타임 실행 전후로 tegrastats 로깅 (GPU/CPU/전력 모니터링)
#   4. 런타임 간 쿨다운(기본 60초)으로 열 안정화
#   5. 결과를 JSON + CSV 형식으로 저장
#
# 사용 예시:
#   python benchmark.py --model mobilenetv3_small --runtimes pytorch_cuda,tensorrt_fp16
#   python benchmark.py --model resnet50 --power-mode 5w --cool-down 30
# ============================================================================

"""Main benchmark orchestrator for all runtimes."""

import argparse
import os
import sys
import json
import csv
import time
from datetime import datetime

import numpy as np

# 상위 디렉토리를 sys.path에 추가하여 같은 폴더 내 모듈을 import할 수 있게 함
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# tegrastats 파서: Jetson 보드의 GPU/CPU/전력 사용량을 실시간 로깅하는 유틸리티
from tegra_parser import TegraStatsLogger


# ============================================================================
# 지원되는 전체 런타임 목록
# 각 런타임은 별도의 run_*.py 파일에서 실행됨
# ============================================================================
ALL_RUNTIMES = [
    "pytorch_cpu",       # PyTorch CPU 추론
    "pytorch_cuda",      # PyTorch CUDA(GPU) 추론
    "tensorrt_fp32",     # TensorRT FP32 정밀도 추론
    "tensorrt_fp16",     # TensorRT FP16 (반정밀도) 추론 - 속도/정확도 균형
    "tensorrt_int8",     # TensorRT INT8 (양자화) 추론 - 최고 속도, 정확도 손실 가능
    "onnxrt_cuda",       # ONNX Runtime + CUDA EP (GPU 가속)
    "onnxrt_trt",        # ONNX Runtime + TensorRT EP (TRT 백엔드 사용)
    "tflite_cpu",        # TensorFlow Lite CPU 추론
    "tflite_gpu",        # TensorFlow Lite GPU 위임(delegate) 추론
    "ncnn_cpu",          # ncnn CPU 추론 (Tencent의 경량 프레임워크)
    "ncnn_vulkan",       # ncnn Vulkan GPU 추론
]


def run_single_benchmark(model_name, runtime, model_dir, num_warmup, num_runs):
    """단일 런타임에 대한 벤치마크를 실행하는 디스패처 함수.

    런타임 이름에 따라 적절한 모듈을 동적으로 import하고 벤치마크를 수행한다.
    모델 파일이 존재하지 않는 경우 [SKIP]으로 건너뛴다.

    Args:
        model_name: 모델 이름 (예: "mobilenetv3_small", "resnet50")
        runtime: 실행할 런타임 이름 (ALL_RUNTIMES 중 하나)
        model_dir: 모델 파일들이 저장된 디렉토리 경로
        num_warmup: 워밍업 횟수 (벤치마크 측정에 포함되지 않음)
        num_runs: 실제 벤치마크 반복 횟수

    Returns:
        dict: 성공 시 지연시간 통계, 실패 시 error 필드를 포함한 딕셔너리.
    """
    print(f"\n  --- {runtime} ---")

    try:
        # --- PyTorch 런타임 ---
        if runtime == "pytorch_cpu":
            from run_pytorch import benchmark_pytorch
            return benchmark_pytorch(model_name, "cpu", num_warmup, num_runs,
                                     model_dir=model_dir)

        elif runtime == "pytorch_cuda":
            from run_pytorch import benchmark_pytorch
            return benchmark_pytorch(model_name, "cuda", num_warmup, num_runs,
                                     model_dir=model_dir)

        # --- TensorRT 런타임 ---
        # 런타임 이름에서 정밀도를 추출 (예: "tensorrt_fp16" -> "fp16")
        elif runtime.startswith("tensorrt_"):
            precision = runtime.split("_")[1]  # fp32, fp16, int8
            engine_path = os.path.join(model_dir, f"{model_name}_{precision}.engine")
            # .engine 파일이 없으면 건너뜀 (사전에 빌드 필요)
            if not os.path.exists(engine_path):
                # Detection models may have _raw in engine name (built from _raw.onnx)
                alt_engine = os.path.join(model_dir, f"{model_name}_raw_{precision}.engine")
                if os.path.exists(alt_engine):
                    engine_path = alt_engine
                else:
                    msg = f"engine not found: {engine_path}"
                    print(f"  [SKIP] {msg}")
                    return {"model": model_name, "runtime": runtime, "error": msg}
            from run_tensorrt import benchmark_tensorrt
            return benchmark_tensorrt(engine_path, num_warmup, num_runs)

        # --- ONNX Runtime 런타임 ---
        # onnxrt_cuda -> CUDAExecutionProvider, onnxrt_trt -> TensorrtExecutionProvider
        elif runtime == "onnxrt_cuda" or runtime == "onnxrt_trt":
            onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
            if not os.path.exists(onnx_path):
                # Detection models may use _raw_fpinput.onnx (float input) or _raw.onnx (uint8 input)
                # Prefer _fpinput variant for ORT compatibility (float32 input expected)
                alt_fpinput = os.path.join(model_dir, f"{model_name}_raw_fpinput.onnx")
                alt_raw = os.path.join(model_dir, f"{model_name}_raw.onnx")
                if os.path.exists(alt_fpinput):
                    onnx_path = alt_fpinput
                elif os.path.exists(alt_raw):
                    onnx_path = alt_raw
                else:
                    msg = f"onnx not found: {onnx_path}"
                    print(f"  [SKIP] {msg}")
                    return {"model": model_name, "runtime": runtime, "error": msg}
            from run_onnxrt import benchmark_onnxrt
            # 런타임 이름에 따라 Execution Provider 결정
            ep = "CUDAExecutionProvider" if runtime == "onnxrt_cuda" else "TensorrtExecutionProvider"
            return benchmark_onnxrt(onnx_path, ep, num_warmup, num_runs)

        # --- TFLite 런타임 ---
        elif runtime.startswith("tflite_"):
            tflite_path = os.path.join(model_dir, f"{model_name}.tflite")
            if not os.path.exists(tflite_path):
                msg = f"tflite not found: {tflite_path}"
                print(f"  [SKIP] {msg}")
                return {"model": model_name, "runtime": runtime, "error": msg}
            from run_tflite import benchmark_tflite
            use_gpu = (runtime == "tflite_gpu")  # GPU 위임 사용 여부
            return benchmark_tflite(tflite_path, num_warmup, num_runs, use_gpu=use_gpu)

        # --- ncnn 런타임 ---
        # ncnn은 .param(네트워크 구조) + .bin(가중치) 두 파일이 필요
        # 검출 모델(ultralytics export)은 <name>_ncnn_model/ 디렉토리에 저장됨
        elif runtime.startswith("ncnn_"):
            # 1순위: <name>_ncnn_model/model.param (ultralytics YOLOv8 export 형식)
            ncnn_subdir = os.path.join(model_dir, f"{model_name}_ncnn_model")
            alt_param = os.path.join(ncnn_subdir, "model.param")
            alt_bin   = os.path.join(ncnn_subdir, "model.bin")
            # 2순위: <name>.param / <name>.bin (convert_ncnn.sh 결과)
            param_path = os.path.join(model_dir, f"{model_name}.param")
            bin_path   = os.path.join(model_dir, f"{model_name}.bin")
            # 3순위: <name>.ncnn.param / <name>.ncnn.bin (onnx2ncnn 직접 변환 결과)
            ncnn_param = os.path.join(model_dir, f"{model_name}.ncnn.param")
            ncnn_bin   = os.path.join(model_dir, f"{model_name}.ncnn.bin")
            if os.path.exists(alt_param) and os.path.exists(alt_bin):
                param_path, bin_path = alt_param, alt_bin
            elif os.path.exists(param_path) and os.path.exists(bin_path):
                pass  # use default .param/.bin
            elif os.path.exists(ncnn_param) and os.path.exists(ncnn_bin):
                param_path, bin_path = ncnn_param, ncnn_bin
            else:
                msg = f"ncnn param not found: {param_path}, {ncnn_param}, or {alt_param}"
                print(f"  [SKIP] {msg}")
                return {"model": model_name, "runtime": runtime, "error": msg}
            from run_ncnn import benchmark_ncnn, get_input_size_for_model
            use_vulkan = (runtime == "ncnn_vulkan")  # Vulkan GPU 사용 여부
            input_size = get_input_size_for_model(model_name)
            return benchmark_ncnn(param_path, bin_path, num_warmup, num_runs,
                                   use_vulkan=use_vulkan, input_size=input_size)

        else:
            msg = f"unknown runtime: {runtime}"
            print(f"  [SKIP] {msg}")
            return {"model": model_name, "runtime": runtime, "error": msg}

    except Exception as e:
        # 특정 런타임이 실패해도 전체 벤치마크는 계속 진행
        msg = f"{runtime}: {e}"
        print(f"  [ERROR] {msg}")
        return {"model": model_name, "runtime": runtime, "error": msg}


def get_power_mode():
    """현재 Jetson 보드의 전력 모드를 조회한다.

    nvpmodel 명령어를 통해 현재 설정된 전력 모드(예: MAXN, 10W, 5W)를 반환한다.
    Jetson이 아닌 환경이나 sudo 권한이 없는 경우 "unknown"을 반환.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["sudo", "nvpmodel", "-q"],
            capture_output=True, text=True, timeout=5
        )
        # 출력에서 "NV Power Mode" 라인을 찾아 반환
        for line in result.stdout.split("\n"):
            if "NV Power Mode" in line:
                return line.strip()
        return "unknown"
    except Exception:
        return "unknown"


def main():
    """메인 함수: CLI 인자 파싱 -> 벤치마크 실행 -> 결과 저장.

    전체 실행 흐름:
      1. argparse로 CLI 인자 파싱
      2. 출력 디렉토리 생성 (타임스탬프 포함)
      3. 각 런타임에 대해 순차적으로:
         a) tegrastats 로깅 시작
         b) 벤치마크 실행
         c) tegrastats 로깅 종료 및 요약 추출
         d) 런타임 간 쿨다운 대기
      4. 전체 결과를 JSON/CSV로 저장
      5. 콘솔에 요약 테이블 출력
    """
    # --- CLI 인자 정의 ---
    parser = argparse.ArgumentParser(description="Jetson Nano Benchmark Suite")
    parser.add_argument(
        "--model", required=True,
        help="Model name (dynamic — must have corresponding model file in --model-dir; "
             "supported: mobilenetv3_small, resnet50, efficientnet_b0, shufflenet_v2_x1_0, "
             "yolov8n, ssd_mobilenet_v2)",
    )
    parser.add_argument(
        "--runtimes", default="all",
        help="Comma-separated runtimes or 'all'",
    )
    parser.add_argument("--power-mode", default="10w", choices=["10w", "5w"])
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--cool-down", type=int, default=0, help="Seconds between runtimes")
    parser.add_argument("--model-dir", default="./models", help="Dir with model files")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--no-tegrastats", action="store_true", help="Skip tegrastats logging")
    args = parser.parse_args()

    # output-dir은 디렉토리 경로만 허용한다.
    # .json 경로를 넘기면 과거처럼 <name>.json/ 하위에 results.json이 생겨
    # 빈 결과가 ok로 오인되는 문제가 생긴다.
    if args.output_dir.lower().endswith(".json"):
        print("[ERROR] --output-dir must be a directory path, not a .json file path")
        sys.exit(2)
    if os.path.exists(args.output_dir) and os.path.isfile(args.output_dir):
        print(f"[ERROR] --output-dir points to a file: {args.output_dir}")
        sys.exit(2)

    # --- 런타임 목록 파싱 및 검증 ---
    if args.runtimes == "all":
        runtimes = ALL_RUNTIMES
    else:
        # 쉼표로 분리하여 개별 런타임 이름 추출
        runtimes = [r.strip() for r in args.runtimes.split(",")]
        # 유효하지 않은 런타임 이름 검출
        invalid = [r for r in runtimes if r not in ALL_RUNTIMES]
        if invalid:
            print(f"Invalid runtimes: {invalid}")
            print(f"Available: {ALL_RUNTIMES}")
            sys.exit(1)

    # --- 출력 디렉토리 생성 ---
    # 형식: results/<모델명>_<전력모드>_<타임스탬프>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{args.model}_{args.power_mode}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 현재 Jetson 전력 모드 조회 (nvpmodel)
    power_mode_str = get_power_mode()

    # --- 벤치마크 설정 요약 출력 ---
    print("=" * 60)
    print(f"  Jetson Nano Benchmark")
    print(f"  Model:      {args.model}")
    print(f"  Power:      {args.power_mode} ({power_mode_str})")
    print(f"  Runtimes:   {', '.join(runtimes)}")
    print(f"  Warmup:     {args.num_warmup}")
    print(f"  Runs:       {args.num_runs}")
    print(f"  Cool-down:  {args.cool_down}s")
    print(f"  Output:     {run_dir}")
    print("=" * 60)

    all_results = []
    measured_count = 0
    failed_count = 0
    # tegrastats 로거 초기화 (--no-tegrastats 옵션으로 비활성화 가능)
    tegra_logger = TegraStatsLogger() if not args.no_tegrastats else None

    # ====================================================================
    # 메인 벤치마크 루프: 각 런타임을 순차적으로 실행
    # ====================================================================
    for i, runtime in enumerate(runtimes):
        print(f"\n[{i+1}/{len(runtimes)}] Benchmarking: {runtime}")

        # --- tegrastats 로깅 시작 ---
        # 100ms 간격으로 GPU/CPU/메모리/전력 사용량을 로그 파일에 기록
        tegra_log = os.path.join(run_dir, f"tegra_{runtime}.log")
        if tegra_logger:
            try:
                tegra_logger.start(tegra_log, interval_ms=100)
            except Exception as e:
                print(f"  [WARN] tegrastats failed: {e}")
                tegra_logger = None  # 실패 시 이후 런타임에서도 tegrastats 비활성화

        # --- 벤치마크 실행 ---
        start_time = time.time()
        result = run_single_benchmark(
            args.model, runtime, args.model_dir, args.num_warmup, args.num_runs
        )
        elapsed = time.time() - start_time  # 벽시계 시간 (워밍업 + 측정 전체)

        # --- tegrastats 로깅 종료 및 요약 추출 ---
        tegra_summary = {}
        if tegra_logger:
            tegra_logger.stop()
            # 로그 파일을 파싱하여 평균/최대 전력, GPU 사용률 등 요약 통계 생성
            tegra_summary = tegra_logger.summary(tegra_log)

        if result is None:
            result = {
                "model": args.model,
                "runtime": runtime,
                "error": "runtime returned no result",
            }

        result["model"] = result.get("model", args.model)
        result["runtime"] = result.get("runtime", runtime)
        result["power_mode"] = args.power_mode
        result["tegrastats"] = tegra_summary
        result["wall_time_s"] = round(elapsed, 2)  # 전체 소요 시간(초)

        # latency 유효성 검사: status=ok인데 latency 없는 케이스를 차단
        latency = result.get("latency") if isinstance(result, dict) else None
        mean_ms = None
        if isinstance(latency, dict):
            try:
                mean_ms = float(latency.get("mean"))
            except (TypeError, ValueError):
                mean_ms = None

        if result.get("error"):
            result["status"] = "error"
            failed_count += 1
            print(f"  [ERROR] {result.get('error')}")
        elif mean_ms is None or mean_ms <= 0:
            result["status"] = "error"
            result["error"] = "missing_or_nonpositive_latency"
            failed_count += 1
            print("  [ERROR] missing_or_nonpositive_latency")
        else:
            result["status"] = "ok"
            measured_count += 1
            lat = result["latency"]
            print(f"  Mean: {lat['mean']:.2f} ms | P95: {lat['p95']:.2f} ms | Std: {lat['std']:.2f} ms")
            # tegrastats에서 전력 데이터가 있으면 함께 출력
            if tegra_summary and "pom_5v_in_current_mw" in tegra_summary:
                pwr = tegra_summary["pom_5v_in_current_mw"]
                print(f"  Power: avg {pwr['mean']:.0f} mW, max {pwr['max']:.0f} mW")

        all_results.append(result)

        # --- 런타임 간 쿨다운 ---
        # 마지막 런타임 이후에는 쿨다운하지 않음
        # 쿨다운의 목적: 열 쓰로틀링 방지를 위해 칩 온도를 안정화
        if i < len(runtimes) - 1 and args.cool_down > 0:
            print(f"  Cooling down for {args.cool_down}s...")
            time.sleep(args.cool_down)

    # ====================================================================
    # 결과 저장: JSON (상세 데이터)
    # ====================================================================
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  JSON saved: {json_path}")

    # ====================================================================
    # 결과 저장: CSV (요약 테이블 - 스프레드시트 분석용)
    # ====================================================================
    csv_path = os.path.join(run_dir, "summary.csv")
    if all_results:
        csv_rows = []
        for r in all_results:
            lat = r.get("latency", {}) if isinstance(r.get("latency"), dict) else {}
            mean = lat.get("mean")
            fps = None
            try:
                if mean is not None and float(mean) > 0:
                    fps = round(1000.0 / float(mean), 2)
            except (TypeError, ValueError):
                fps = None

            row = {
                "model": r.get("model", args.model),
                "runtime": r.get("runtime", ""),
                "power_mode": r.get("power_mode", ""),
                "status": r.get("status", "error"),
                "error": r.get("error", ""),
                "mean_ms": _safe_float(lat.get("mean")),
                "std_ms": _safe_float(lat.get("std")),
                "min_ms": _safe_float(lat.get("min")),
                "max_ms": _safe_float(lat.get("max")),
                "p50_ms": _safe_float(lat.get("p50")),
                "p95_ms": _safe_float(lat.get("p95")),
                "p99_ms": _safe_float(lat.get("p99")),
                "memory_mb": _safe_float(r.get("memory_mb")),
                "model_size_mb": _safe_float(r.get("model_size_mb")),
                "fps": fps,
            }
            # tegrastats에서 전력 데이터가 있으면 CSV에도 추가
            tegra = r.get("tegrastats", {})
            if "pom_5v_in_current_mw" in tegra:
                row["power_avg_mw"] = _safe_float(tegra["pom_5v_in_current_mw"].get("mean"))
                row["power_max_mw"] = _safe_float(tegra["pom_5v_in_current_mw"].get("max"))
            csv_rows.append(row)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  CSV saved:  {csv_path}")

    # ====================================================================
    # 콘솔 최종 요약 테이블 출력
    # 모든 런타임의 핵심 지표를 한눈에 비교할 수 있도록 정렬하여 표시
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"  Summary: {args.model} @ {args.power_mode}")
    print(f"{'='*70}")
    print(f"  {'Runtime':<20} {'Mean(ms)':<10} {'P95(ms)':<10} {'FPS':<10} {'Mem(MB)'}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    measured_results = [
        r for r in all_results
        if r.get("status") == "ok" and isinstance(r.get("latency"), dict)
    ]
    for r in measured_results:
        lat = r["latency"]
        fps = 1000.0 / lat["mean"] if lat["mean"] > 0 else 0
        print(
            f"  {r['runtime']:<20} "
            f"{lat['mean']:<10.2f} "
            f"{lat['p95']:<10.2f} "
            f"{fps:<10.1f} "
            f"{r.get('memory_mb', 0)}"
        )
    print()

    print(f"  measured: {measured_count} | failed: {failed_count}")
    if measured_count == 0:
        print("  [ERROR] no measured runtime results")
        sys.exit(2)


def _safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
