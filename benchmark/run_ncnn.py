#!/usr/bin/env python3
# =============================================================================
# run_ncnn.py - ncnn 추론 벤치마크
# =============================================================================
# 역할: Tencent ncnn 프레임워크를 사용한 모델 추론 성능 측정.
#
# 두 가지 벤치마크 모드 지원:
#   1. Python 바인딩 모드 (ncnn 모듈): 직접 Python에서 추론 실행
#   2. 서브프로세스 모드 (benchncnn 바이너리): C++ 바이너리를 호출하여 측정
#
# 우선순위: Python 바인딩 우선 시도 → 없으면 benchncnn 서브프로세스 폴백
#
# ncnn 모델 형식:
#   - .param 파일: 네트워크 구조 정의 (텍스트)
#   - .bin 파일: 가중치 데이터 (바이너리)
#   - 두 파일이 같은 디렉토리에 같은 이름으로 존재해야 함
#
# Vulkan: GPU 가속을 위한 그래픽 API. use_vulkan=True면 GPU 사용.
# =============================================================================

"""ncnn inference benchmark (via subprocess or Python binding)."""

import time
import os
import re
import subprocess
import numpy as np


# ============================================================================
# 모델별 입력 크기 매핑 (C, H, W)
# 검출 모델은 분류보다 큰 입력이 필요하므로 하드코딩된 224 대신 동적으로 선택한다.
# 새 모델 추가 시 이 딕셔너리에 항목을 추가하기만 하면 된다.
# ============================================================================
_NCNN_INPUT_SIZES = {
    "yolov8n":          (3, 640, 640),
    "ssd_mobilenet_v2": (3, 320, 320),
}

# 입출력 레이어 이름 매핑 (ultralytics/ncnn 공식 export 결과 기준)
# 값: (input_name, output_name)
_NCNN_IO_NAMES = {
    "yolov8n": ("in0", "out0"),  # ultralytics export: images -> output
}


def get_input_size_for_model(model_name):
    """모델 이름에서 ncnn 입력 크기를 반환한다.

    분류 모델(ImageNet)은 224x224 기본값, 검출 모델은 _NCNN_INPUT_SIZES 참조.
    """
    return _NCNN_INPUT_SIZES.get(model_name, (3, 224, 224))


def _find_benchncnn():
    """Locate benchncnn binary."""
    # benchncnn: ncnn에 포함된 공식 벤치마크 바이너리 (C++로 작성됨)
    import shutil

    # 1단계: 시스템 PATH에서 benchncnn 검색
    path = shutil.which("benchncnn")
    if path:
        return path

    # 2단계: 일반적인 설치 경로에서 검색
    # NCNN_DIR 환경변수가 설정되어 있으면 해당 경로, 아니면 ~/ncnn 기본값
    ncnn_dir = os.environ.get("NCNN_DIR", os.path.expanduser("~/ncnn"))
    candidate = os.path.join(ncnn_dir, "build", "benchmark", "benchncnn")
    if os.path.exists(candidate):
        return candidate

    # benchncnn을 찾지 못한 경우
    return None


def benchmark_ncnn_subprocess(param_path, num_warmup=10, num_runs=100, num_threads=4, gpu_index=0):
    """Benchmark ncnn using benchncnn binary.

    Returns dict with latency stats.
    """
    # benchncnn 바이너리 위치 확인
    benchncnn = _find_benchncnn()
    if not benchncnn:
        raise FileNotFoundError(
            "benchncnn not found. Set NCNN_DIR or add to PATH."
        )

    # 모델 이름 추출 (경로와 확장자 제거)
    model_name = os.path.splitext(os.path.basename(param_path))[0]
    # benchncnn은 .param과 .bin을 같은 디렉토리에서 찾으므로 작업 디렉토리 설정 필요
    param_dir = os.path.dirname(os.path.abspath(param_path))

    # benchncnn 명령어 구성
    # 인자 순서: <param파일> <총실행횟수> <스레드수> <GPU인덱스>
    # 주의: benchncnn은 워밍업과 실행을 합산한 총 횟수를 받음
    cmd = [
        benchncnn,
        param_path,
        str(num_warmup + num_runs),  # 워밍업 + 실제 측정 횟수의 합
        str(num_threads),
        str(gpu_index),              # -1이면 CPU, 0 이상이면 해당 GPU 사용
    ]

    # 서브프로세스로 benchncnn 실행 (최대 300초 타임아웃)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, cwd=param_dir
    )

    if result.returncode != 0:
        raise RuntimeError(f"benchncnn failed: {result.stderr}")

    # benchncnn 출력 파싱
    # 출력 형식 예시: "mobilenetv3  min = 12.34  max = 15.67  avg = 13.45"
    latencies_from_output = []
    for line in result.stdout.strip().split("\n"):
        match = re.search(r"min\s*=\s*([\d.]+)\s+max\s*=\s*([\d.]+)\s+avg\s*=\s*([\d.]+)", line)
        if match:
            latencies_from_output.append({
                "min": float(match.group(1)),
                "max": float(match.group(2)),
                "avg": float(match.group(3)),
            })

    if not latencies_from_output:
        raise RuntimeError(f"Could not parse benchncnn output:\n{result.stdout}")

    # 마지막 줄이 최종 요약 통계 (여러 줄 출력될 수 있으므로 마지막 것 사용)
    summary = latencies_from_output[-1]

    # 모델 가중치 파일(.bin) 크기 계산
    bin_path = param_path.replace(".param", ".bin")
    model_size_mb = 0.0
    if os.path.exists(bin_path):
        model_size_mb = os.path.getsize(bin_path) / (1024 * 1024)

    # 결과 딕셔너리 반환
    # 주의: benchncnn은 std, p50, p95, p99를 직접 보고하지 않으므로 근사값 사용
    return {
        "runtime": f"ncnn_gpu{gpu_index}" if gpu_index >= 0 else "ncnn_cpu",
        "model": model_name,
        "latency": {
            "mean": summary["avg"],
            "std": 0.0,              # benchncnn은 표준편차를 제공하지 않음
            "min": summary["min"],
            "max": summary["max"],
            "p50": summary["avg"],   # 중앙값 근사: 평균으로 대체
            "p95": summary["max"],   # 95번째 백분위 근사: 최대값으로 대체
            "p99": summary["max"],   # 99번째 백분위 근사: 최대값으로 대체
        },
        "memory_mb": 0.0,
        "model_size_mb": round(model_size_mb, 2),
        "num_runs": num_runs,
        "num_threads": num_threads,
        "gpu_index": gpu_index,
        "raw_output": result.stdout.strip(),  # 원본 출력 보존 (디버깅용)
    }


def benchmark_ncnn_python(param_path, bin_path, num_warmup=10, num_runs=100,
                           use_vulkan=False, input_size=(3, 224, 224),
                           input_name=None, output_name=None):
    """Benchmark ncnn using Python binding (if available).

    Args:
        param_path: .param 파일 경로
        bin_path: .bin 파일 경로
        num_warmup/num_runs: 측정 반복 횟수
        use_vulkan: Vulkan GPU 가속 사용 여부
        input_size: (C, H, W) 입력 텐서 크기 — 하드코딩 224 해제
        input_name/output_name: None이면 net.input_names()/output_names() 자동 감지
    """
    # ncnn Python 바인딩 임포트 (pyncnn 패키지 필요)
    import ncnn

    # ncnn 네트워크 생성 및 설정
    net = ncnn.Net()
    net.opt.use_vulkan_compute = use_vulkan  # Vulkan GPU 가속 활성화 여부
    net.load_param(param_path)  # 네트워크 구조 로드
    net.load_model(bin_path)    # 가중치 로드

    # 입/출력 레이어 이름 자동 감지 (하드코딩된 "input"/"output" 제거)
    # 분류 모델은 보통 "input"/"output"이지만 YOLOv8n ncnn export는 "in0"/"out0"
    if input_name is None or output_name is None:
        try:
            in_names = list(net.input_names())
            out_names = list(net.output_names())
            auto_in = in_names[0] if in_names else "input"
            auto_out = out_names[0] if out_names else "output"
        except Exception:
            auto_in, auto_out = "input", "output"
        input_name = input_name or auto_in
        output_name = output_name or auto_out

    # 입력 Mat 생성: (C, H, W) → ncnn.Mat(w, h, c) 순서 주의
    c, h, w = input_size
    mat_in = ncnn.Mat(w, h, c)

    import psutil
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024 * 1024)

    # 워밍업: GPU 셰이더 컴파일, 메모리 할당 등을 미리 수행
    for _ in range(num_warmup):
        ex = net.create_extractor()            # 추론 세션(extractor) 생성
        ex.input(input_name, mat_in)           # 입력 레이어에 데이터 전달
        _, _ = ex.extract(output_name)         # 출력 레이어에서 결과 추출

    # 본 측정: 개별 추론 레이턴시를 리스트에 기록
    latencies = []
    mem_peak = mem_before
    for _ in range(num_runs):
        start = time.perf_counter()
        ex = net.create_extractor()
        ex.input(input_name, mat_in)
        _, _ = ex.extract(output_name)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        cur = proc.memory_info().rss / (1024 * 1024)
        if cur > mem_peak:
            mem_peak = cur

    memory_mb = round(max(0.0, mem_peak - mem_before), 2)
    latencies = np.array(latencies)

    # 모델 가중치 파일 크기 계산
    model_size_mb = 0.0
    if os.path.exists(bin_path):
        model_size_mb = os.path.getsize(bin_path) / (1024 * 1024)

    # Python 바인딩은 개별 레이턴시를 모두 기록하므로 정확한 통계 산출 가능
    return {
        "runtime": "ncnn_vulkan" if use_vulkan else "ncnn_cpu",
        "model": os.path.basename(param_path).replace(".param", ""),
        "latency": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "memory_mb": memory_mb,
        "model_size_mb": round(model_size_mb, 2),
        "num_runs": num_runs,
    }


def benchmark_ncnn(param_path, bin_path=None, num_warmup=10, num_runs=100,
                   num_threads=4, use_vulkan=False, input_size=(3, 224, 224),
                   input_name=None, output_name=None):
    """Benchmark ncnn (auto-select Python binding or subprocess)."""
    # bin_path가 없으면 param_path에서 확장자만 바꿔서 자동 유추
    # 예: model.param → model.bin
    if bin_path is None:
        bin_path = param_path.replace(".param", ".bin")

    # 1순위: Python 바인딩 시도 (더 정확한 레이턴시 통계 산출 가능)
    try:
        import ncnn  # noqa: F401
        return benchmark_ncnn_python(
            param_path, bin_path, num_warmup, num_runs,
            use_vulkan=use_vulkan, input_size=input_size,
            input_name=input_name, output_name=output_name,
        )
    except ImportError:
        pass

    # 2순위: benchncnn 바이너리 서브프로세스로 폴백
    # gpu_index: Vulkan 사용 시 0번 GPU, 미사용 시 -1 (CPU)
    gpu_index = 0 if use_vulkan else -1
    return benchmark_ncnn_subprocess(param_path, num_warmup, num_runs, num_threads, gpu_index)


# 스크립트 직접 실행 시 CLI 인터페이스 제공
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--param", required=True, help="Path to .param file")
    parser.add_argument("--bin", default=None, help="Path to .bin file")
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (-1 for CPU)")
    args = parser.parse_args()

    # 자동 선택 함수로 벤치마크 실행 (Python 바인딩 또는 서브프로세스)
    result = benchmark_ncnn(
        args.param, args.bin, args.num_warmup, args.num_runs, args.threads, args.gpu
    )
    print(json.dumps(result, indent=2))
