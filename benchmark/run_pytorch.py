#!/usr/bin/env python3
# ============================================================================
# run_pytorch.py - PyTorch CPU/CUDA 추론 벤치마크
# ============================================================================
# 역할: PyTorch 모델을 CPU 또는 CUDA(GPU)에서 로드하고 추론 성능을 측정한다.
#
# 시간 측정 방식:
#   - CUDA 모드: torch.cuda.Event를 사용한 GPU 커널 시간 측정
#     * GPU 작업은 비동기적이므로 time.perf_counter()로는 정확한 GPU 시간을 알 수 없음
#     * CUDA Event는 GPU 타임라인에 직접 마커를 삽입하여 정확한 GPU 실행 시간을 측정
#     * synchronize() 호출이 핵심: GPU 작업 완료를 기다린 후 시간을 읽어야 정확함
#   - CPU 모드: time.perf_counter()를 사용한 벽시계 시간 측정
#
# 워밍업의 필요성:
#   - 첫 몇 번의 추론은 CUDA 컨텍스트 초기화, JIT 컴파일, 메모리 할당 등으로
#     실제 추론보다 훨씬 느림. 워밍업으로 이런 일회성 비용을 제거.
# ============================================================================

"""PyTorch CPU/CUDA inference benchmark."""

import os
import time
import numpy as np
import torch
import torchvision.models as models


# ============================================================================
# 지원되는 모델 팩토리 매핑
#   값: (factory, input_size)
# torchvision에서 제공하는 사전학습된 모델을 이름으로 불러올 수 있게 매핑한다.
# input_size는 (N, C, H, W) 튜플이며 검출 모델은 더 큰 해상도를 사용한다.
# ============================================================================
MODEL_FACTORIES = {
    "mobilenetv3_small":  (models.mobilenet_v3_small,  (1, 3, 224, 224)),
    "resnet50":           (models.resnet50,            (1, 3, 224, 224)),
    "efficientnet_b0":    (models.efficientnet_b0,     (1, 3, 224, 224)),
    "shufflenet_v2_x1_0": (models.shufflenet_v2_x1_0,  (1, 3, 224, 224)),
}

# TorchScript로 로딩되는 검출/특수 모델 전용 입력 크기
# (torchvision factory로 복원 불가 → .torchscript 파일 필수)
TORCHSCRIPT_INPUT_SIZES = {
    "yolov8n":          (1, 3, 640, 640),
    "ssd_mobilenet_v2": (1, 3, 320, 320),
}


def benchmark_pytorch(model_name, device="cpu", num_warmup=10, num_runs=100,
                      model_dir="models"):
    """PyTorch 모델의 추론 벤치마크를 수행한다.

    모델 로드 우선순위:
      1) models/<name>.torchscript 가 있으면 TorchScript로 로드 (검출 모델 경로)
      2) MODEL_FACTORIES 에 정의된 torchvision 팩토리로 로드 (분류 모델 경로)

    CPU와 CUDA 두 가지 모드를 지원하며, 각각 다른 시간 측정 방식을 사용한다.
    CUDA 모드에서는 GPU Event 기반의 정확한 커널 실행 시간을 측정하고,
    CPU 모드에서는 time.perf_counter()를 사용한다.

    Args:
        model_name: 모델 이름 (분류 4종 또는 검출 모델명)
        device: 실행 디바이스 ("cpu" 또는 "cuda")
        num_warmup: 워밍업 추론 횟수 (측정에 포함되지 않음)
        num_runs: 실제 벤치마크 반복 횟수
        model_dir: TorchScript 파일을 찾을 디렉토리 경로

    Returns:
        dict: 런타임 이름, 지연시간 통계(mean/std/min/max/p50/p95/p99),
              메모리 사용량(MB), 모델 크기(MB) 등을 포함한 결과 딕셔너리
    """
    device = torch.device(device)

    # --- TorchScript 우선 로딩 (검출 모델 경로) ---
    ts_path = os.path.join(model_dir, f"{model_name}.torchscript")
    if os.path.exists(ts_path):
        # torch.jit.load: 저장된 ScriptModule을 역직렬화
        # 장점: torchvision/ultralytics 의존성 없이 모델 로드 가능
        model = torch.jit.load(ts_path, map_location=device)
        model.eval()
        # 입력 크기는 TORCHSCRIPT_INPUT_SIZES 에서 결정, 없으면 224 기본값
        input_size = TORCHSCRIPT_INPUT_SIZES.get(model_name, (1, 3, 224, 224))
    else:
        # --- torchvision 팩토리 로딩 (분류 모델 경로) ---
        if model_name not in MODEL_FACTORIES:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available factories: {list(MODEL_FACTORIES.keys())}. "
                f"No TorchScript found at {ts_path}."
            )
        factory, input_size = MODEL_FACTORIES[model_name]
        # pretrained=True: ImageNet으로 사전학습된 가중치 로드
        # eval(): 배치 정규화(BatchNorm)와 드롭아웃을 추론 모드로 전환
        model = factory(pretrained=True).eval()
        model = model.to(device)  # CPU 또는 GPU로 모델 이동

    # --- 모델 크기 계산 (MB) ---
    # 파라미터(가중치 + 편향)와 버퍼(BatchNorm의 running_mean 등)의 총 바이트 수
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)

    # --- 더미 입력 텐서 생성 ---
    # 모델별 입력 크기 사용 (분류: 224, yolov8n: 640, ssd: 320)
    input_tensor = torch.randn(*input_size, device=device)

    use_cuda = device.type == "cuda"

    if use_cuda:
        # 피크 메모리 통계를 리셋하여 벤치마크 기간의 최대 메모리만 추적
        torch.cuda.reset_peak_memory_stats(device)

    # CPU 경로에서도 RSS 기반 피크 메모리를 추적 (GPU는 torch API 사용)
    proc = None
    mem_before = 0.0
    mem_peak = 0.0
    if not use_cuda:
        import psutil
        proc = psutil.Process(os.getpid())
        mem_before = proc.memory_info().rss / (1024 * 1024)
        mem_peak = mem_before

    # --- 워밍업 단계 ---
    # inference_mode(): no_grad()보다 효율적인 추론 전용 컨텍스트
    # autograd 그래프를 완전히 비활성화하여 메모리와 연산 절약
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(input_tensor)
            if use_cuda:
                # synchronize(): GPU 작업이 완료될 때까지 CPU가 대기
                # 워밍업에서도 동기화해야 GPU가 안정 상태에 도달
                torch.cuda.synchronize()

    # --- 벤치마크 측정 단계 ---
    latencies = []
    with torch.inference_mode():
        for _ in range(num_runs):
            if use_cuda:
                # CUDA Event 기반 GPU 시간 측정
                # GPU 작업은 비동기적으로 실행되므로 CPU 타이머로는 부정확
                # CUDA Event는 GPU 스트림에 타임스탬프 마커를 삽입
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()      # GPU 스트림에 시작 마커 기록
                _ = model(input_tensor)    # 추론 실행 (비동기, CPU는 즉시 반환)
                end_event.record()         # GPU 스트림에 종료 마커 기록
                torch.cuda.synchronize()   # GPU 작업 완료 대기 (필수!)
                # 두 이벤트 사이의 GPU 경과 시간 (밀리초)
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                # CPU 모드: 벽시계 시간으로 측정 (동기 실행이므로 정확)
                start = time.perf_counter()
                _ = model(input_tensor)
                elapsed_ms = (time.perf_counter() - start) * 1000  # 초 -> 밀리초 변환
                cur = proc.memory_info().rss / (1024 * 1024)
                if cur > mem_peak:
                    mem_peak = cur

            latencies.append(elapsed_ms)

    latencies = np.array(latencies)

    # --- 메모리 사용량 조회 ---
    # CUDA: torch.cuda.max_memory_allocated / CPU: psutil RSS 피크 증가량
    if use_cuda:
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        memory_mb = max(0.0, mem_peak - mem_before)

    # --- 결과 딕셔너리 반환 ---
    return {
        "runtime": f"pytorch_{'cuda' if use_cuda else 'cpu'}",
        "model": model_name,
        "latency": {
            "mean": float(np.mean(latencies)),              # 평균 지연시간
            "std": float(np.std(latencies)),                 # 표준편차 (안정성 지표)
            "min": float(np.min(latencies)),                 # 최소 지연시간 (최상의 경우)
            "max": float(np.max(latencies)),                 # 최대 지연시간 (최악의 경우)
            "p50": float(np.percentile(latencies, 50)),      # 중간값
            "p95": float(np.percentile(latencies, 95)),      # 95번째 백분위 (꼬리 지연)
            "p99": float(np.percentile(latencies, 99)),      # 99번째 백분위 (극단적 꼬리)
        },
        "memory_mb": round(memory_mb, 2),
        "model_size_mb": round(model_size_mb, 2),
        "num_runs": num_runs,
    }


# ============================================================================
# 독립 실행 모드: 이 파일을 직접 실행할 때 CLI 인터페이스 제공
# 예: python run_pytorch.py --model mobilenetv3_small --device cuda
# ============================================================================
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Model name (classification factory or *.torchscript name)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--model-dir", default="models",
                        help="Directory to find TorchScript files")
    args = parser.parse_args()

    result = benchmark_pytorch(
        args.model, args.device, args.num_warmup, args.num_runs, args.model_dir
    )
    print(json.dumps(result, indent=2))
