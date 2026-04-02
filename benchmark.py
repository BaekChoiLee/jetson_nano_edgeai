# benchmark.py
import time
import numpy as np
import csv
from pathlib import Path

def run_benchmark(infer_fn, input_data, warmup=10, runs=100):
    """
    infer_fn: 런타임별 추론 함수 (람다로 넘김), 런타임에 상관없이 실행될 수 있음 
    input_data: 모델 입력 numpy array
    warmup: 본 측정 전 미리 돌려보는 횟수
    runs: 실제 통계를 내기 위해 반복 측정하는 횟수
    """
    
    """
    GPU나 Edge 가속기(NPU)는 처음에 모델을 로드하거나 JIT(Just-In-Time) 컴파일을 수행할 때 시간이 오래 걸림
    만약 웜업 없이 측정하면 첫 번째 실행 시간 때문에 전체 평균값이 왜곡되므로, "엔진을 예열"시킨 후 측정하는 것이 정석
    """
    for _ in range(warmup):
        infer_fn(input_data)

    # 본 측정
    times = []
    for _ in range(runs):
        start = time.perf_counter() # perf_counter() : 고해상도 타이머
        infer_fn(input_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms 단위

    times = np.array(times)
    return {
        'mean_ms':   round(float(np.mean(times)), 3),
        'std_ms':    round(float(np.std(times)), 3),
        'min_ms':    round(float(np.min(times)), 3),
        'max_ms':    round(float(np.max(times)), 3),
        'median_ms': round(float(np.median(times)), 3),
    }

"""
사용 예시
import torch
import torchvision.models as models

model = models.mobilenet_v3_small(pretrained=True).eval()
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
inp   = torch.from_numpy(dummy)

result = run_benchmark(
    infer_fn=lambda x: model(x),
    input_data=inp
)
print(result)
# {'mean_ms': 45.2, 'std_ms': 1.3, 'min_ms': 43.1, ...}
"""