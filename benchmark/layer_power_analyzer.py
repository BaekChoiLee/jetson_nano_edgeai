#!/usr/bin/env python3
# =============================================================================
# layer_power_analyzer.py — Layer Amplification Loop 기반 레이어별 전력 측정
# =============================================================================
# 역할: 기존 layer_analyzer.py 가 forward_hook 으로 지연만 측정하던 한계를 보완.
#       각 leaf 모듈을 N 회 독립 반복(amplification)하면서 tegrastats 100ms 샘플
#       구간에 충분한 사이클이 들어가게 만들어 평균 전력을 의미 있게 측정한다.
#
# 핵심 절차:
#   1) 정상 forward 1회로 각 leaf 모듈의 입력 텐서를 캐시
#   2) leaf 별로:
#        a) GPU 캐시 비우기 + 쿨다운
#        b) tegrastats 시작
#        c) cached_input 으로 N (=2000~5000) 회 반복 실행 + cuda.synchronize
#        d) tegrastats 정지 → 평균/최대 전력 추출
#        e) CUDA Event(GPU) 또는 perf_counter(CPU)로 단일 실행 정확 지연 측정
#   3) CSV 저장: layer_name, layer_type, num_amplify, avg_latency_ms,
#      avg_power_mw, gpu_power_mw, cpu_power_mw, energy_per_run_mj, gpu_temp_c
#
# ⚠️ 중요 캐비어트 (사용자 동의):
#   "독립 실행 격리 전력" — leaf 를 실제 추론 graph 컨텍스트와 분리해서 측정한
#   값이므로, in-context 전력과는 차이가 있을 수 있다. 차트 캡션·CSV header
#   주석에 반드시 명시할 것.
#
# 사용법:
#   python benchmark/layer_power_analyzer.py \
#     --model mobilenetv3_small \
#     --num-amplify 2000 \
#     --output-dir results/layer_power
# =============================================================================

"""Layer Amplification Loop power profiler for 6-model coverage."""

import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager

import numpy as np


# =============================================================================
# 모델 팩토리 — 6 모델 통합 (분류 4 + 검출 2)
# 검출 모델은 TorchScript 로딩이 우선
# =============================================================================
def _torchvision_factory(name):
    """torchvision.models 에서 lazily 모델을 가져온다."""
    import torchvision.models as M
    return getattr(M, name)


CLASSIFICATION_MODELS = {
    "mobilenetv3_small":  ("mobilenet_v3_small",   (1, 3, 224, 224)),
    "resnet50":           ("resnet50",             (1, 3, 224, 224)),
    "efficientnet_b0":    ("efficientnet_b0",      (1, 3, 224, 224)),
    "shufflenet_v2_x1_0": ("shufflenet_v2_x1_0",   (1, 3, 224, 224)),
}

DETECTION_INPUT_SIZES = {
    "yolov8n":          (1, 3, 640, 640),
    "ssd_mobilenet_v2": (1, 3, 320, 320),
}


def load_model_for_profiling(model_name, model_dir, device):
    """6 모델 통합 로더.

    검출 모델: <model_dir>/<name>.torchscript
    분류 모델: torchvision factory(pretrained=True)

    Returns:
        (model, input_shape)
    """
    import torch

    ts_path = os.path.join(model_dir, f"{model_name}.torchscript")
    if model_name in DETECTION_INPUT_SIZES:
        if not os.path.exists(ts_path):
            raise FileNotFoundError(
                f"Detection TorchScript not found: {ts_path}. "
                f"Convert via convert/export_detection.py 또는 onnx_to_pytorch.py"
            )
        model = torch.jit.load(ts_path, map_location=device)
        model.eval()
        return model, DETECTION_INPUT_SIZES[model_name]

    if model_name not in CLASSIFICATION_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: {list(CLASSIFICATION_MODELS.keys()) + list(DETECTION_INPUT_SIZES.keys())}"
        )

    factory_name, input_size = CLASSIFICATION_MODELS[model_name]
    factory = _torchvision_factory(factory_name)
    model = factory(pretrained=True).eval().to(device)
    return model, input_size


# =============================================================================
# Leaf 모듈 입력 캐싱 (forward_hook)
# =============================================================================
def cache_leaf_inputs(model, dummy_input):
    """정상 추론 1회로 각 leaf 모듈의 입력 텐서를 캐시.

    검출 모델은 출력이 dict/list/tuple 일 수 있으므로 try/except 로 감싸서
    forward 실패해도 cache 는 살아남도록 한다.
    """
    import torch

    leaf_inputs = {}
    handles = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            if name in leaf_inputs:
                return
            if not isinstance(inp, tuple) or len(inp) == 0:
                return
            first = inp[0]
            if not hasattr(first, "shape"):
                return
            try:
                leaf_inputs[name] = first.detach().clone()
            except Exception:
                pass
        return hook_fn

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.inference_mode():
            _ = model(dummy_input)
    except Exception as e:
        print(f"  [warn] forward 실패 (검출 모델 dict 출력 등): {e}")

    for h in handles:
        h.remove()

    return leaf_inputs


# =============================================================================
# 단일 레이어 정확 지연 측정
# =============================================================================
def measure_layer_latency(layer, input_tensor, device, num_runs=100):
    """단일 leaf 의 평균 레이턴시(ms).

    GPU: torch.cuda.Event 기반 (가장 정확)
    CPU: time.perf_counter 기반
    """
    import torch

    times = []
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        # 워밍업
        with torch.inference_mode():
            for _ in range(min(5, num_runs)):
                _ = layer(input_tensor)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
        with torch.inference_mode():
            for i in range(num_runs):
                starts[i].record()
                _ = layer(input_tensor)
                ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    else:
        with torch.inference_mode():
            for _ in range(min(5, num_runs)):
                _ = layer(input_tensor)
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = layer(input_tensor)
                times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)) if times else 0.0


# =============================================================================
# Layer Amplification Loop + tegrastats 샌드위치
# =============================================================================
def amplified_run_with_power(layer, input_tensor, device, tegra_log_path,
                             num_amplify=2000):
    """N 회 반복 실행하면서 tegrastats 로 전력 평균을 측정.

    Returns:
        dict with avg/gpu/cpu power_mw, gpu_temp_c. tegrastats 미작동 시 0.
    """
    import torch

    # tegrastats 로거 lazy import (PC dry-run 환경에서도 import 가능하게)
    try:
        from tegra_parser import TegraStatsLogger
        logger = TegraStatsLogger()
    except Exception as e:
        print(f"  [warn] tegrastats logger unavailable: {e}")
        logger = None

    if logger:
        logger.start(tegra_log_path, interval_ms=100)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(num_amplify):
            _ = layer(input_tensor)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    if not logger:
        return {
            "avg_power_mw": 0.0,
            "gpu_power_mw": 0.0,
            "cpu_power_mw": 0.0,
            "gpu_temp_c": 0.0,
            "samples": 0,
        }

    logger.stop()
    try:
        summary = logger.summary(tegra_log_path)
    except Exception as e:
        print(f"  [warn] tegrastats parse failed: {e}")
        summary = {}

    return {
        "avg_power_mw": _get_metric(summary, "pom_5v_in_current_mw"),
        "gpu_power_mw": _get_metric(summary, "pom_5v_gpu_current_mw"),
        "cpu_power_mw": _get_metric(summary, "pom_5v_cpu_current_mw"),
        "gpu_temp_c":   _get_metric(summary, "temp_gpu_c"),
        "samples":      _get_count(summary, "pom_5v_in_current_mw"),
    }


def _get_metric(summary, key):
    """tegra summary dict 에서 평균값을 안전하게 추출."""
    entry = summary.get(key) if isinstance(summary, dict) else None
    if isinstance(entry, dict):
        return float(entry.get("mean", 0.0) or 0.0)
    return 0.0


def _get_count(summary, key):
    entry = summary.get(key) if isinstance(summary, dict) else None
    if isinstance(entry, dict):
        return int(entry.get("count", 0) or 0)
    return 0


# =============================================================================
# 메인 분석 루프
# =============================================================================
def analyze_layers_with_power(model_name, device="cuda",
                              num_amplify=2000, num_latency_runs=100,
                              output_dir="./results/layer_power",
                              model_dir="models",
                              auto_extend_threshold=5):
    """6 모델 공통 레이어별 전력+지연 분석.

    Args:
        model_name: 6 모델 중 하나
        device: 'cuda' 또는 'cpu'
        num_amplify: tegrastats 샘플 확보용 반복 횟수
        num_latency_runs: 단일 레이어 정확 지연 측정 반복
        output_dir: CSV/log 저장 디렉토리
        model_dir: 모델 아티팩트 디렉토리
        auto_extend_threshold: tegrastats 샘플 < N 이면 num_amplify 2배 자동 연장
    """
    import torch

    os.makedirs(output_dir, exist_ok=True)

    # 1) 모델 + 더미 입력 로드
    print(f"[1/3] Loading model {model_name} on {device}...")
    model, input_shape = load_model_for_profiling(model_name, model_dir, device)
    dummy = torch.randn(*input_shape, device=device)

    # 2) Leaf 입력 캐싱
    print(f"[2/3] Caching leaf module inputs (1 forward pass)...")
    leaf_inputs = cache_leaf_inputs(model, dummy)
    leaves = [
        (name, module) for name, module in model.named_modules()
        if len(list(module.children())) == 0 and name in leaf_inputs
    ]
    print(f"  cached: {len(leaves)} leaf modules")

    # 3) 레이어별 amplification + 전력 측정
    print(f"[3/3] Profiling {len(leaves)} layers with amplification "
          f"(num_amplify={num_amplify})...")

    results = []
    for idx, (name, layer) in enumerate(leaves, 1):
        cached = leaf_inputs[name]
        safe_name = name.replace(".", "_").replace("/", "_") or "root"
        tegra_log = os.path.join(
            output_dir, f"tegra_{model_name}_{safe_name}.log"
        )

        # 쿨다운 (GPU 캐시 비우기 + 짧은 sleep)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        time.sleep(0.5)

        # amplification + tegrastats
        power = amplified_run_with_power(
            layer, cached, device, tegra_log, num_amplify=num_amplify
        )

        # 샘플 부족 → 자동 연장 1회 시도
        actual_amplify = num_amplify
        if power["samples"] < auto_extend_threshold:
            extended = num_amplify * 2
            print(f"  [{idx}/{len(leaves)}] {name}: only {power['samples']} "
                  f"samples → re-running with num_amplify={extended}")
            power = amplified_run_with_power(
                layer, cached, device, tegra_log, num_amplify=extended
            )
            actual_amplify = extended

        # 정확 지연 측정 (별도)
        lat_ms = measure_layer_latency(
            layer, cached, device, num_runs=num_latency_runs
        )

        # 단위 환산: avg_power_mw * lat_ms ms / 1000 = mJ
        energy_per_run_mj = (power["avg_power_mw"] * lat_ms) / 1000.0

        try:
            params = sum(p.numel() for p in layer.parameters())
        except Exception:
            params = 0

        record = {
            "layer_name":        name,
            "layer_type":        layer.__class__.__name__,
            "params":            int(params),
            "input_shape":       list(cached.shape),
            "num_amplify":       int(actual_amplify),
            "tegra_samples":     int(power["samples"]),
            "avg_latency_ms":    round(lat_ms, 4),
            "avg_power_mw":      round(power["avg_power_mw"], 2),
            "gpu_power_mw":      round(power["gpu_power_mw"], 2),
            "cpu_power_mw":      round(power["cpu_power_mw"], 2),
            "energy_per_run_mj": round(energy_per_run_mj, 4),
            "gpu_temp_c":        round(power["gpu_temp_c"], 1),
        }
        results.append(record)

        if idx % 10 == 0 or idx == len(leaves):
            print(f"  [{idx}/{len(leaves)}] {name} "
                  f"({record['layer_type']}) "
                  f"lat={lat_ms:.3f} ms, "
                  f"power={power['avg_power_mw']:.0f} mW")

    # CSV 저장
    csv_path = os.path.join(output_dir, f"{model_name}_layer_power.csv")
    _save_csv(csv_path, results)

    # JSON 도 함께 (디버깅·재가공용)
    json_path = os.path.join(output_dir, f"{model_name}_layer_power.json")
    with open(json_path, "w") as f:
        json.dump({
            "model": model_name,
            "device": str(device),
            "num_amplify_default": num_amplify,
            "input_shape": list(input_shape),
            "warning": (
                "독립 실행 격리 전력 측정값 — 실제 추론 in-context 전력과 "
                "차이가 있을 수 있음 (Layer Amplification Loop)"
            ),
            "layers": results,
        }, f, indent=2)

    print(f"\n[DONE] Layer power profile saved:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  ⚠️  CAVEAT: 독립 실행 격리 전력 — in-context 전력과 차이 가능")
    return results


def _save_csv(path, records):
    """records 리스트를 CSV 로 저장 (header 자동)."""
    if not records:
        return
    fields = list(records[0].keys())
    with open(path, "w", newline="") as f:
        # 첫 줄에 caveat 주석 (pandas 등은 # 주석 옵션으로 무시 가능)
        f.write("# 독립 실행 격리 전력 — in-context 추론 전력과 차이 가능\n")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            # list 필드는 문자열로
            row = {k: (json.dumps(v) if isinstance(v, list) else v)
                   for k, v in r.items()}
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Layer Amplification Loop power profiler (6 models)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help=f"Model name. One of: "
             f"{list(CLASSIFICATION_MODELS.keys()) + list(DETECTION_INPUT_SIZES.keys())}",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-amplify", type=int, default=2000)
    parser.add_argument("--num-latency-runs", type=int, default=100)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="./results/layer_power")
    args = parser.parse_args()

    analyze_layers_with_power(
        model_name=args.model,
        device=args.device,
        num_amplify=args.num_amplify,
        num_latency_runs=args.num_latency_runs,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
