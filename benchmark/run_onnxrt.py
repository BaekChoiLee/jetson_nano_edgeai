#!/usr/bin/env python3
# ============================================================================
# run_onnxrt.py - ONNX Runtime 추론 벤치마크 (Fallback 방어 추가)
# ============================================================================

import time
import os
import numpy as np

def build_ort_session(onnx_path, provider):
    import onnxruntime as ort
    so = ort.SessionOptions()
    # 그래프 최적화 극대화
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    if provider == "CUDAExecutionProvider":
        so.log_severity_level = 0
        so.log_verbosity_level = 1
        prov_config = [("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "1",
        })]
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=prov_config)
        # CPU Fallback 방어선 구축 (Codex 로직)
        active = sess.get_providers()
        if active != ["CUDAExecutionProvider"]:
            print(f"[WARN] CPU fallback risk detected: active providers={active}. Performance may degrade.")
    else:
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=[provider])
        
    return sess

def benchmark_onnxrt(onnx_path, provider="CUDAExecutionProvider", num_warmup=10, num_runs=100):
    import onnxruntime as ort
    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(f"Provider {provider} not available. Available: {available}")

    session = build_ort_session(onnx_path, provider)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    concrete_shape = [1 if isinstance(d, str) or d is None else d for d in input_shape]
    input_data = np.random.randn(*concrete_shape).astype(np.float32)

    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    import psutil
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024 * 1024)

    for _ in range(num_warmup):
        session.run(None, {input_name: input_data})

    latencies = []
    mem_peak = mem_before
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: input_data})
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        cur = proc.memory_info().rss / (1024 * 1024)
        if cur > mem_peak:
            mem_peak = cur

    memory_mb = round(max(0.0, mem_peak - mem_before), 2)

    latencies = np.array(latencies)
    provider_short = provider.replace("ExecutionProvider", "")

    return {
        "runtime": f"onnxrt_{provider_short.lower()}",
        "model": os.path.basename(onnx_path),
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
        "provider": provider,
    }

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--provider", default="CUDAExecutionProvider", choices=["CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"])
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    args = parser.parse_args()

    result = benchmark_onnxrt(args.onnx, args.provider, args.num_warmup, args.num_runs)
    print(json.dumps(result, indent=2))
