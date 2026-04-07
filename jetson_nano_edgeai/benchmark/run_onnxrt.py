#!/usr/bin/env python3
"""ONNX Runtime inference benchmark."""

import time
import os
import numpy as np


def benchmark_onnxrt(onnx_path, provider="CUDAExecutionProvider", num_warmup=10, num_runs=100):
    """Benchmark ONNX Runtime with specified execution provider.

    Providers: CUDAExecutionProvider, TensorrtExecutionProvider, CPUExecutionProvider
    Returns dict with latency stats.
    """
    import onnxruntime as ort

    # Check provider availability
    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(
            f"Provider {provider} not available. Available: {available}"
        )

    # Create session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options, providers=[provider])

    # Get input/output info
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    # Replace dynamic dims with 1
    concrete_shape = [1 if isinstance(d, str) or d is None else d for d in input_shape]

    # Create input
    input_data = np.random.randn(*concrete_shape).astype(np.float32)

    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    # Warm-up
    for _ in range(num_warmup):
        session.run(None, {input_name: input_data})

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: input_data})
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

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
        "memory_mb": 0.0,
        "model_size_mb": round(model_size_mb, 2),
        "num_runs": num_runs,
        "provider": provider,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument(
        "--provider",
        default="CUDAExecutionProvider",
        choices=["CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"],
    )
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    args = parser.parse_args()

    result = benchmark_onnxrt(args.onnx, args.provider, args.num_warmup, args.num_runs)
    print(json.dumps(result, indent=2))
