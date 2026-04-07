#!/usr/bin/env python3
"""TFLite inference benchmark."""

import time
import os
import numpy as np


def _get_interpreter(model_path, use_gpu=False):
    """Load TFLite interpreter (try tflite_runtime first, then tf.lite)."""
    delegates = None
    try:
        import tflite_runtime.interpreter as tflite
        if use_gpu:
            try:
                delegates = [tflite.load_delegate("libdelegate_gpu.so")]
            except Exception as e:
                print(f"  [WARN] Failed to load GPU delegate via tflite_runtime: {e}")
        return tflite.Interpreter(model_path=model_path, experimental_delegates=delegates)
    except ImportError:
        import tensorflow as tf
        if use_gpu:
            try:
                delegates = [tf.lite.experimental.load_delegate("libdelegate_gpu.so")]
            except Exception as e:
                print(f"  [WARN] Failed to load GPU delegate via tf.lite: {e}")
        return tf.lite.Interpreter(model_path=model_path, experimental_delegates=delegates)


def benchmark_tflite(model_path, num_warmup=10, num_runs=100, use_gpu=False):
    """Benchmark TFLite model.

    Returns dict with latency stats.
    """
    interpreter = _get_interpreter(model_path, use_gpu)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create input data matching expected shape and dtype
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]
    input_data = np.random.randn(*input_shape).astype(input_dtype)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Warm-up
    for _ in range(num_warmup):
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies = np.array(latencies)

    return {
        "runtime": "tflite_gpu" if use_gpu else "tflite_cpu",
        "model": os.path.basename(model_path),
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
        "input_shape": input_shape.tolist(),
        "input_dtype": str(input_dtype),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    args = parser.parse_args()

    result = benchmark_tflite(args.model, args.num_warmup, args.num_runs)
    print(json.dumps(result, indent=2))
