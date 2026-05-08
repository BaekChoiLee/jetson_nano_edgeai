#!/usr/bin/env python3
"""TensorRT inference benchmark."""

import time
import os
import numpy as np

# Patch for newer numpy versions that removed np.bool
if not hasattr(np, 'bool'):
    np.bool = np.bool_

def benchmark_tensorrt(engine_path, num_warmup=10, num_runs=100):
    """Benchmark TensorRT engine.

    Returns dict with latency stats and memory usage.
    """
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 - initializes CUDA context

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Detect precision from filename
    basename = os.path.basename(engine_path).lower()
    if "int8" in basename:
        precision = "int8"
    elif "fp16" in basename:
        precision = "fp16"
    else:
        precision = "fp32"

    # Load engine
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    model_size_mb = os.path.getsize(engine_path) / (1024 * 1024)

    # Allocate buffers
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for i in range(engine.num_bindings):
        shape = engine.get_binding_shape(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.binding_is_input(i):
            # Fill with random data
            host_mem[:] = np.random.randn(*host_mem.shape).astype(dtype).flatten()
            inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
        else:
            outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

    def infer():
        """Run single inference."""
        for inp in inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
        stream.synchronize()

    # Warm-up
    for _ in range(num_warmup):
        infer()

    # Benchmark with CUDA events
    latencies = []
    for _ in range(num_runs):
        start = cuda.Event()
        end = cuda.Event()
        start.record(stream)
        infer()
        end.record(stream)
        end.synchronize()
        elapsed_ms = start.time_till(end)
        latencies.append(elapsed_ms)

    latencies = np.array(latencies)

    # GPU memory (approximate from allocated buffers)
    total_buffer_mb = sum(
        inp["host"].nbytes for inp in inputs
    ) + sum(
        out["host"].nbytes for out in outputs
    )
    total_buffer_mb = total_buffer_mb / (1024 * 1024)

    return {
        "runtime": f"tensorrt_{precision}",
        "model": os.path.basename(engine_path),
        "latency": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "memory_mb": round(total_buffer_mb, 2),
        "model_size_mb": round(model_size_mb, 2),
        "num_runs": num_runs,
        "precision": precision,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="Path to TensorRT engine file")
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    args = parser.parse_args()

    result = benchmark_tensorrt(args.engine, args.num_warmup, args.num_runs)
    print(json.dumps(result, indent=2))
