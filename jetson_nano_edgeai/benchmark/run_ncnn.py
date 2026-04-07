#!/usr/bin/env python3
"""ncnn inference benchmark (via subprocess or Python binding)."""

import time
import os
import re
import subprocess
import numpy as np


def _find_benchncnn():
    """Locate benchncnn binary."""
    import shutil

    # Check PATH
    path = shutil.which("benchncnn")
    if path:
        return path

    # Check common locations
    ncnn_dir = os.environ.get("NCNN_DIR", os.path.expanduser("~/ncnn"))
    candidate = os.path.join(ncnn_dir, "build", "benchmark", "benchncnn")
    if os.path.exists(candidate):
        return candidate

    return None


def benchmark_ncnn_subprocess(param_path, num_warmup=10, num_runs=100, num_threads=4, gpu_index=0):
    """Benchmark ncnn using benchncnn binary.

    Returns dict with latency stats.
    """
    benchncnn = _find_benchncnn()
    if not benchncnn:
        raise FileNotFoundError(
            "benchncnn not found. Set NCNN_DIR or add to PATH."
        )

    model_name = os.path.splitext(os.path.basename(param_path))[0]
    param_dir = os.path.dirname(os.path.abspath(param_path))

    # benchncnn <param> <runs> <threads> <gpu_index>
    # NOTE: benchncnn reads .param and .bin from same directory
    cmd = [
        benchncnn,
        param_path,
        str(num_warmup + num_runs),
        str(num_threads),
        str(gpu_index),
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, cwd=param_dir
    )

    if result.returncode != 0:
        raise RuntimeError(f"benchncnn failed: {result.stderr}")

    # Parse output: "model  min = X  max = Y  avg = Z"
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

    # Use last line (summary)
    summary = latencies_from_output[-1]

    # Get bin file size
    bin_path = param_path.replace(".param", ".bin")
    model_size_mb = 0.0
    if os.path.exists(bin_path):
        model_size_mb = os.path.getsize(bin_path) / (1024 * 1024)

    return {
        "runtime": f"ncnn_gpu{gpu_index}" if gpu_index >= 0 else "ncnn_cpu",
        "model": model_name,
        "latency": {
            "mean": summary["avg"],
            "std": 0.0,  # benchncnn doesn't report std
            "min": summary["min"],
            "max": summary["max"],
            "p50": summary["avg"],  # approximation
            "p95": summary["max"],  # approximation
            "p99": summary["max"],  # approximation
        },
        "memory_mb": 0.0,
        "model_size_mb": round(model_size_mb, 2),
        "num_runs": num_runs,
        "num_threads": num_threads,
        "gpu_index": gpu_index,
        "raw_output": result.stdout.strip(),
    }


def benchmark_ncnn_python(param_path, bin_path, num_warmup=10, num_runs=100, use_vulkan=False):
    """Benchmark ncnn using Python binding (if available)."""
    import ncnn

    net = ncnn.Net()
    net.opt.use_vulkan_compute = use_vulkan
    net.load_param(param_path)
    net.load_model(bin_path)

    # Create input
    mat_in = ncnn.Mat(224, 224, 3)

    # Warm-up
    for _ in range(num_warmup):
        ex = net.create_extractor()
        ex.input("input", mat_in)
        _, _ = ex.extract("output")

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ex = net.create_extractor()
        ex.input("input", mat_in)
        _, _ = ex.extract("output")
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies = np.array(latencies)

    model_size_mb = 0.0
    if os.path.exists(bin_path):
        model_size_mb = os.path.getsize(bin_path) / (1024 * 1024)

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
        "memory_mb": 0.0,
        "model_size_mb": round(model_size_mb, 2),
        "num_runs": num_runs,
    }


def benchmark_ncnn(param_path, bin_path=None, num_warmup=10, num_runs=100,
                   num_threads=4, use_vulkan=False):
    """Benchmark ncnn (auto-select Python binding or subprocess)."""
    if bin_path is None:
        bin_path = param_path.replace(".param", ".bin")

    # Try Python binding first
    try:
        import ncnn  # noqa: F401
        return benchmark_ncnn_python(param_path, bin_path, num_warmup, num_runs, use_vulkan=use_vulkan)
    except ImportError:
        pass

    # Fall back to subprocess
    gpu_index = 0 if use_vulkan else -1
    return benchmark_ncnn_subprocess(param_path, num_warmup, num_runs, num_threads, gpu_index)


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

    result = benchmark_ncnn(
        args.param, args.bin, args.num_warmup, args.num_runs, args.threads, args.gpu
    )
    print(json.dumps(result, indent=2))
