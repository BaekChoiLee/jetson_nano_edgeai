#!/usr/bin/env python3
"""Main benchmark orchestrator for all runtimes."""

import argparse
import os
import sys
import json
import csv
import time
from datetime import datetime

import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tegra_parser import TegraStatsLogger


ALL_RUNTIMES = [
    "pytorch_cpu",
    "pytorch_cuda",
    "tensorrt_fp32",
    "tensorrt_fp16",
    "tensorrt_int8",
    "onnxrt_cuda",
    "onnxrt_trt",
    "tflite_cpu",
    "tflite_gpu",
    "ncnn_cpu",
    "ncnn_vulkan",
]


def run_single_benchmark(model_name, runtime, model_dir, num_warmup, num_runs):
    """Run benchmark for a single runtime. Returns result dict or None on error."""
    print(f"\n  --- {runtime} ---")

    try:
        if runtime == "pytorch_cpu":
            from run_pytorch import benchmark_pytorch
            return benchmark_pytorch(model_name, "cpu", num_warmup, num_runs)

        elif runtime == "pytorch_cuda":
            from run_pytorch import benchmark_pytorch
            return benchmark_pytorch(model_name, "cuda", num_warmup, num_runs)

        elif runtime.startswith("tensorrt_"):
            precision = runtime.split("_")[1]  # fp32, fp16, int8
            engine_path = os.path.join(model_dir, f"{model_name}_{precision}.engine")
            if not os.path.exists(engine_path):
                print(f"  [SKIP] Engine not found: {engine_path}")
                return None
            from run_tensorrt import benchmark_tensorrt
            return benchmark_tensorrt(engine_path, num_warmup, num_runs)

        elif runtime == "onnxrt_cuda" or runtime == "onnxrt_trt":
            onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
            if not os.path.exists(onnx_path):
                print(f"  [SKIP] ONNX not found: {onnx_path}")
                return None
            from run_onnxrt import benchmark_onnxrt
            ep = "CUDAExecutionProvider" if runtime == "onnxrt_cuda" else "TensorrtExecutionProvider"
            return benchmark_onnxrt(onnx_path, ep, num_warmup, num_runs)

        elif runtime.startswith("tflite_"):
            tflite_path = os.path.join(model_dir, f"{model_name}.tflite")
            if not os.path.exists(tflite_path):
                print(f"  [SKIP] TFLite not found: {tflite_path}")
                return None
            from run_tflite import benchmark_tflite
            use_gpu = (runtime == "tflite_gpu")
            return benchmark_tflite(tflite_path, num_warmup, num_runs, use_gpu=use_gpu)

        elif runtime.startswith("ncnn_"):
            param_path = os.path.join(model_dir, f"{model_name}.param")
            bin_path = os.path.join(model_dir, f"{model_name}.bin")
            if not os.path.exists(param_path):
                print(f"  [SKIP] ncnn param not found: {param_path}")
                return None
            from run_ncnn import benchmark_ncnn
            use_vulkan = (runtime == "ncnn_vulkan")
            return benchmark_ncnn(param_path, bin_path, num_warmup, num_runs, use_vulkan=use_vulkan)

        else:
            print(f"  [SKIP] Unknown runtime: {runtime}")
            return None

    except Exception as e:
        print(f"  [ERROR] {runtime}: {e}")
        return None


def get_power_mode():
    """Get current Jetson power mode."""
    try:
        import subprocess
        result = subprocess.run(
            ["sudo", "nvpmodel", "-q"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "NV Power Mode" in line:
                return line.strip()
        return "unknown"
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Jetson Nano Benchmark Suite")
    parser.add_argument(
        "--model", required=True,
        choices=["mobilenetv3_small", "resnet50", "shufflenet_v2", "ssd_mobilenet_v2"],
        help="Model to benchmark",
    )
    parser.add_argument(
        "--runtimes", default="all",
        help="Comma-separated runtimes or 'all'",
    )
    parser.add_argument("--power-mode", default="10w", choices=["10w", "5w"])
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--cool-down", type=int, default=60, help="Seconds between runtimes")
    parser.add_argument("--model-dir", default="./models", help="Dir with model files")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--no-tegrastats", action="store_true", help="Skip tegrastats logging")
    args = parser.parse_args()

    # Parse runtimes
    if args.runtimes == "all":
        runtimes = ALL_RUNTIMES
    else:
        runtimes = [r.strip() for r in args.runtimes.split(",")]
        invalid = [r for r in runtimes if r not in ALL_RUNTIMES]
        if invalid:
            print(f"Invalid runtimes: {invalid}")
            print(f"Available: {ALL_RUNTIMES}")
            sys.exit(1)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{args.model}_{args.power_mode}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    power_mode_str = get_power_mode()

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
    tegra_logger = TegraStatsLogger() if not args.no_tegrastats else None

    for i, runtime in enumerate(runtimes):
        print(f"\n[{i+1}/{len(runtimes)}] Benchmarking: {runtime}")

        # Start tegrastats logging
        tegra_log = os.path.join(run_dir, f"tegra_{runtime}.log")
        if tegra_logger:
            try:
                tegra_logger.start(tegra_log, interval_ms=100)
            except Exception as e:
                print(f"  [WARN] tegrastats failed: {e}")
                tegra_logger = None

        # Run benchmark
        start_time = time.time()
        result = run_single_benchmark(
            args.model, runtime, args.model_dir, args.num_warmup, args.num_runs
        )
        elapsed = time.time() - start_time

        # Stop tegrastats
        tegra_summary = {}
        if tegra_logger:
            tegra_logger.stop()
            tegra_summary = tegra_logger.summary(tegra_log)

        if result:
            result["power_mode"] = args.power_mode
            result["tegrastats"] = tegra_summary
            result["wall_time_s"] = round(elapsed, 2)
            all_results.append(result)

            # Print quick summary
            lat = result["latency"]
            print(f"  Mean: {lat['mean']:.2f} ms | P95: {lat['p95']:.2f} ms | Std: {lat['std']:.2f} ms")
            if tegra_summary and "pom_5v_in_current_mw" in tegra_summary:
                pwr = tegra_summary["pom_5v_in_current_mw"]
                print(f"  Power: avg {pwr['mean']:.0f} mW, max {pwr['max']:.0f} mW")

        # Cool-down between runtimes
        if i < len(runtimes) - 1 and args.cool_down > 0:
            print(f"  Cooling down for {args.cool_down}s...")
            time.sleep(args.cool_down)

    # Save results
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  JSON saved: {json_path}")

    # Save CSV summary
    csv_path = os.path.join(run_dir, "summary.csv")
    if all_results:
        csv_rows = []
        for r in all_results:
            row = {
                "model": r.get("model", args.model),
                "runtime": r.get("runtime", ""),
                "power_mode": r.get("power_mode", ""),
                "mean_ms": r["latency"]["mean"],
                "std_ms": r["latency"]["std"],
                "min_ms": r["latency"]["min"],
                "max_ms": r["latency"]["max"],
                "p50_ms": r["latency"]["p50"],
                "p95_ms": r["latency"]["p95"],
                "p99_ms": r["latency"]["p99"],
                "memory_mb": r.get("memory_mb", 0),
                "model_size_mb": r.get("model_size_mb", 0),
                "fps": round(1000.0 / r["latency"]["mean"], 2) if r["latency"]["mean"] > 0 else 0,
            }
            # Add power if available
            tegra = r.get("tegrastats", {})
            if "pom_5v_in_current_mw" in tegra:
                row["power_avg_mw"] = tegra["pom_5v_in_current_mw"]["mean"]
                row["power_max_mw"] = tegra["pom_5v_in_current_mw"]["max"]
            csv_rows.append(row)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  CSV saved:  {csv_path}")

    # Final summary table
    print(f"\n{'='*70}")
    print(f"  Summary: {args.model} @ {args.power_mode}")
    print(f"{'='*70}")
    print(f"  {'Runtime':<20} {'Mean(ms)':<10} {'P95(ms)':<10} {'FPS':<10} {'Mem(MB)'}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in all_results:
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


if __name__ == "__main__":
    main()
