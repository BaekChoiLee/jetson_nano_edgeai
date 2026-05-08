#!/usr/bin/env python3
"""
패자부활전: TFLite GPU + ncnn Vulkan 전용 벤치마크
==================================================
기존 벤치마크 파일을 건드리지 않는 독립 실행 스크립트.

수정사항:
  - ncnn Vulkan: create_gpu_instance() + fp16 최적화 추가
  - TFLite GPU: 빌드된 delegate .so 경로 지정 가능

Usage:
  python3 run_failed_runtimes.py --model resnet50 --power-mode 10w
  python3 run_failed_runtimes.py --model resnet50 --power-mode 5w
  python3 run_failed_runtimes.py --model resnet50 --power-mode all
"""

import argparse
import json
import csv
import os
import sys
import time
import subprocess
from datetime import datetime

import numpy as np

# Add benchmark dir to path for tegra_parser
BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BENCHMARK_DIR, "benchmark"))

try:
    from tegra_parser import TegraStatsLogger
except ImportError:
    TegraStatsLogger = None
    print("[WARN] tegra_parser not found. Power logging disabled.")


# =====================================================================
#  ncnn Vulkan — FIXED version
# =====================================================================
def benchmark_ncnn_vulkan_fixed(param_path, bin_path, num_warmup=10, num_runs=100):
    """ncnn Vulkan benchmark with proper GPU instance initialization.

    Bug fix: The original run_ncnn.py never calls create_gpu_instance(),
    causing silent CPU fallback despite use_vulkan_compute=True.
    """
    import ncnn

    # CRITICAL: Initialize Vulkan GPU instance BEFORE creating Net
    ncnn.create_gpu_instance()

    gpu_count = ncnn.get_gpu_count()
    if gpu_count < 1:
        ncnn.destroy_gpu_instance()
        raise RuntimeError("No Vulkan GPU found after create_gpu_instance()")

    gpu_info = ncnn.get_gpu_info(0)
    print(f"  Vulkan GPU: {gpu_info.device_name()}")
    print(f"  GPU count:  {gpu_count}")

    net = ncnn.Net()

    # Vulkan compute options — MUST be set before load_param/load_model
    net.opt.use_vulkan_compute = True
    net.opt.use_fp16_packed = True
    net.opt.use_fp16_storage = True
    net.opt.use_fp16_arithmetic = True

    # Reduce CPU thread contention when GPU does the heavy lifting
    net.opt.num_threads = 2

    net.load_param(param_path)
    net.load_model(bin_path)

    # Input tensor: 224x224x3 (standard ImageNet)
    mat_in = ncnn.Mat(224, 224, 3)

    # Warm-up (includes Vulkan shader compilation)
    print(f"  Warming up ({num_warmup} runs, includes shader compilation)...")
    for _ in range(num_warmup):
        ex = net.create_extractor()
        ex.input("input", mat_in)
        _, _ = ex.extract("output")

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ex = net.create_extractor()
        ex.input("input", mat_in)
        _, _ = ex.extract("output")
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies_arr = np.array(latencies)

    model_size_mb = 0.0
    if os.path.exists(bin_path):
        model_size_mb = os.path.getsize(bin_path) / (1024 * 1024)

    result = {
        "runtime": "ncnn_vulkan_fixed",
        "model": os.path.basename(param_path).replace(".param", ""),
        "latency": {
            "mean": float(np.mean(latencies_arr)),
            "std": float(np.std(latencies_arr)),
            "min": float(np.min(latencies_arr)),
            "max": float(np.max(latencies_arr)),
            "p50": float(np.percentile(latencies_arr, 50)),
            "p95": float(np.percentile(latencies_arr, 95)),
            "p99": float(np.percentile(latencies_arr, 99)),
        },
        "memory_mb": 0.0,
        "model_size_mb": round(model_size_mb, 2),
        "num_runs": num_runs,
        "fix_applied": "create_gpu_instance + fp16",
    }

    # NOTE: Do NOT call ncnn.destroy_gpu_instance() here.
    # The ncnn Vulkan allocator segfaults if destroy is called while
    # Python objects still hold references to Vulkan resources.
    # Let the OS handle cleanup when the process exits.

    return result


# =====================================================================
#  TFLite GPU — with custom delegate path
# =====================================================================
def benchmark_tflite_gpu_fixed(model_path, num_warmup=10, num_runs=100,
                                delegate_path=None):
    """TFLite GPU delegate benchmark with explicit .so path.

    Searches for the delegate in multiple locations.
    """
    import tflite_runtime.interpreter as tflite

    # Search order for GPU delegate .so
    search_paths = [
        delegate_path,
        os.path.join(BENCHMARK_DIR, "lib", "libtensorflowlite_gpu_delegate.so"),
        "/usr/local/lib/libtensorflowlite_gpu_delegate.so",
        os.path.expanduser("~/jetson-benchmark/lib/libtensorflowlite_gpu_delegate.so"),
    ]

    delegate = None
    delegate_found = None
    for path in search_paths:
        if path and os.path.exists(path):
            try:
                delegate = tflite.load_delegate(path)
                delegate_found = path
                print(f"  GPU delegate loaded: {path}")
                break
            except Exception as e:
                print(f"  [WARN] Failed to load {path}: {e}")

    if delegate is None:
        raise FileNotFoundError(
            "GPU delegate .so not found. Build it first with build_tflite_gpu.sh.\n"
            f"  Searched: {[p for p in search_paths if p]}"
        )

    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[delegate],
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]
    input_data = np.random.randn(*input_shape).astype(input_dtype)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Warm-up
    print(f"  Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies_arr = np.array(latencies)

    return {
        "runtime": "tflite_gpu_fixed",
        "model": os.path.basename(model_path),
        "latency": {
            "mean": float(np.mean(latencies_arr)),
            "std": float(np.std(latencies_arr)),
            "min": float(np.min(latencies_arr)),
            "max": float(np.max(latencies_arr)),
            "p50": float(np.percentile(latencies_arr, 50)),
            "p95": float(np.percentile(latencies_arr, 95)),
            "p99": float(np.percentile(latencies_arr, 99)),
        },
        "memory_mb": 0.0,
        "model_size_mb": round(model_size_mb, 2),
        "num_runs": num_runs,
        "delegate_path": delegate_found,
    }


# =====================================================================
#  Utility: Power mode switching
# =====================================================================
def set_power_mode(mode):
    """Set Jetson power mode (10w or 5w)."""
    mode_id = "0000" if mode == "10w" else "0001"
    mode_name = "MAXN" if mode == "10w" else "5W"
    try:
        subprocess.run(
            ["sudo", "nvpmodel", "-m", "1" if mode == "5w" else "0"],
            capture_output=True, text=True, timeout=10,
        )
        # Wait for mode to stabilize
        time.sleep(5)
        result = subprocess.run(
            ["sudo", "nvpmodel", "-q"],
            capture_output=True, text=True, timeout=5,
        )
        print(f"  Power mode: {result.stdout.strip()}")
    except Exception as e:
        print(f"  [WARN] Failed to set power mode: {e}")


def get_power_mode():
    """Get current Jetson power mode string."""
    try:
        result = subprocess.run(
            ["sudo", "nvpmodel", "-q"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "NV Power Mode" in line:
                return line.strip()
        return "unknown"
    except Exception:
        return "unknown"


# =====================================================================
#  Main orchestrator
# =====================================================================
def run_benchmark_suite(model_name, power_mode, model_dir, output_dir,
                        num_warmup, num_runs, cool_down, delegate_path,
                        skip_tflite):
    """Run the failed runtimes benchmark suite."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"failed_runtimes_{model_name}_{power_mode}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    param_path = os.path.join(model_dir, f"{model_name}.param")
    bin_path = os.path.join(model_dir, f"{model_name}.bin")
    tflite_path = os.path.join(model_dir, f"{model_name}.tflite")

    print("=" * 60)
    print(f"  Failed Runtimes Benchmark (패자부활전)")
    print(f"  Model:      {model_name}")
    print(f"  Power:      {power_mode} ({get_power_mode()})")
    print(f"  Warmup:     {num_warmup}")
    print(f"  Runs:       {num_runs}")
    print(f"  Cool-down:  {cool_down}s")
    print(f"  Output:     {run_dir}")
    print("=" * 60)

    all_results = []
    tegra_logger = TegraStatsLogger() if TegraStatsLogger else None

    runtimes_to_run = []

    # 1. ncnn Vulkan (always)
    if os.path.exists(param_path):
        runtimes_to_run.append(("ncnn_vulkan_fixed", param_path, bin_path))
    else:
        print(f"  [SKIP] ncnn: {param_path} not found")

    # 2. TFLite GPU (only if delegate exists or explicitly requested)
    if not skip_tflite and os.path.exists(tflite_path):
        runtimes_to_run.append(("tflite_gpu_fixed", tflite_path, None))
    elif skip_tflite:
        print("  [SKIP] TFLite GPU: --skip-tflite flag set (delegate not built yet)")
    else:
        print(f"  [SKIP] TFLite GPU: {tflite_path} not found")

    for i, (runtime_name, model_path, extra_path) in enumerate(runtimes_to_run):
        print(f"\n[{i+1}/{len(runtimes_to_run)}] Benchmarking: {runtime_name}")

        # Thermal stabilization
        if i > 0:
            print(f"  Cooling down for {cool_down}s...")
            time.sleep(cool_down)

        # Start tegrastats logging
        tegra_log = os.path.join(run_dir, f"tegra_{runtime_name}.log")
        if tegra_logger:
            try:
                tegra_logger.start(tegra_log, interval_ms=100)
            except Exception as e:
                print(f"  [WARN] tegrastats failed: {e}")
                tegra_logger = None

        start_time = time.time()
        result = None

        try:
            if runtime_name == "ncnn_vulkan_fixed":
                result = benchmark_ncnn_vulkan_fixed(
                    model_path, extra_path, num_warmup, num_runs
                )
            elif runtime_name == "tflite_gpu_fixed":
                result = benchmark_tflite_gpu_fixed(
                    model_path, num_warmup, num_runs, delegate_path
                )
        except Exception as e:
            print(f"  [ERROR] {runtime_name}: {e}")
            import traceback
            traceback.print_exc()

        elapsed = time.time() - start_time

        # Stop tegrastats
        tegra_summary = {}
        if tegra_logger:
            tegra_logger.stop()
            tegra_summary = tegra_logger.summary(tegra_log)

        if result:
            result["power_mode"] = power_mode
            result["tegrastats"] = tegra_summary
            result["wall_time_s"] = round(elapsed, 2)
            all_results.append(result)

            # Print summary
            lat = result["latency"]
            fps = 1000.0 / lat["mean"] if lat["mean"] > 0 else 0
            print(f"\n  --- {runtime_name} ---")
            print(f"  Mean: {lat['mean']:.2f} ms | P95: {lat['p95']:.2f} ms | Std: {lat['std']:.2f} ms")
            print(f"  FPS:  {fps:.1f}")
            if tegra_summary and "pom_5v_in_current_mw" in tegra_summary:
                pwr = tegra_summary["pom_5v_in_current_mw"]
                print(f"  Power: avg {pwr['mean']:.0f} mW, max {pwr['max']:.0f} mW")

    # Save results
    if all_results:
        json_path = os.path.join(run_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  JSON saved: {json_path}")

        csv_path = os.path.join(run_dir, "summary.csv")
        csv_rows = []
        for r in all_results:
            row = {
                "model": r.get("model", model_name),
                "runtime": r.get("runtime", ""),
                "power_mode": r.get("power_mode", ""),
                "mean_ms": r["latency"]["mean"],
                "std_ms": r["latency"]["std"],
                "min_ms": r["latency"]["min"],
                "max_ms": r["latency"]["max"],
                "p50_ms": r["latency"]["p50"],
                "p95_ms": r["latency"]["p95"],
                "p99_ms": r["latency"]["p99"],
                "fps": round(1000.0 / r["latency"]["mean"], 2) if r["latency"]["mean"] > 0 else 0,
                "memory_mb": r.get("memory_mb", 0),
                "model_size_mb": r.get("model_size_mb", 0),
            }
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

    # Final summary
    print(f"\n{'='*70}")
    print(f"  패자부활전 Summary: {model_name} @ {power_mode}")
    print(f"{'='*70}")
    print(f"  {'Runtime':<25} {'Mean(ms)':<10} {'P95(ms)':<10} {'FPS':<10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for r in all_results:
        lat = r["latency"]
        fps = 1000.0 / lat["mean"] if lat["mean"] > 0 else 0
        print(f"  {r['runtime']:<25} {lat['mean']:<10.2f} {lat['p95']:<10.2f} {fps:<10.1f}")

    # Compare with original results
    print(f"\n  Original benchmark results (for comparison):")
    print(f"  {'ncnn_vulkan (original)':<25} {'785.00':<10} {'859.64':<10} {'1.3':<10}")
    print(f"  {'tflite_gpu (original)':<25} {'784.88':<10} {'810.29':<10} {'1.3':<10}")
    print()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="패자부활전: Failed runtimes re-benchmark",
    )
    parser.add_argument(
        "--model", default="resnet50",
        choices=["mobilenetv3_small", "resnet50"],
        help="Model to benchmark",
    )
    parser.add_argument(
        "--power-mode", default="10w",
        choices=["10w", "5w", "all"],
        help="Power mode (or 'all' for both)",
    )
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--cool-down", type=int, default=60)
    parser.add_argument(
        "--model-dir",
        default=os.path.join(BENCHMARK_DIR, "models"),
        help="Directory containing model files",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(BENCHMARK_DIR, "results"),
    )
    parser.add_argument(
        "--delegate-path",
        default=None,
        help="Path to libtensorflowlite_gpu_delegate.so",
    )
    parser.add_argument(
        "--skip-tflite",
        action="store_true",
        help="Skip TFLite GPU if delegate not yet built",
    )
    args = parser.parse_args()

    power_modes = ["10w", "5w"] if args.power_mode == "all" else [args.power_mode]

    for pm in power_modes:
        print(f"\n{'#'*60}")
        print(f"  Setting power mode: {pm}")
        print(f"{'#'*60}")
        set_power_mode(pm)

        # Extra stabilization wait
        print(f"  Waiting 30s for thermal stabilization...")
        time.sleep(30)

        run_benchmark_suite(
            model_name=args.model,
            power_mode=pm,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            cool_down=args.cool_down,
            delegate_path=args.delegate_path,
            skip_tflite=args.skip_tflite,
        )

    print("\n  All experiments complete!")


if __name__ == "__main__":
    main()
