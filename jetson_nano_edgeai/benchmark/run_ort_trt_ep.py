#!/usr/bin/env python3
"""Run ONNX Runtime TensorRT EP benchmark for both models."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_onnxrt import benchmark_onnxrt
from tegra_parser import TegraStatsLogger


def run_experiments(power_mode_label):
    """Run ORT TRT EP experiments for both models at current power setting."""
    output_dir = os.path.expanduser("~/jetson-benchmark/results/ort_trt_ep")
    model_dir = os.path.expanduser("~/jetson-benchmark/models")
    os.makedirs(output_dir, exist_ok=True)

    models = [
        ("mobilenetv3_small", os.path.join(model_dir, "mobilenetv3_small.onnx")),
        ("resnet50", os.path.join(model_dir, "resnet50.onnx")),
    ]

    results = []
    tegra = TegraStatsLogger()

    for model_name, onnx_path in models:
        print(f"\n=== {model_name} ORT TRT EP @{power_mode_label} ===")

        tegra_log = os.path.join(
            output_dir, f"tegra_{model_name}_ort_trt_{power_mode_label}.log"
        )

        try:
            tegra.start(tegra_log, interval_ms=100)
        except Exception as e:
            print(f"  tegrastats warn: {e}")

        start = time.time()
        result = benchmark_onnxrt(onnx_path, "TensorrtExecutionProvider", 10, 100)
        elapsed = time.time() - start

        tegra_summary = {}
        try:
            tegra.stop()
            tegra_summary = tegra.summary(tegra_log)
        except Exception as e:
            print(f"  tegra summary warn: {e}")

        result["power_mode"] = power_mode_label
        result["tegrastats"] = tegra_summary
        result["wall_time_s"] = round(elapsed, 2)
        result["model_name"] = model_name
        results.append(result)

        lat = result["latency"]
        fps = 1000.0 / lat["mean"] if lat["mean"] > 0 else 0
        print(f"  Mean: {lat['mean']:.2f} ms | P95: {lat['p95']:.2f} ms | FPS: {fps:.1f}")

        tegra_s = result.get("tegrastats", {})
        if "pom_5v_in_current_mw" in tegra_s:
            pwr = tegra_s["pom_5v_in_current_mw"]
            print(f"  Power: avg {pwr['mean']:.0f} mW, max {pwr['max']:.0f} mW")

        # Cool down between models
        if model_name != models[-1][0]:
            print("  Cooling 60s...")
            time.sleep(60)

    out_path = os.path.join(output_dir, f"results_{power_mode_label}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--power-mode", default="10w", choices=["10w", "5w"])
    args = parser.parse_args()

    run_experiments(args.power_mode)
