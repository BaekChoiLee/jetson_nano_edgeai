#!/usr/bin/env python3
"""Consolidate all benchmark results into a single CSV.

Handles:
  - Multiple results directories (scattered by runtime)
  - Model name normalization (strips file extensions)
  - Duplicate deduplication (keeps latest result per key)
  - Inline carbon emissions calculation
  - Inline memory delta from tegrastats

Output: all_results.csv + carbon_summary.csv
"""

import argparse
import csv
import json
import os
import re
import sys

# Korea 2023 carbon intensity: 0.459 kgCO2/kWh (환경부)
CARBON_INTENSITY_KG_PER_KWH = 0.459


def normalize_model_name(model_name):
    """Strip file extensions from model names.

    Examples:
      mobilenetv3_small.onnx -> mobilenetv3_small
      resnet50_fp16.engine -> resnet50
      mobilenetv3_small.tflite -> mobilenetv3_small
    """
    # Remove .onnx, .tflite, .engine extensions
    name = re.sub(r'\.(onnx|tflite|engine)$', '', model_name)
    # Remove precision suffixes like _fp32, _fp16, _int8 (for TRT engine names)
    name = re.sub(r'_(fp32|fp16|int8)$', '', name)
    return name


def load_all_experiments(results_dir):
    """Load all results.json files, deduplicate by (model, runtime, power_mode).

    For duplicates, keeps the entry from the latest directory (by timestamp).
    """
    # Collect all experiments with their directory timestamps
    raw_experiments = []

    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        json_path = os.path.join(entry_path, "results.json")

        if not os.path.isdir(entry_path) or not os.path.exists(json_path):
            continue

        with open(json_path, "r") as f:
            experiments = json.load(f)

        for exp in experiments:
            raw_experiments.append((entry, exp))

    # Deduplicate: keep latest directory for each (normalized_model, runtime, power_mode)
    best = {}
    for dir_name, exp in raw_experiments:
        model_raw = exp.get("model", "unknown")
        model = normalize_model_name(model_raw)
        runtime = exp.get("runtime", "unknown")
        power_mode = exp.get("power_mode", "unknown")
        key = (model, runtime, power_mode)

        if key not in best or dir_name > best[key][0]:
            best[key] = (dir_name, exp)

    print(f"Loaded {len(raw_experiments)} raw experiments, "
          f"deduplicated to {len(best)} unique combinations")

    return [(dir_name, exp) for dir_name, exp in best.values()]


def calc_carbon(power_mw, latency_ms, num_runs):
    """Calculate energy and CO2 per inference.

    Returns (energy_per_inference_mj, co2_per_inference_mg,
             total_energy_mj, total_co2_mg)
    """
    if power_mw <= 0 or latency_ms <= 0:
        return (0, 0, 0, 0)

    # Energy per inference: power(mW) * time(ms) / 1000 = energy(mJ)
    energy_per_mj = power_mw * latency_ms / 1000.0
    total_energy_mj = energy_per_mj * num_runs

    # CO2: energy(mJ) -> Wh -> kWh -> kgCO2 -> mgCO2
    energy_wh = energy_per_mj / 3_600_000.0
    co2_per_mg = energy_wh * CARBON_INTENSITY_KG_PER_KWH * 1_000_000.0
    total_co2_mg = co2_per_mg * num_runs

    return (
        round(energy_per_mj, 4),
        round(co2_per_mg, 6),
        round(total_energy_mj, 2),
        round(total_co2_mg, 4),
    )


def consolidate(results_dir, output_path):
    """Merge all data sources into a single CSV."""
    experiments = load_all_experiments(results_dir)

    rows = []
    carbon_rows = []

    for dir_name, exp in sorted(experiments, key=lambda x: (
        normalize_model_name(x[1].get("model", "")),
        x[1].get("runtime", ""),
        x[1].get("power_mode", ""),
    )):
        model_raw = exp.get("model", "unknown")
        model = normalize_model_name(model_raw)
        runtime = exp.get("runtime", "unknown")
        power_mode = exp.get("power_mode", "unknown")

        latency = exp.get("latency", {})
        tegra = exp.get("tegrastats", {})
        num_runs = exp.get("num_runs", 100)

        # Power from tegrastats
        power_in_mean = tegra.get("pom_5v_in_current_mw", {}).get("mean", 0)
        latency_mean = latency.get("mean", 0)

        # Calculate carbon inline
        energy_per, co2_per, total_energy, total_co2 = calc_carbon(
            power_in_mean, latency_mean, num_runs
        )

        # Memory delta from tegrastats
        ram_info = tegra.get("ram_used_mb", {})
        ram_min = ram_info.get("min", 0)
        ram_max = ram_info.get("max", 0)
        ram_delta = round(ram_max - ram_min, 1) if ram_max and ram_min else 0

        row = {
            "model": model,
            "runtime": runtime,
            "power_mode": power_mode,
            "num_runs": num_runs,
            "model_size_mb": exp.get("model_size_mb", 0),
            # Latency
            "latency_mean_ms": round(latency.get("mean", 0), 2),
            "latency_std_ms": round(latency.get("std", 0), 2),
            "latency_min_ms": round(latency.get("min", 0), 2),
            "latency_max_ms": round(latency.get("max", 0), 2),
            "latency_p50_ms": round(latency.get("p50", 0), 2),
            "latency_p95_ms": round(latency.get("p95", 0), 2),
            "latency_p99_ms": round(latency.get("p99", 0), 2),
            "fps": round(1000.0 / latency_mean, 2) if latency_mean > 0 else 0,
            # Memory
            "memory_gpu_mb": exp.get("memory_mb", 0),
            "ram_mean_mb": round(ram_info.get("mean", 0), 1),
            "ram_max_mb": ram_max,
            "ram_min_mb": ram_min,
            "ram_delta_mb": ram_delta,
            # Power
            "power_in_mean_mw": round(power_in_mean, 2),
            "power_in_max_mw": tegra.get("pom_5v_in_current_mw", {}).get("max", 0),
            "power_cpu_mean_mw": round(tegra.get("pom_5v_cpu_current_mw", {}).get("mean", 0), 2),
            "power_gpu_mean_mw": round(tegra.get("pom_5v_gpu_current_mw", {}).get("mean", 0), 2),
            # Utilization
            "gpu_util_mean_pct": round(tegra.get("gpu_util_pct", {}).get("mean", 0), 2),
            "cpu_util_mean_pct": round(tegra.get("cpu_avg_util_pct", {}).get("mean", 0), 2),
            # Temperature
            "temp_cpu_mean_c": tegra.get("temp_cpu_c", {}).get("mean", 0),
            "temp_gpu_mean_c": tegra.get("temp_gpu_c", {}).get("mean", 0),
            # Wall time
            "wall_time_s": exp.get("wall_time_s", 0),
            # Carbon
            "energy_per_inference_mj": energy_per,
            "co2_per_inference_mg": co2_per,
            "total_energy_mj": total_energy,
            "total_co2_mg": total_co2,
            # Source directory
            "source_dir": dir_name,
        }

        rows.append(row)

        carbon_rows.append({
            "model": model,
            "runtime": runtime,
            "power_mode": power_mode,
            "num_runs": num_runs,
            "latency_mean_ms": round(latency_mean, 2),
            "power_mean_mw": round(power_in_mean, 2),
            "energy_per_inference_mj": energy_per,
            "total_energy_mj": total_energy,
            "co2_per_inference_mg": co2_per,
            "total_co2_mg": total_co2,
        })

    if not rows:
        print("[WARN] No data to consolidate.")
        return

    # Write main CSV
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write carbon summary CSV
    carbon_path = os.path.join(os.path.dirname(output_path), "carbon_summary.csv")
    carbon_fields = list(carbon_rows[0].keys())
    with open(carbon_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=carbon_fields)
        writer.writeheader()
        writer.writerows(carbon_rows)

    print(f"\nConsolidated {len(rows)} experiments to: {output_path}")
    print(f"Carbon summary: {carbon_path}")

    # Print summary table
    print(f"\n{'Model':<22} {'Runtime':<16} {'Power':<6} {'Latency(ms)':<12} "
          f"{'FPS':<8} {'Power(mW)':<10} {'RAM Δ(MB)':<10} {'E/inf(mJ)'}")
    print("-" * 110)
    for r in rows:
        print(f"{r['model']:<22} {r['runtime']:<16} {r['power_mode']:<6} "
              f"{r['latency_mean_ms']:<12} {r['fps']:<8} "
              f"{r['power_in_mean_mw']:<10} {r['ram_delta_mb']:<10} "
              f"{r['energy_per_inference_mj']}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate benchmark results")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "results_jetson_full"),
        help="Directory containing results subdirectories",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <project>/results/results/all_results.csv)",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    # Output to the main results directory for chart generation
    default_output = os.path.join(
        os.path.dirname(__file__), "..", "results", "results", "all_results.csv"
    )
    output_path = args.output or os.path.abspath(default_output)

    consolidate(results_dir, output_path)


if __name__ == "__main__":
    main()
