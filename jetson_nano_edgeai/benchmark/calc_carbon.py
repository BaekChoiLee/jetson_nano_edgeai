#!/usr/bin/env python3
"""Calculate per-inference energy (mJ) and CO2 emissions (mgCO2) from benchmark results.

Formula:
    Energy(mJ) = power(mW) * latency(ms) / 1000   [mW * ms = uJ, /1000 = mJ]
    Energy_per_run(Wh) = Energy(mJ) / 3_600_000
    CO2(mgCO2) = Energy(Wh) * carbon_intensity(gCO2/kWh) * 1000 / 1000
               = Energy(Wh) * carbon_intensity(gCO2/kWh)
               = Energy(mJ) / 3_600_000 * 459 * 1000  [convert to mgCO2]

Korea grid carbon intensity: 0.459 kgCO2/kWh = 459 gCO2/kWh (2023, Ministry of Environment)
"""

import argparse
import csv
import json
import os
import sys


CARBON_INTENSITY_G_PER_KWH = 459  # Korea, 2023


def calc_carbon_from_results(results_dir):
    """Calculate energy and carbon for each experiment from results.json files.

    Returns list of dicts with energy/carbon data.
    """
    rows = []

    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        json_path = os.path.join(entry_path, "results.json")

        if not os.path.isdir(entry_path) or not os.path.exists(json_path):
            continue

        with open(json_path, "r") as f:
            experiments = json.load(f)

        for exp in experiments:
            runtime = exp.get("runtime", "unknown")
            model = exp.get("model", "unknown")
            power_mode = exp.get("power_mode", "unknown")
            num_runs = exp.get("num_runs", 100)
            latency_mean_ms = exp.get("latency", {}).get("mean", 0)

            tegra = exp.get("tegrastats", {})
            power_data = tegra.get("pom_5v_in_current_mw", {})
            power_mean_mw = power_data.get("mean", 0)

            if latency_mean_ms <= 0 or power_mean_mw <= 0:
                continue

            # Per-inference energy: power(mW) * time(ms) = uJ, convert to mJ
            energy_per_inference_mj = power_mean_mw * latency_mean_ms / 1000.0

            # Total energy for all runs
            total_energy_mj = energy_per_inference_mj * num_runs

            # Convert to Wh: 1 Wh = 3,600,000 mJ
            energy_per_inference_wh = energy_per_inference_mj / 3_600_000

            # CO2: Energy(kWh) * carbon_intensity(gCO2/kWh) * 1000 = mgCO2
            co2_per_inference_mg = (energy_per_inference_wh / 1000) * CARBON_INTENSITY_G_PER_KWH * 1000

            # Total CO2 for all runs
            total_co2_mg = co2_per_inference_mg * num_runs

            rows.append({
                "model": model,
                "runtime": runtime,
                "power_mode": power_mode,
                "num_runs": num_runs,
                "latency_mean_ms": round(latency_mean_ms, 2),
                "power_mean_mw": round(power_mean_mw, 1),
                "energy_per_inference_mj": round(energy_per_inference_mj, 4),
                "total_energy_mj": round(total_energy_mj, 2),
                "co2_per_inference_mg": round(co2_per_inference_mg, 6),
                "total_co2_mg": round(total_co2_mg, 4),
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Calculate carbon emissions from benchmark results")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "results"),
        help="Directory containing results subdirectories",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: results_dir/carbon_summary.csv)",
    )
    parser.add_argument(
        "--carbon-intensity",
        type=float,
        default=CARBON_INTENSITY_G_PER_KWH,
        help=f"Carbon intensity in gCO2/kWh (default: {CARBON_INTENSITY_G_PER_KWH})",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        print(f"[ERROR] Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Scanning: {results_dir}")
    print(f"Carbon intensity: {args.carbon_intensity} gCO2/kWh")

    rows = calc_carbon_from_results(results_dir)

    if not rows:
        print("[WARN] No data found.")
        return

    output_path = args.output or os.path.join(results_dir, "carbon_summary.csv")
    fieldnames = [
        "model", "runtime", "power_mode", "num_runs",
        "latency_mean_ms", "power_mean_mw",
        "energy_per_inference_mj", "total_energy_mj",
        "co2_per_inference_mg", "total_co2_mg",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} entries to: {output_path}")

    # Print summary table
    print(f"\n{'Model':<20} {'Runtime':<15} {'Power':<6} {'Energy(mJ)':<12} {'CO2(mg)'}")
    print("-" * 70)
    for r in rows:
        print(f"{r['model']:<20} {r['runtime']:<15} {r['power_mode']:<6} "
              f"{r['energy_per_inference_mj']:<12.4f} {r['co2_per_inference_mg']:.6f}")


if __name__ == "__main__":
    main()
