#!/usr/bin/env python3
"""Extract peak/mean/min RAM usage from tegrastats logs.

Works both on Jetson (with live logs) and on PC (with copied results).
Parses tegra_*.log files from each results directory.
"""

import argparse
import csv
import os
import re
import sys


def parse_ram_from_log(log_path):
    """Parse RAM usage values from a tegrastats log file.

    Returns list of ram_used_mb values.
    """
    ram_values = []
    with open(log_path, "r") as f:
        for line in f:
            match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
            if match:
                ram_values.append(int(match.group(1)))
    return ram_values


def collect_memory(results_dir):
    """Collect memory stats from all results subdirectories.

    Returns list of dicts with memory statistics per runtime.
    """
    rows = []

    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        # Parse directory name: model_powermode_timestamp
        # e.g., resnet50_10w_20260327_193322
        parts = entry.rsplit("_", 2)
        if len(parts) < 3:
            continue

        model_and_power = entry.rsplit("_", 2)[0]
        # Find power mode suffix
        if "_10w" in model_and_power:
            model = model_and_power.replace("_10w", "")
            power_mode = "10w"
        elif "_5w" in model_and_power:
            model = model_and_power.replace("_5w", "")
            power_mode = "5w"
        else:
            continue

        # Find all tegra_*.log files
        for fname in sorted(os.listdir(entry_path)):
            if not fname.startswith("tegra_") or not fname.endswith(".log"):
                continue

            runtime = fname.replace("tegra_", "").replace(".log", "")
            log_path = os.path.join(entry_path, fname)

            ram_values = parse_ram_from_log(log_path)
            if not ram_values:
                continue

            rows.append({
                "model": model,
                "runtime": runtime,
                "power_mode": power_mode,
                "ram_min_mb": min(ram_values),
                "ram_max_mb": max(ram_values),
                "ram_mean_mb": round(sum(ram_values) / len(ram_values), 1),
                "ram_delta_mb": max(ram_values) - min(ram_values),
                "num_samples": len(ram_values),
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect memory stats from tegrastats logs")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "results"),
        help="Directory containing results subdirectories",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: results_dir/memory_summary.csv)",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        print(f"[ERROR] Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Scanning: {results_dir}")
    rows = collect_memory(results_dir)

    if not rows:
        print("[WARN] No memory data found.")
        return

    output_path = args.output or os.path.join(results_dir, "memory_summary.csv")
    fieldnames = ["model", "runtime", "power_mode", "ram_min_mb", "ram_max_mb",
                  "ram_mean_mb", "ram_delta_mb", "num_samples"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} entries to: {output_path}")

    # Print summary table
    print(f"\n{'Model':<20} {'Runtime':<18} {'Power':<6} {'Min MB':<8} {'Max MB':<8} {'Delta MB'}")
    print("-" * 80)
    for r in rows:
        print(f"{r['model']:<20} {r['runtime']:<18} {r['power_mode']:<6} "
              f"{r['ram_min_mb']:<8} {r['ram_max_mb']:<8} {r['ram_delta_mb']}")


if __name__ == "__main__":
    main()
