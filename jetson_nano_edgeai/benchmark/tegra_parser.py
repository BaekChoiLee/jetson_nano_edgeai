#!/usr/bin/env python3
"""tegrastats log parser and background logger for Jetson Nano."""

import os
import re
import subprocess
import signal
import time
import threading
import numpy as np


class TegraStatsLogger:
    """Start/stop tegrastats and parse its output.

    Usage:
        logger = TegraStatsLogger()
        logger.start("log.txt", interval_ms=100)
        # ... run inference ...
        logger.stop()
        summary = logger.parse("log.txt")
    """

    def __init__(self):
        self._process = None
        self._log_file = None

    def start(self, log_file, interval_ms=100):
        """Start tegrastats logging to file."""
        self._log_file = log_file
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        with open(log_file, "w") as f:
            self._process = subprocess.Popen(
                ["sudo", "tegrastats", "--interval", str(interval_ms)],
                stdout=f,
                stderr=subprocess.DEVNULL,
            )
        # Brief wait for first sample
        time.sleep(0.3)

    def stop(self):
        """Stop tegrastats process (started with sudo)."""
        if self._process:
            try:
                subprocess.run(
                    ["sudo", "kill", "-SIGINT", str(self._process.pid)],
                    timeout=3,
                    capture_output=True,
                )
                self._process.wait(timeout=5)
            except (subprocess.TimeoutExpired, Exception):
                subprocess.run(
                    ["sudo", "kill", "-9", str(self._process.pid)],
                    timeout=3,
                    capture_output=True,
                )
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
            self._process = None

    def parse(self, log_file=None):
        """Parse tegrastats log file into structured data.

        Returns list of dicts, one per sample.
        """
        log_file = log_file or self._log_file
        if not log_file or not os.path.exists(log_file):
            return []

        samples = []
        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = self._parse_line(line)
                if sample:
                    samples.append(sample)
        return samples

    def _parse_line(self, line):
        """Parse a single tegrastats output line.

        Example line:
        RAM 2345/3964MB (lfb 1x4MB) SWAP 123/4096MB (cached 0MB)
        CPU [25%@1479,30%@1479,20%@1479,35%@1479] EMC_FREQ 0%@1600
        GR3D_FREQ 50%@921 APE 25 PLL@41C CPU@42.5C PMIC@100C
        GPU@40C AO@44.5C thermal@41.5C POM_5V_IN 4567/4567
        POM_5V_GPU 1234/1234 POM_5V_CPU 890/890
        """
        sample = {}

        # RAM: current/total MB
        ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if ram_match:
            sample["ram_used_mb"] = int(ram_match.group(1))
            sample["ram_total_mb"] = int(ram_match.group(2))

        # SWAP
        swap_match = re.search(r"SWAP\s+(\d+)/(\d+)MB", line)
        if swap_match:
            sample["swap_used_mb"] = int(swap_match.group(1))
            sample["swap_total_mb"] = int(swap_match.group(2))

        # CPU utilization: [25%@1479,30%@1479,...]
        cpu_match = re.search(r"CPU\s+\[([^\]]+)\]", line)
        if cpu_match:
            cpu_parts = cpu_match.group(1).split(",")
            cpu_utils = []
            cpu_freqs = []
            for part in cpu_parts:
                pct_match = re.match(r"(\d+)%@(\d+)", part.strip())
                if pct_match:
                    cpu_utils.append(int(pct_match.group(1)))
                    cpu_freqs.append(int(pct_match.group(2)))
                elif part.strip() == "off":
                    cpu_utils.append(0)
                    cpu_freqs.append(0)
            sample["cpu_util_pct"] = cpu_utils
            sample["cpu_avg_util_pct"] = np.mean(cpu_utils) if cpu_utils else 0
            sample["cpu_freq_mhz"] = cpu_freqs

        # GPU utilization: GR3D_FREQ X%@freq
        gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%@(\d+)", line)
        if gpu_match:
            sample["gpu_util_pct"] = int(gpu_match.group(1))
            sample["gpu_freq_mhz"] = int(gpu_match.group(2))

        # Temperatures: CPU@42.5C GPU@40C thermal@41.5C
        for temp_name in ["CPU", "GPU", "PMIC", "AO", "thermal", "PLL"]:
            temp_match = re.search(rf"{temp_name}@([\d.]+)C", line)
            if temp_match:
                sample[f"temp_{temp_name.lower()}_c"] = float(temp_match.group(1))

        # Power: POM_5V_IN current/average (mW)
        for power_name in ["POM_5V_IN", "POM_5V_GPU", "POM_5V_CPU"]:
            power_match = re.search(rf"{power_name}\s+(\d+)/(\d+)", line)
            if power_match:
                key = power_name.lower()
                sample[f"{key}_current_mw"] = int(power_match.group(1))
                sample[f"{key}_average_mw"] = int(power_match.group(2))

        return sample if sample else None

    def summary(self, log_file=None):
        """Compute summary statistics from tegrastats log.

        Returns dict with mean/max for each metric.
        """
        samples = self.parse(log_file)
        if not samples:
            return {"error": "No samples parsed"}

        result = {"num_samples": len(samples)}

        # Aggregate numeric fields
        numeric_keys = set()
        for s in samples:
            for k, v in s.items():
                if isinstance(v, (int, float)):
                    numeric_keys.add(k)

        for key in sorted(numeric_keys):
            values = [s[key] for s in samples if key in s]
            if values:
                result[key] = {
                    "mean": round(float(np.mean(values)), 2),
                    "max": round(float(np.max(values)), 2),
                    "min": round(float(np.min(values)), 2),
                }

        return result


def run_with_logging(func, log_file, interval_ms=100):
    """Convenience: run a function while logging tegrastats.

    Args:
        func: Callable to run (benchmark function)
        log_file: Path for tegrastats log
        interval_ms: Logging interval in ms

    Returns:
        (func_result, tegra_summary)
    """
    logger = TegraStatsLogger()
    logger.start(log_file, interval_ms)
    try:
        result = func()
    finally:
        logger.stop()
    summary = logger.summary(log_file)
    return result, summary


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--parse", help="Parse existing tegrastats log file")
    parser.add_argument("--record", type=int, default=0, help="Record for N seconds")
    parser.add_argument("--output", default="tegrastats.log", help="Log output file")
    parser.add_argument("--interval", type=int, default=100, help="Interval in ms")
    args = parser.parse_args()

    if args.parse:
        logger = TegraStatsLogger()
        summary = logger.summary(args.parse)
        print(json.dumps(summary, indent=2))
    elif args.record > 0:
        logger = TegraStatsLogger()
        print(f"Recording tegrastats for {args.record}s to {args.output}...")
        logger.start(args.output, args.interval)
        time.sleep(args.record)
        logger.stop()
        summary = logger.summary(args.output)
        print(json.dumps(summary, indent=2))
    else:
        print("Usage:")
        print(f"  {__file__} --parse <log_file>")
        print(f"  {__file__} --record <seconds> --output <file>")
