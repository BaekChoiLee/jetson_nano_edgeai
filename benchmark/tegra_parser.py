#!/usr/bin/env python3
# =============================================================================
# tegra_parser.py — Jetson Nano tegrastats 로그 파서 및 백그라운드 로거
# =============================================================================
# 역할: NVIDIA Jetson Nano의 tegrastats 유틸리티를 백그라운드 프로세스로
#        시작/정지하고, 그 출력(텍스트 로그)을 구조화된 데이터로 파싱한다.
#
# 파싱하는 메트릭:
#   - RAM: 사용량/전체 (MB)
#   - SWAP: 사용량/전체 (MB)
#   - CPU: 코어별 사용률(%), 코어별 주파수(MHz), 평균 사용률
#   - GPU: 사용률(%), 주파수(MHz)  — GR3D_FREQ 필드
#   - 온도: CPU, GPU, PMIC, AO, thermal, PLL (섭씨)
#   - 전력: POM_5V_IN, POM_5V_GPU, POM_5V_CPU (현재값/평균값, mW)
#
# summary() 메서드로 전체 로그의 평균/최대/최소 통계를 계산할 수 있다.
#
# 사용법:
#   # 기존 로그 파싱
#   python tegra_parser.py --parse tegrastats.log
#
#   # N초간 녹화 후 요약 출력
#   python tegra_parser.py --record 10 --output tegrastats.log
#
#   # 코드에서 사용 (벤치마크 함수 실행 중 로깅)
#   result, summary = run_with_logging(my_benchmark_func, "log.txt")
# =============================================================================

"""tegrastats log parser and background logger for Jetson Nano."""

import os
import re
import subprocess
import time
import numpy as np


class TegraStatsLogger:
    """tegrastats 프로세스를 시작/정지하고 출력을 파싱하는 클래스.

    tegrastats는 NVIDIA Jetson 디바이스 전용 시스템 모니터링 유틸리티로,
    CPU/GPU 사용률, 메모리, 온도, 전력 등을 실시간으로 출력한다.
    sudo 권한이 필요하다.

    사용 패턴:
        logger = TegraStatsLogger()
        logger.start("log.txt", interval_ms=100)
        # ... 추론 벤치마크 실행 ...
        logger.stop()
        summary = logger.parse("log.txt")
    """

    def __init__(self):
        self._process = None      # 유지 호환용 (daemon 모드 사용)
        self._log_file = None     # 로그 저장 경로
        self._log_fp = None       # tegrastats stdout 파일 핸들
        self._launch_cmd = None   # 실제로 성공한 tegrastats 실행 커맨드

    def start(self, log_file, interval_ms=100):
        """tegrastats 백그라운드 로깅을 시작한다.

        Args:
            log_file: 로그를 저장할 파일 경로
            interval_ms: 샘플링 간격 (밀리초, 기본 100ms = 초당 10회)
        """
        self._log_file = log_file
        # 로그 파일의 부모 디렉토리 생성 (없으면)
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # tegrastats daemon 모드 사용:
        # - stdout 파이프 방식은 환경에 따라 샘플 미기록/프로세스 누수 가능
        # - --logfile + --start/--stop 이 Jetson에서 더 안정적
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
            except Exception:
                pass

        # 남아있는 tegrastats가 있으면 먼저 중지 시도
        for stop_cmd in (["tegrastats", "--stop"], ["sudo", "-n", "tegrastats", "--stop"]):
            try:
                subprocess.run(stop_cmd, capture_output=True, timeout=3)
            except Exception:
                pass

        abs_log_file = os.path.abspath(log_file)
        candidates = [
            ["sudo", "-n", "tegrastats", "--interval", str(interval_ms), "--logfile", abs_log_file, "--start"],
        ]
        last_err = None
        started = False
        for cmd in candidates:
            try:
                p = subprocess.run(cmd, capture_output=True, timeout=5)
                if p.returncode == 0:
                    started = True
                    self._launch_cmd = " ".join(cmd)
                    break
                last_err = RuntimeError(
                    f"rc={p.returncode}, stdout={p.stdout.decode('utf-8','replace')}, "
                    f"stderr={p.stderr.decode('utf-8','replace')}"
                )
            except Exception as e:
                last_err = e
        if not started:
            raise RuntimeError(f"Failed to launch tegrastats daemon: {last_err}")

        # 첫 샘플이 기록될 시간 확보
        time.sleep(0.5)

    def stop(self):
        """tegrastats 프로세스를 정지한다.

        daemon 모드에서 tegrastats --stop 으로 중지한다.
        """
        for stop_cmd in (["sudo", "-n", "tegrastats", "--stop"],):
            try:
                p = subprocess.run(stop_cmd, capture_output=True, timeout=5)
                if p.returncode == 0:
                    break
            except Exception:
                pass

        self._process = None
        if self._log_fp:
            try:
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None

    def parse(self, log_file=None):
        """tegrastats 로그 파일을 구조화된 데이터로 파싱한다.

        Args:
            log_file: 파싱할 로그 파일 경로 (None이면 start()에서 사용한 경로)

        Returns:
            딕셔너리 리스트 — 각 샘플(라인)당 하나의 딕셔너리.
            키 예시: ram_used_mb, cpu_util_pct, gpu_util_pct, temp_cpu_c 등
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
        """tegrastats 출력의 한 줄을 파싱한다.

        tegrastats 출력 예시:
        RAM 2345/3964MB (lfb 1x4MB) SWAP 123/4096MB (cached 0MB)
        CPU [25%@1479,30%@1479,20%@1479,35%@1479] EMC_FREQ 0%@1600
        GR3D_FREQ 50%@921 APE 25 PLL@41C CPU@42.5C PMIC@100C
        GPU@40C AO@44.5C thermal@41.5C POM_5V_IN 4567/4567
        POM_5V_GPU 1234/1234 POM_5V_CPU 890/890

        Returns:
            파싱된 메트릭 딕셔너리, 또는 파싱 실패 시 None
        """
        sample = {}

        # --- RAM 사용량 파싱 ---
        # "RAM 2345/3964MB" → 현재 사용량 / 전체 용량 (MB)
        ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if ram_match:
            sample["ram_used_mb"] = int(ram_match.group(1))
            sample["ram_total_mb"] = int(ram_match.group(2))

        # --- SWAP 사용량 파싱 ---
        # "SWAP 123/4096MB" → 현재 사용량 / 전체 용량 (MB)
        swap_match = re.search(r"SWAP\s+(\d+)/(\d+)MB", line)
        if swap_match:
            sample["swap_used_mb"] = int(swap_match.group(1))
            sample["swap_total_mb"] = int(swap_match.group(2))

        # --- CPU 사용률 파싱 ---
        # "CPU [25%@1479,30%@1479,20%@1479,35%@1479]"
        # 각 코어의 사용률(%)과 주파수(MHz)를 추출
        # Jetson Nano는 4코어 ARM Cortex-A57
        cpu_match = re.search(r"CPU\s+\[([^\]]+)\]", line)
        if cpu_match:
            cpu_parts = cpu_match.group(1).split(",")
            cpu_utils = []    # 코어별 사용률 (%)
            cpu_freqs = []    # 코어별 주파수 (MHz)
            for part in cpu_parts:
                pct_match = re.match(r"(\d+)%@(\d+)", part.strip())
                if pct_match:
                    cpu_utils.append(int(pct_match.group(1)))
                    cpu_freqs.append(int(pct_match.group(2)))
                elif part.strip() == "off":
                    # 비활성화된 코어는 0으로 기록
                    cpu_utils.append(0)
                    cpu_freqs.append(0)
            sample["cpu_util_pct"] = cpu_utils          # 코어별 사용률 리스트
            sample["cpu_avg_util_pct"] = np.mean(cpu_utils) if cpu_utils else 0  # 전체 평균
            sample["cpu_freq_mhz"] = cpu_freqs          # 코어별 주파수 리스트

        # --- GPU 사용률 파싱 ---
        # "GR3D_FREQ 50%@921" → GPU 사용률 50%, 주파수 921MHz
        # GR3D = Jetson의 통합 GPU (Maxwell 128코어)
        gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%@(\d+)", line)
        if gpu_match:
            sample["gpu_util_pct"] = int(gpu_match.group(1))
            sample["gpu_freq_mhz"] = int(gpu_match.group(2))

        # --- 온도 파싱 ---
        # "CPU@42.5C GPU@40C PMIC@100C AO@44.5C thermal@41.5C PLL@41C"
        # 각 센서별 온도를 섭씨로 추출
        for temp_name in ["CPU", "GPU", "PMIC", "AO", "thermal", "PLL"]:
            temp_match = re.search(rf"{temp_name}@([\d.]+)C", line)
            if temp_match:
                sample[f"temp_{temp_name.lower()}_c"] = float(temp_match.group(1))

        # --- 전력 소비 파싱 ---
        # "POM_5V_IN 4567/4567" → 현재 전력 / 이동평균 전력 (mW)
        # POM_5V_IN: 보드 전체 입력 전력
        # POM_5V_GPU: GPU 전력
        # POM_5V_CPU: CPU 전력
        for power_name in ["POM_5V_IN", "POM_5V_GPU", "POM_5V_CPU"]:
            power_match = re.search(rf"{power_name}\s+(\d+)/(\d+)(?:mW)?", line)
            if power_match:
                key = power_name.lower()
                sample[f"{key}_current_mw"] = int(power_match.group(1))   # 순간 전력 (mW)
                sample[f"{key}_average_mw"] = int(power_match.group(2))   # 이동평균 전력 (mW)

        # 유효한 데이터가 하나도 파싱되지 않았으면 None 반환
        return sample if sample else None

    def summary(self, log_file=None):
        """tegrastats 로그에서 요약 통계를 계산한다.

        전체 샘플에 대해 각 수치형 필드의 평균, 최대, 최소를 구한다.
        벤치마크 실행 중의 시스템 자원 사용 현황을 한눈에 파악할 수 있다.

        Args:
            log_file: 파싱할 로그 파일 경로

        Returns:
            요약 딕셔너리. 형식:
            {
                "num_samples": 150,
                "ram_used_mb": {"mean": 2345.0, "max": 2500.0, "min": 2200.0},
                "gpu_util_pct": {"mean": 85.5, "max": 99.0, "min": 0.0},
                ...
            }
        """
        samples = self.parse(log_file)
        if not samples:
            return {"error": "No samples parsed"}

        result = {"num_samples": len(samples)}

        # 모든 샘플에서 수치형(int, float) 키만 추출
        # 리스트형 필드(cpu_util_pct, cpu_freq_mhz)는 자동으로 제외됨
        numeric_keys = set()
        for s in samples:
            for k, v in s.items():
                if isinstance(v, (int, float)):
                    numeric_keys.add(k)

        # 각 수치형 필드에 대해 평균/최대/최소 계산
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
    """편의 함수: 함수 실행 중 tegrastats를 자동으로 로깅한다.

    벤치마크 함수를 실행하면서 동시에 tegrastats를 백그라운드로 기록하고,
    함수 종료 후 자동으로 로깅을 중단하여 요약 통계를 반환한다.
    try/finally로 예외 발생 시에도 반드시 tegrastats를 중지한다.

    Args:
        func: 실행할 벤치마크 함수 (인자 없는 callable)
        log_file: tegrastats 로그 저장 경로
        interval_ms: 로깅 간격 (밀리초)

    Returns:
        (func_result, tegra_summary) 튜플
        - func_result: func()의 반환값
        - tegra_summary: summary() 딕셔너리 (평균/최대/최소 통계)
    """
    logger = TegraStatsLogger()
    logger.start(log_file, interval_ms)
    try:
        result = func()
    finally:
        # 예외 발생 여부와 관계없이 반드시 tegrastats 프로세스 정지
        logger.stop()
    summary = logger.summary(log_file)
    return result, summary


# =============================================================================
# CLI 진입점: 기존 로그 파싱 또는 실시간 녹화
# =============================================================================
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
        # 모드 1: 기존 로그 파일 파싱 → JSON 요약 출력
        logger = TegraStatsLogger()
        summary = logger.summary(args.parse)
        print(json.dumps(summary, indent=2))
    elif args.record > 0:
        # 모드 2: 지정 시간만큼 실시간 녹화 → 정지 → JSON 요약 출력
        logger = TegraStatsLogger()
        print(f"Recording tegrastats for {args.record}s to {args.output}...")
        logger.start(args.output, args.interval)
        time.sleep(args.record)
        logger.stop()
        summary = logger.summary(args.output)
        print(json.dumps(summary, indent=2))
    else:
        # 인자 없이 실행 시 사용법 안내
        print("Usage:")
        print(f"  {__file__} --parse <log_file>")
        print(f"  {__file__} --record <seconds> --output <file>")
