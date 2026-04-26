import torch
import time
import numpy as np
import os
import argparse
import json
import csv
from datetime import datetime

# 기존에 만든 하드웨어 모니터 클래스 임포트 (파일이 같은 경로에 있다고 가정)
from benchmark.tegra_parser import TegraMonitor 
from benchmark.result_saver import ResultSaver
from benchmark.power_mapper import PowerMapper
try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    OfflineEmissionsTracker = None

# 각 런타임별 로딩 및 추론 엔진 (예시 구조)
# 실제 환경에 맞게 각 run_xxx.py 파일에서 함수를 가져오거나 내부에 구현합니다.

class BenchmarkMaster:
    def __init__(self, model_name, model_dir="./models"):
        self.model_name = model_name
        self.model_dir = model_dir
        self.results = []
        self.saver = ResultSaver(path=f"./results/benchmark_{self.model_name}.csv")

    def calc_detailed_stats(self, latencies):
        """코드 B의 장점: 상세 백분위 통계 계산"""
        latencies = np.array(latencies)
        return {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "std_ms":  round(float(np.std(latencies)), 2),
            "min_ms":  round(float(np.min(latencies)), 2),
            "max_ms":  round(float(np.max(latencies)), 2),
            "p50_ms":  round(float(np.percentile(latencies, 50)), 2),
            "p95_ms":  round(float(np.percentile(latencies, 95)), 2),
            "p99_ms":  round(float(np.percentile(latencies, 99)), 2),
            "fps":     round(1000.0 / np.mean(latencies), 2) if np.mean(latencies) > 0 else 0
        }

    def run_runtime_benchmark(self, runtime, device_str, warmup=10, runs=100):
        print(f"\n>>> Running Benchmark: [{runtime}] on [{device_str}]")
        
        # 1. 하드웨어 모니터링 시작 (코드 A의 장점)
        monitor = TegraMonitor(interval_ms=100)
        monitor.start()
        
        tracker = None
        if OfflineEmissionsTracker:
            tracker = OfflineEmissionsTracker(country_iso_code="KOR", log_level="error")
            tracker.start()

        # 2. 모델 및 입력 준비 (런타임별 분기)
        # 여기서는 구조적 예시만 보여드립니다. 실제 로직은 각 엔진별 라이브러리 호출 필요.
        device = torch.device(device_str)
        
        # [Placeholder] 실제 모델 로드 및 추론 함수 정의 (lambda 혹은 def)
        # 예: infer_fn = lambda x: model(x)
        # dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        # --- (이 부분에 각 런타임별 로드/추론 로직 삽입) ---
        # 3. 워밍업 (Warm-up)
        print(f"  - Warming up {warmup} times...")
        # for _ in range(warmup): infer_fn(dummy_input)
        if device.type == "cuda": torch.cuda.synchronize()

        # 4. 본 측정 및 메모리 체크 (코드 B의 장점)
        latencies = []
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
        
        print(f"  - Measuring {runs} iterations...")
        for _ in range(runs):
            if device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter()
            
            # infer_fn(dummy_input) # 실제 실행
            time.sleep(0.01) # 테스트용 더미 딜레이
            
            if device.type == "cuda": torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        # 메모리 측정
        mem_stats = {"peak_usage_mb": 0}
        if device.type == "cuda":
            mem_stats["peak_usage_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)

        # 5. 하드웨어 데이터 수집 종료
        monitor.stop()
        hw_summary = monitor.summary()
        
        co2_kg = 0.0
        if tracker:
            co2_kg = tracker.stop()

        # 6. 결과 통합 및 전력 매핑
        stats = self.calc_detailed_stats(latencies)
        
        # [NEW] Power Mapper를 이용해 레이어 타임스탬프와 tegrastats 전력 매핑
        # 현재는 dummy layer_profiles를 예시로 넘기지만, 
        # 실제 각 프레임워크(ONNX, TFLite 등)의 profiler parser 결과를 여기에 주입해야 합니다.
        dummy_layer_profiles = [
            {"layer_name": "dummy_conv", "type": "Conv", "mean_ms": stats["mean_ms"], "timestamp": time.time()}
        ]
        
        power_mapper = PowerMapper()
        mapped_layer_profiles = power_mapper.map_power(dummy_layer_profiles, monitor.records)
        power_mapper.save_csv(mapped_layer_profiles, self.model_name, runtime)
        
        result = {
            "runtime": runtime,
            "device": device_str,
            "latency": stats,
            "memory": mem_stats,
            "hardware": hw_summary,
            "co2_kg": co2_kg,
            "layer_profiles": mapped_layer_profiles
        }
        self.results.append(result)
        
        # ResultSaver를 통한 실시간 CSV 저장
        self.saver.save({
            "model": self.model_name,
            "runtime": runtime,
            "precision": "fp32" if "fp32" in runtime else ("fp16" if "fp16" in runtime else "int8" if "int8" in runtime else "N/A"),
            "power_mode": "10w", # 기본값, 외부 주입 가능
            "mean_ms": stats["mean_ms"],
            "std_ms": stats["std_ms"],
            "min_ms": stats["min_ms"],
            "max_ms": stats["max_ms"],
            "median_ms": stats["p50_ms"],
            "power_total_mw_mean": hw_summary.get("power_total_mw_mean", 0),
            "power_gpu_mw_mean": hw_summary.get("power_gpu_mw_mean", 0),
            "power_cpu_mw_mean": hw_summary.get("power_cpu_mw_mean", 0),
            "ram_used_mb_mean": hw_summary.get("ram_used_mb_mean", 0),
            "gpu_util_pct_mean": hw_summary.get("gpu_util_pct_mean", 0),
            "gpu_temp_c_mean": hw_summary.get("gpu_temp_c_mean", 0),
            "co2_kg": co2_kg,
            "accuracy_top1": 0 # 나중에 검증 단계에서 업데이트
        })
        
        return result

    def save_all(self, output_dir="./results"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 저장 (상세 데이터)
        json_path = os.path.join(output_dir, f"report_{self.model_name}_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=4)
            
        # CSV 저장 (요약 데이터 - 엑셀 보고용)
        csv_path = os.path.join(output_dir, f"summary_{self.model_name}_{timestamp}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Runtime", "Device", "Mean(ms)", "P95(ms)", "FPS", "PeakMem(MB)", "AvgPower(mW)"])
            for r in self.results:
                writer.writerow([
                    r["runtime"], r["device"], 
                    r["latency"]["mean_ms"], r["latency"]["p95_ms"], 
                    r["latency"]["fps"], r["memory"]["peak_usage_mb"],
                    r["hardware"].get("power_total_mw_mean", 0)
                ])
        print(f"\n[DONE] Results saved to {output_dir}")

# --- 실행부 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenet_v3")
    args = parser.parse_args()

    master = BenchmarkMaster(args.model)
    
    # 비교하고 싶은 런타임들을 리스트업
    target_runtimes = [
        ("pytorch_cpu", "cpu"),
        ("pytorch_cuda", "cuda"),
        ("tensorrt_fp16", "cuda"),
        ("onnxrt_cuda", "cuda"),
        ("tflite_gpu", "cuda")
    ]

    for rt, dev in target_runtimes:
        master.run_runtime_benchmark(rt, dev)
        print(f"  - Waiting for cool-down (5s)...")
        time.sleep(5) # 장치 열 식히기

    master.save_all()