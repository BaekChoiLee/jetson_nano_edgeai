import torch
import torchvision.models as tv_models
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
from benchmark.torch_layer_analyer import LayerProfiler
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

    def load_pytorch_model(self):
        builders = {
            "mobilenetv3s": tv_models.mobilenet_v3_small,
            "efficientnetb0": tv_models.efficientnet_b0,
            "shufflenetv2": tv_models.shufflenet_v2_x1_0,
            "resnet50": tv_models.resnet50,
        }
        if self.model_name == "yolov8n":
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError("ultralytics is required for YOLOv8 layer profiling. Install it or place a PyTorch YOLO loader here.") from exc

            candidates = [
                os.path.join(self.model_dir, "yolov8n.pt"),
                "yolov8n.pt",
            ]
            weights_path = next((path for path in candidates if os.path.exists(path)), candidates[-1])
            return YOLO(weights_path).model

        if self.model_name not in builders:
            return None

        builder = builders[self.model_name]
        try:
            return builder(weights=None)
        except TypeError:
            return builder(pretrained=False)

    def input_shape(self):
        if self.model_name in {"yolov8n", "ssd_mv2"}:
            return (1, 3, 640, 640)
        return (1, 3, 224, 224)

    def run_runtime_benchmark(self, runtime, device_str, warmup=10, runs=100):
        print(f"\n>>> Running Benchmark: [{runtime}] on [{device_str}]")
        
        # 1. 하드웨어 모니터링 시작 (코드 A의 장점)
        monitor = TegraMonitor(interval_ms=100)
        monitor.start()
        
        tracker = None
        if OfflineEmissionsTracker:
            tracker = OfflineEmissionsTracker(country_iso_code="KOR", log_level="error")
            tracker.start()

        device = torch.device(device_str)
        model = None
        dummy_input = None
        infer_fn = None
        if runtime.startswith("pytorch"):
            model = self.load_pytorch_model()
            if model is None:
                raise ValueError(f"PyTorch model loader is not implemented for {self.model_name}")
            model = model.to(device).eval()
            dummy_input = torch.randn(*self.input_shape()).to(device)
            infer_fn = lambda: model(dummy_input)

        # 3. 워밍업 (Warm-up)
        print(f"  - Warming up {warmup} times...")
        with torch.no_grad():
            for _ in range(warmup):
                if infer_fn:
                    infer_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()

        # 4. 본 측정 및 메모리 체크 (코드 B의 장점)
        latencies = []
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
        
        print(f"  - Measuring {runs} iterations...")
        for _ in range(runs):
            if device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter()
            
            if infer_fn:
                with torch.no_grad():
                    infer_fn()
            else:
                time.sleep(0.01) # 아직 실제 런타임 profiler가 붙지 않은 엔진용 placeholder
            
            if device.type == "cuda": torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        # 메모리 측정
        mem_stats = {"peak_usage_mb": 0}
        if device.type == "cuda":
            mem_stats["peak_usage_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)

        layer_profiles = []
        if model is not None and dummy_input is not None:
            print(f"  - Profiling PyTorch layers over {runs} iterations...")
            layer_profiler = LayerProfiler(model, device=device)
            layer_profiles = layer_profiler.profile(dummy_input, warmup=warmup, runs=runs)
        else:
            print("  - [Warning] Runtime-specific layer profiler is not implemented yet; skipping layer CSV.")

        # 5. 하드웨어 데이터 수집 종료
        monitor.stop()
        hw_summary = monitor.summary()
        
        co2_kg = 0.0
        if tracker:
            try:
                co2_kg = tracker.stop()
            except Exception as e:
                print(f"  [Warning] CodeCarbon failed to stop cleanly: {e}")
                co2_kg = 0.0

        # 6. 결과 통합 및 전력 매핑
        stats = self.calc_detailed_stats(latencies)
        power_mapper = PowerMapper()
        mapped_layer_profiles = power_mapper.map_power(layer_profiles, monitor.records) if layer_profiles else []
        if mapped_layer_profiles:
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
