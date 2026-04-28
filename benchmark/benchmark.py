import torch
import torchvision.models as tv_models
import time
import numpy as np
import os
import argparse
import json
import csv
import shutil
import subprocess
from datetime import datetime

# 기존에 만든 하드웨어 모니터 클래스 임포트 (파일이 같은 경로에 있다고 가정)
from benchmark.tegra_parser import TegraMonitor 
from benchmark.result_saver import ResultSaver
from benchmark.power_mapper import PowerMapper
from benchmark.torch_layer_analyer import LayerProfiler
from benchmark.onnxrt_profiler import ONNXRuntimeProfilerParser
from benchmark.trt_profiler import TRTExecProfilerParser
from benchmark.tflite_profiler import TFLiteProfilerParser
from benchmark.ncnn_profiler import NCNNProfilerParser
try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    OfflineEmissionsTracker = None

COMMON_TOOL_PATHS = {
    "trtexec": [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "/usr/bin/trtexec",
    ],
    "benchmark_model": [
        "/usr/local/bin/benchmark_model",
        "/usr/bin/benchmark_model",
    ],
    "benchncnn": [
        "/usr/local/bin/benchncnn",
        "/usr/bin/benchncnn",
    ],
}


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def common_tool_paths(default_name):
    root = repo_root()
    repo_paths = {
        "benchmark_model": [
            os.path.join(root, "benchmark_model"),
            os.path.join(root, "build_tflite", "tools", "benchmark", "benchmark_model"),
            os.path.join(root, "tensorflow", "bazel-bin", "tensorflow", "lite", "tools", "benchmark", "benchmark_model"),
        ],
        "benchncnn": [
            os.path.join(root, "build_ncnn", "benchmark", "benchncnn"),
            os.path.join(root, "build_ncnn", "benchncnn"),
            os.path.join(root, "benchncnn"),
        ],
    }
    return repo_paths.get(default_name, []) + COMMON_TOOL_PATHS.get(default_name, [])

# 각 런타임별 로딩 및 추론 엔진 (예시 구조)
# 실제 환경에 맞게 각 run_xxx.py 파일에서 함수를 가져오거나 내부에 구현합니다.

class BenchmarkMaster:
    def __init__(self, model_name, model_dir="./models", power_mode="10w"):
        self.model_name = model_name
        self.model_dir = model_dir
        self.power_mode = power_mode
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

    def onnx_model_path(self):
        return os.path.join(self.model_dir, f"{self.model_name}.onnx")

    def tflite_model_path(self):
        return os.path.join(self.model_dir, f"{self.model_name}.tflite")

    def ncnn_model_paths(self):
        return (
            os.path.join(self.model_dir, f"{self.model_name}.param"),
            os.path.join(self.model_dir, f"{self.model_name}.bin"),
        )

    def trt_engine_candidates(self, runtime):
        precision = runtime.replace("tensorrt_", "")
        return [
            os.path.join(self.model_dir, f"{self.model_name}_{precision}.engine"),
            os.path.join(self.model_dir, f"{self.model_name}_{precision}.trt"),
            os.path.join(self.model_dir, f"{self.model_name}.engine"),
            os.path.join(self.model_dir, f"{self.model_name}.trt"),
        ]

    def _tool_path(self, env_name, default_name):
        override = os.environ.get(env_name)
        if override:
            resolved = shutil.which(override) or override
            if os.path.exists(resolved) or shutil.which(resolved):
                return resolved
            raise FileNotFoundError(f"{default_name} tool not found at {override} from ${env_name}")

        resolved = shutil.which(default_name)
        if resolved:
            return resolved
        for candidate in common_tool_paths(default_name):
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"{default_name} tool not found in PATH. Set ${env_name} to its executable path.")

    def _require_file(self, path, description):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{description} not found: {path}")

    def _run_external_runtime_benchmark(self, runtime, device_str, warmup, runs):
        monitor = TegraMonitor(interval_ms=100)
        monitor.start()

        tracker = None
        if OfflineEmissionsTracker:
            tracker = OfflineEmissionsTracker(country_iso_code="KOR", log_level="error")
            tracker.start()

        co2_kg = 0.0
        try:
            if runtime.startswith("tensorrt"):
                latencies, layer_profiles = self._run_trtexec_profile(runtime, warmup, runs)
            elif runtime == "tflite_cpu":
                latencies, layer_profiles = self._run_tflite_profile(runs)
            elif runtime == "ncnn_vulkan":
                latencies, layer_profiles = self._run_ncnn_profile(runs)
            else:
                raise NotImplementedError(f"{runtime} external profiler is not implemented.")
        finally:
            monitor.stop()
            if tracker:
                try:
                    co2_kg = tracker.stop()
                except Exception as e:
                    print(f"  [Warning] CodeCarbon failed to stop cleanly: {e}")

        return self._record_result(runtime, device_str, latencies, layer_profiles, monitor, co2_kg)

    def _run_trtexec_profile(self, runtime, warmup, runs):
        onnx_path = self.onnx_model_path()
        trtexec = self._tool_path("TRTEXEC_PATH", "trtexec")
        parser = TRTExecProfilerParser()
        engine_path = next((path for path in self.trt_engine_candidates(runtime) if os.path.exists(path)), None)

        if engine_path:
            cmd = [trtexec, f"--loadEngine={engine_path}"]
        else:
            self._require_file(onnx_path, "ONNX model for TensorRT")
            cmd = [trtexec, f"--onnx={onnx_path}"]
            if runtime == "tensorrt_fp16":
                cmd.append("--fp16")
            elif runtime == "tensorrt_int8":
                cmd.append("--int8")

        cmd.extend([
            f"--iterations={runs}",
            f"--warmUp={max(warmup, 1) * 1000}",
            "--dumpProfile",
            "--separateProfileRun",
        ])

        print(f"  - Running TensorRT profiler: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"trtexec failed with exit code {result.returncode}\n{result.stdout[-2000:]}")

        mean_ms = parser.parse_mean_latency_ms(result.stdout)
        if mean_ms is None:
            raise RuntimeError(f"Failed to parse TensorRT mean latency from trtexec output.\n{result.stdout[-2000:]}")

        layer_profiles = parser.parse(result.stdout)
        if not layer_profiles:
            print("  - [Warning] TensorRT ran, but no layer profile rows were parsed from trtexec output.")
        return [mean_ms], layer_profiles

    def _run_tflite_profile(self, runs):
        model_path = self.tflite_model_path()
        self._require_file(model_path, "TFLite model")
        tool_path = self._tool_path("TFLITE_BENCHMARK_MODEL_PATH", "benchmark_model")

        print(f"  - Running TFLite profiler: {tool_path}")
        profiler = TFLiteProfilerParser(model_path, benchmark_tool_path=tool_path)
        layer_profiles = profiler.run_and_parse(runs=runs)
        if profiler.last_returncode not in (0, None):
            raise RuntimeError(f"benchmark_model failed with exit code {profiler.last_returncode}\n{profiler.last_output[-2000:]}")

        mean_ms = profiler.parse_mean_latency_ms()
        if mean_ms is None:
            if layer_profiles:
                mean_ms = sum(float(row.get("mean_ms", 0)) for row in layer_profiles)
            else:
                raise RuntimeError(f"Failed to parse TFLite mean latency.\n{profiler.last_output[-2000:]}")
        return [mean_ms], layer_profiles

    def _run_ncnn_profile(self, runs):
        param_path, bin_path = self.ncnn_model_paths()
        self._require_file(param_path, "ncnn param model")
        self._require_file(bin_path, "ncnn bin model")
        tool_path = self._tool_path("BENCHNCNN_PATH", "benchncnn")

        print(f"  - Running ncnn profiler: {tool_path}")
        profiler = NCNNProfilerParser(param_path, bin_path, benchncnn_path=tool_path)
        layer_profiles = profiler.run_and_parse(runs=runs)
        if profiler.last_returncode not in (0, None):
            raise RuntimeError(f"benchncnn failed with exit code {profiler.last_returncode}\n{profiler.last_output[-2000:]}")

        mean_ms = profiler.parse_mean_latency_ms()
        if mean_ms is None:
            if layer_profiles:
                mean_ms = sum(float(row.get("mean_ms", 0)) for row in layer_profiles)
            else:
                raise RuntimeError(f"Failed to parse ncnn mean latency.\n{profiler.last_output[-2000:]}")
        return [mean_ms], layer_profiles

    def run_runtime_benchmark(self, runtime, device_str, warmup=10, runs=100):
        print(f"\n>>> Running Benchmark: [{runtime}] on [{device_str}]")

        supported_runtimes = {
            "pytorch_cpu", "pytorch_cuda", "onnxrt_cpu", "onnxrt_cuda",
            "tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8",
            "tflite_cpu", "ncnn_vulkan",
        }
        if runtime not in supported_runtimes:
            raise NotImplementedError(
                f"{runtime} is not wired to a real inference backend yet. "
                "Skipping to avoid recording placeholder latency."
            )

        if runtime.startswith("tensorrt") or runtime in {"tflite_cpu", "ncnn_vulkan"}:
            return self._run_external_runtime_benchmark(runtime, device_str, warmup, runs)
        
        torch_device = None
        model = None
        dummy_input = None
        infer_fn = None
        onnx_profile_path = None

        if runtime.startswith("pytorch"):
            torch_device = torch.device(device_str)
            model = self.load_pytorch_model()
            if model is None:
                raise ValueError(f"PyTorch model loader is not implemented for {self.model_name}")
            model = model.to(torch_device).eval()
            dummy_input = torch.randn(*self.input_shape()).to(torch_device)
            infer_fn = lambda: model(dummy_input)
        elif runtime in {"onnxrt_cpu", "onnxrt_cuda"}:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(f"onnxruntime is required for {runtime}. Install onnxruntime first.") from exc

            model_path = self.onnx_model_path()
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found: {model_path}")

            session_options = ort.SessionOptions()
            session_options.enable_profiling = True
            provider = "CUDAExecutionProvider" if runtime == "onnxrt_cuda" else "CPUExecutionProvider"
            if provider not in ort.get_available_providers():
                raise RuntimeError(
                    f"{provider} is not available. Available ONNX Runtime providers: {ort.get_available_providers()}"
                )
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=[provider],
            )
            input_name = session.get_inputs()[0].name
            dummy_input = np.random.randn(*self.input_shape()).astype(np.float32)
            infer_fn = lambda: session.run(None, {input_name: dummy_input})

        # 1. 하드웨어 모니터링 시작 (backend 준비 시간은 측정에서 제외)
        monitor = TegraMonitor(interval_ms=100)
        monitor.start()

        tracker = None
        if OfflineEmissionsTracker:
            tracker = OfflineEmissionsTracker(country_iso_code="KOR", log_level="error")
            tracker.start()

        # 3. 워밍업 (Warm-up)
        print(f"  - Warming up {warmup} times...")
        with torch.no_grad():
            for _ in range(warmup):
                if infer_fn:
                    infer_fn()
        if torch_device is not None and torch_device.type == "cuda":
            torch.cuda.synchronize()

        # 4. 본 측정 및 메모리 체크 (코드 B의 장점)
        latencies = []
        if torch_device is not None and torch_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        print(f"  - Measuring {runs} iterations...")
        for _ in range(runs):
            if torch_device is not None and torch_device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            if infer_fn:
                with torch.no_grad():
                    infer_fn()
            
            if torch_device is not None and torch_device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        # 메모리 측정
        mem_stats = {"peak_usage_mb": 0}
        if torch_device is not None and torch_device.type == "cuda":
            mem_stats["peak_usage_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)

        layer_profiles = []
        if model is not None and dummy_input is not None:
            print(f"  - Profiling PyTorch layers over {runs} iterations...")
            layer_profiler = LayerProfiler(model, device=torch_device)
            layer_profiles = layer_profiler.profile(dummy_input, warmup=warmup, runs=runs)
        elif runtime in {"onnxrt_cpu", "onnxrt_cuda"}:
            onnx_profile_path = session.end_profiling()
            print(f"  - Parsing ONNX Runtime layer profile: {onnx_profile_path}")
            layer_profiles = ONNXRuntimeProfilerParser(onnx_profile_path).parse(num_runs=runs)
        else:
            print("  - [Warning] Runtime-specific layer profiler is not available; skipping layer CSV.")

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

        return self._record_result(runtime, device_str, latencies, layer_profiles, monitor, co2_kg, mem_stats)

    def _record_result(self, runtime, device_str, latencies, layer_profiles, monitor, co2_kg, mem_stats=None):
        mem_stats = mem_stats or {"peak_usage_mb": 0}
        hw_summary = monitor.summary()
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
            "power_mode": self.power_mode,
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
