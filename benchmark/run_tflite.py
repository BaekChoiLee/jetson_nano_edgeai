#!/usr/bin/env python3
# =============================================================================
# run_tflite.py - TFLite CPU/GPU 추론 벤치마크 (Fallback 방어 및 로깅 강화)
# =============================================================================

import time
import os
import numpy as np

def _get_interpreter(model_path, use_gpu, num_threads=4):
    delegates = []
    if use_gpu:
        try:
            import tflite_runtime.interpreter as tflite
            delegates.append(tflite.load_delegate("libdelegate_gpu.so"))
            print("[INFO] TFLite GPU delegate loaded successfully.")
            InterpreterClass = tflite.Interpreter
        except Exception as e:
            print(f"[WARN] GPU delegate load failed -> CPU: {e}")
            try:
                import tensorflow as tf
                try:
                    delegates.append(tf.lite.experimental.load_delegate("libdelegate_gpu.so"))
                    print("[INFO] tf.lite GPU delegate loaded.")
                except Exception as e2:
                    print(f"[WARN] tf.lite GPU delegate also failed -> CPU fallback: {e2}")
                InterpreterClass = tf.lite.Interpreter
            except ImportError:
                import tflite_runtime.interpreter as tflite
                InterpreterClass = tflite.Interpreter
                print("[WARN] Running completely on CPU mode due to delegate failure.")
    else:
        try:
            import tflite_runtime.interpreter as tflite
            InterpreterClass = tflite.Interpreter
        except ImportError:
            import tensorflow as tf
            InterpreterClass = tf.lite.Interpreter
            
    interpreter = InterpreterClass(
        model_path=model_path, 
        experimental_delegates=delegates if delegates else None,
        num_threads=num_threads
    )
    return interpreter

def benchmark_tflite(model_path, num_warmup=10, num_runs=100, use_gpu=False):
    interpreter = _get_interpreter(model_path, use_gpu)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]
    input_data = np.random.randn(*input_shape).astype(input_dtype)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    import psutil
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024 * 1024)

    for _ in range(num_warmup):
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

    latencies = []
    mem_peak = mem_before
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        cur = proc.memory_info().rss / (1024 * 1024)
        if cur > mem_peak:
            mem_peak = cur

    memory_mb = round(max(0.0, mem_peak - mem_before), 2)
    latencies = np.array(latencies)

    return {
        "runtime": "tflite_gpu" if use_gpu else "tflite_cpu",
        "model": os.path.basename(model_path),
        "latency": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "memory_mb": memory_mb,
        "model_size_mb": round(model_size_mb, 2),
        "num_runs": num_runs,
        "input_shape": input_shape.tolist(),
        "input_dtype": str(input_dtype),
    }

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    args = parser.parse_args()

    result = benchmark_tflite(args.model, args.num_warmup, args.num_runs)
    print(json.dumps(result, indent=2))
