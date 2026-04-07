#!/usr/bin/env python3
"""PyTorch CPU/CUDA inference benchmark."""

import time
import numpy as np
import torch
import torchvision.models as models
import torchvision.models.detection as det_models


MODEL_FACTORIES = {
    "mobilenetv3_small": models.mobilenet_v3_small,
    "resnet50": models.resnet50,
    "shufflenet_v2": models.shufflenet_v2_x1_0,
    "ssd_mobilenet_v2": det_models.ssdlite320_mobilenet_v3_large,
}

# Detection models require list input and produce different output
DETECTION_MODELS = {"ssd_mobilenet_v2"}

# Input sizes per model (default 224 for classification)
INPUT_SIZES = {
    "ssd_mobilenet_v2": (1, 3, 320, 320),
}


def benchmark_pytorch(model_name, device="cpu", num_warmup=10, num_runs=100):
    """Benchmark PyTorch model on CPU or CUDA.

    Returns dict with latency stats, memory usage, and model size.
    """
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_FACTORIES.keys())}")

    is_detection = model_name in DETECTION_MODELS

    # Load model
    model = MODEL_FACTORIES[model_name](pretrained=True).eval()
    device = torch.device(device)
    model = model.to(device)

    # Model size (MB)
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)

    # Input tensor
    input_size = INPUT_SIZES.get(model_name, (1, 3, 224, 224))
    input_tensor = torch.randn(*input_size, device=device)

    use_cuda = device.type == "cuda"

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # Helper: run inference for classification or detection
    def run_inference():
        if is_detection:
            return model([input_tensor[0]])
        else:
            return model(input_tensor)

    # Warm-up
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = run_inference()
            if use_cuda:
                torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.inference_mode():
        for _ in range(num_runs):
            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = run_inference()
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                start = time.perf_counter()
                _ = run_inference()
                elapsed_ms = (time.perf_counter() - start) * 1000

            latencies.append(elapsed_ms)

    latencies = np.array(latencies)

    # Memory usage
    memory_mb = 0.0
    if use_cuda:
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "runtime": f"pytorch_{'cuda' if use_cuda else 'cpu'}",
        "model": model_name,
        "latency": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "memory_mb": round(memory_mb, 2),
        "model_size_mb": round(model_size_mb, 2),
        "num_runs": num_runs,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_FACTORIES.keys()))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    args = parser.parse_args()

    result = benchmark_pytorch(args.model, args.device, args.num_warmup, args.num_runs)
    print(json.dumps(result, indent=2))
