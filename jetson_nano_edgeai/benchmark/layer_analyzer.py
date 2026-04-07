#!/usr/bin/env python3
"""Layer-by-layer analysis using PyTorch forward hooks."""

import argparse
import os
import time
import csv
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

DETECTION_MODELS = {"ssd_mobilenet_v2"}

INPUT_SIZES = {
    "ssd_mobilenet_v2": (1, 3, 320, 320),
}


def analyze_layers(model_name, device="cuda", num_runs=20, output_dir="./results"):
    """Profile per-layer execution time using forward hooks.

    Returns list of dicts with layer info and timing.
    """
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {model_name}")

    is_detection = model_name in DETECTION_MODELS

    model = MODEL_FACTORIES[model_name](pretrained=True).eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    use_cuda = device.type == "cuda"
    input_size = INPUT_SIZES.get(model_name, (1, 3, 224, 224))
    input_tensor = torch.randn(*input_size, device=device)

    # Collect layer info
    layer_times = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if name not in layer_times:
                layer_times[name] = {
                    "type": module.__class__.__name__,
                    "params": sum(p.numel() for p in module.parameters()),
                    "times": [],
                }
                # Input/output shapes
                if isinstance(input, tuple) and len(input) > 0:
                    inp = input[0]
                    if hasattr(inp, "shape"):
                        layer_times[name]["input_shape"] = list(inp.shape)
                if hasattr(output, "shape"):
                    layer_times[name]["output_shape"] = list(output.shape)

            if use_cuda:
                torch.cuda.synchronize()
            layer_times[name]["_end"] = time.perf_counter()
        return hook_fn

    # Register hooks on all leaf modules
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf module
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Helper: run inference for classification or detection
    def run_inference():
        if is_detection:
            return model([input_tensor[0]])
        else:
            return model(input_tensor)

    # Warm-up
    with torch.inference_mode():
        for _ in range(5):
            _ = run_inference()
            if use_cuda:
                torch.cuda.synchronize()

    # Profile: measure each layer by timing between consecutive hooks
    all_layer_results = {name: [] for name in layer_times}

    for run in range(num_runs):
        # Reset timestamps
        timestamps = []

        def make_timing_hook(name, timestamps_list):
            def hook_fn(module, input, output):
                if use_cuda:
                    torch.cuda.synchronize()
                timestamps_list.append((name, time.perf_counter()))
            return hook_fn

        # Remove old hooks and register timing hooks
        for h in hooks:
            h.remove()
        hooks.clear()
        timestamps = []

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(make_timing_hook(name, timestamps)))

        with torch.inference_mode():
            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = run_inference()
            if use_cuda:
                torch.cuda.synchronize()

        # Calculate per-layer times
        prev_time = start
        for layer_name, t in timestamps:
            elapsed_ms = (t - prev_time) * 1000
            if layer_name not in all_layer_results:
                all_layer_results[layer_name] = []
            all_layer_results[layer_name].append(elapsed_ms)
            prev_time = t

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compile results
    results = []
    for name, info in layer_times.items():
        times = all_layer_results.get(name, [])
        if not times:
            continue
        times = np.array(times)
        results.append({
            "layer_name": name,
            "layer_type": info["type"],
            "params": info["params"],
            "input_shape": info.get("input_shape", []),
            "output_shape": info.get("output_shape", []),
            "mean_ms": round(float(np.mean(times)), 4),
            "std_ms": round(float(np.std(times)), 4),
            "min_ms": round(float(np.min(times)), 4),
            "max_ms": round(float(np.max(times)), 4),
        })

    # Sort by mean time (descending)
    results.sort(key=lambda x: x["mean_ms"], reverse=True)

    # Print top-10 bottleneck layers
    print(f"\n{'='*70}")
    print(f"  Layer Analysis: {model_name} on {device}")
    print(f"  {len(results)} layers, {num_runs} runs each")
    print(f"{'='*70}")
    print(f"  {'Rank':<5} {'Layer':<35} {'Type':<15} {'Mean(ms)':<10} {'Params'}")
    print(f"  {'-'*5} {'-'*35} {'-'*15} {'-'*10} {'-'*10}")
    for i, r in enumerate(results[:10]):
        short_name = r["layer_name"][-35:]
        print(f"  {i+1:<5} {short_name:<35} {r['layer_type']:<15} {r['mean_ms']:<10.4f} {r['params']:,}")

    total_ms = sum(r["mean_ms"] for r in results)
    print(f"\n  Total layer time: {total_ms:.2f} ms")

    # Save CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{model_name}_layers.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer_name", "layer_type", "params", "input_shape", "output_shape",
            "mean_ms", "std_ms", "min_ms", "max_ms",
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Full results saved to: {csv_path}")

    # Generate bar chart (if matplotlib available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top20 = results[:20]
        names = [r["layer_name"].split(".")[-1][:20] for r in top20]
        times = [r["mean_ms"] for r in top20]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(range(len(names)), times, color="steelblue")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Mean Latency (ms)")
        ax.set_title(f"Top-20 Bottleneck Layers: {model_name}")
        ax.invert_yaxis()
        plt.tight_layout()

        chart_path = os.path.join(output_dir, f"{model_name}_layers.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"  Chart saved to: {chart_path}")
    except ImportError:
        print("  [INFO] matplotlib not available, skipping chart.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer-by-layer profiling")
    parser.add_argument("--model", required=True, choices=list(MODEL_FACTORIES.keys()))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()

    analyze_layers(args.model, args.device, args.num_runs, args.output_dir)
