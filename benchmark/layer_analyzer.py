#!/usr/bin/env python3
"""Layer-by-layer analysis using PyTorch forward hooks."""

import argparse
import os
import time
import csv
import numpy as np
import torch
import torchvision.models as models


MODEL_FACTORIES = {
    "mobilenetv3_small": models.mobilenet_v3_small,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
    "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
}

# Detection models: loaded from .torchscript files
TORCHSCRIPT_INPUT_SIZES = {
    "yolov8n": (1, 3, 640, 640),
    "ssd_mobilenet_v2": (1, 3, 320, 320),
}

ALL_MODELS = list(MODEL_FACTORIES.keys()) + list(TORCHSCRIPT_INPUT_SIZES.keys())


def _analyze_hooks(model, model_name, input_tensor, device, use_cuda, num_runs):
    """Profile classification models using forward hooks."""
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

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Warm-up
    with torch.inference_mode():
        for _ in range(5):
            _ = model(input_tensor)
            if use_cuda:
                torch.cuda.synchronize()

    all_layer_results = {name: [] for name in layer_times}

    for run in range(num_runs):
        timestamps = []

        def make_timing_hook(name, timestamps_list):
            def hook_fn(module, input, output):
                if use_cuda:
                    torch.cuda.synchronize()
                timestamps_list.append((name, time.perf_counter()))
            return hook_fn

        for h in hooks:
            h.remove()
        hooks.clear()

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(make_timing_hook(name, timestamps)))

        with torch.inference_mode():
            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            if use_cuda:
                torch.cuda.synchronize()

        prev_time = start
        for layer_name, t in timestamps:
            elapsed_ms = (t - prev_time) * 1000
            if layer_name not in all_layer_results:
                all_layer_results[layer_name] = []
            all_layer_results[layer_name].append(elapsed_ms)
            prev_time = t

    for h in hooks:
        h.remove()

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
    return results


def _analyze_profiler(model, model_name, input_tensor, device, use_cuda, num_runs):
    """Profile TorchScript models using torch.autograd.profiler."""
    # Warm-up
    with torch.inference_mode():
        for _ in range(5):
            _ = model(input_tensor)
            if use_cuda:
                torch.cuda.synchronize()

    # Aggregate operator times across runs
    op_totals = {}
    for run in range(num_runs):
        with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
            with torch.inference_mode():
                _ = model(input_tensor)

        for evt in prof.function_events:
            name = evt.name
            dur_ms = evt.cuda_time_total / 1000.0 if use_cuda else evt.cpu_time_total / 1000.0
            if name not in op_totals:
                op_totals[name] = {"times": [], "count": 0}
            op_totals[name]["times"].append(dur_ms)
            op_totals[name]["count"] = max(op_totals[name]["count"], evt.count)

    results = []
    for name, info in op_totals.items():
        times = info["times"]
        if not times:
            continue
        times = np.array(times)
        results.append({
            "layer_name": name,
            "layer_type": "operator",
            "params": 0,
            "input_shape": [],
            "output_shape": [],
            "mean_ms": round(float(np.mean(times)), 4),
            "std_ms": round(float(np.std(times)), 4),
            "min_ms": round(float(np.min(times)), 4),
            "max_ms": round(float(np.max(times)), 4),
        })
    return results


def analyze_layers(model_name, device="cuda", num_runs=20, output_dir="./results",
                   model_dir="./models"):
    """Profile per-layer execution time.

    Classification models use forward hooks; TorchScript (detection) models
    use torch.autograd.profiler since ScriptModules don't support hooks.

    Returns list of dicts with layer info and timing.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    if model_name in MODEL_FACTORIES:
        model = MODEL_FACTORIES[model_name](pretrained=True).eval().to(device)
        input_tensor = torch.randn(1, 3, 224, 224, device=device)
        results = _analyze_hooks(model, model_name, input_tensor, device, use_cuda, num_runs)
    elif model_name in TORCHSCRIPT_INPUT_SIZES:
        ts_path = os.path.join(model_dir, f"{model_name}.torchscript")
        if not os.path.exists(ts_path):
            raise FileNotFoundError(f"TorchScript file not found: {ts_path}")
        model = torch.jit.load(ts_path, map_location=device).eval()
        input_size = TORCHSCRIPT_INPUT_SIZES[model_name]
        input_tensor = torch.randn(*input_size, device=device)
        results = _analyze_profiler(model, model_name, input_tensor, device, use_cuda, num_runs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: {ALL_MODELS}")

    # Detection models (profiler): sort by mean time — no sequential order exists.
    # Classification models (hooks): keep named_modules() sequential order.
    if model_name in TORCHSCRIPT_INPUT_SIZES:
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
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--model-dir", default="./models")
    args = parser.parse_args()

    analyze_layers(args.model, args.device, args.num_runs, args.output_dir, args.model_dir)
