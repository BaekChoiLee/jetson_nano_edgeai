#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jetson_edgeai_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/jetson_edgeai_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNTIME_ORDER = [
    "pytorch_cuda",
    "tensorrt_fp32",
    "tensorrt_fp16",
    "tensorrt_int8",
    "onnxrt_cpu",
    "onnxrt_cuda",
    "tflite_cpu",
    "ncnn_vulkan",
    "pytorch_cpu",
]

RUNTIME_LABELS = {
    "pytorch_cuda": "PyTorch CUDA",
    "pytorch_cpu": "PyTorch CPU",
    "tensorrt_fp32": "TensorRT FP32",
    "tensorrt_fp16": "TensorRT FP16",
    "tensorrt_int8": "TensorRT INT8",
    "onnxrt_cpu": "ONNX RT CPU",
    "onnxrt_cuda": "ONNX RT CUDA",
    "tflite_cpu": "TFLite CPU",
    "ncnn_vulkan": "ncnn Vulkan",
}

MODEL_LABELS = {
    "mobilenetv3s": "MobileNetV3-Small",
    "efficientnetb0": "EfficientNet-B0",
    "shufflenetv2": "ShuffleNetV2",
    "resnet50": "ResNet-50",
    "yolov8n": "YOLOv8n",
    "ssd_mv2": "SSD-MobileNetV2",
}

LAYER_FILE_RE = re.compile(
    r"^layer_power_(?P<model>.+)_(?P<runtime>"
    r"pytorch_cpu|pytorch_cuda|onnxrt_cpu|onnxrt_cuda|tflite_cpu|ncnn_vulkan|"
    r"tensorrt_fp32|tensorrt_fp16|tensorrt_int8)\.csv$"
)


def model_label(model):
    return MODEL_LABELS.get(str(model), str(model))


def runtime_label(runtime):
    return RUNTIME_LABELS.get(str(runtime), str(runtime))


def ordered_runtime_frame(df):
    present = [rt for rt in RUNTIME_ORDER if rt in set(df["runtime"].astype(str))]
    extra = sorted(set(df["runtime"].astype(str)) - set(present))
    order = present + extra
    return df.assign(runtime_order=df["runtime"].astype(str).map({rt: i for i, rt in enumerate(order)})).sort_values("runtime_order")


def load_benchmarks(results_dir):
    frames = []
    for path in sorted(results_dir.glob("benchmark_*.csv")):
        if path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "model" not in df.columns:
            df["model"] = path.stem.replace("benchmark_", "")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "median_ms",
        "power_total_mw_mean",
        "power_gpu_mw_mean",
        "power_cpu_mw_mean",
        "ram_used_mb_mean",
        "gpu_util_pct_mean",
        "gpu_temp_c_mean",
        "co2_kg",
        "accuracy_top1",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["runtime"] = df["runtime"].astype(str)
    return df.dropna(subset=["model", "runtime"]).reset_index(drop=True)


def load_layer_profiles(results_dir):
    frames = []
    for path in sorted(results_dir.glob("layer_power_*.csv")):
        match = LAYER_FILE_RE.match(path.name)
        if not match or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if df.empty or "layer_name" not in df.columns:
            continue
        df["model"] = match.group("model")
        df["runtime"] = match.group("runtime")
        if "layer_index" not in df.columns:
            df["layer_index"] = range(1, len(df) + 1)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for col in ["layer_index", "mean_ms", "power_gpu_mw", "power_total_mw", "param_count", "trainable_param_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["layer_name"] = df["layer_name"].astype(str)
    df["type"] = df.get("type", "").astype(str)
    return clean_layer_rows(df)


def clean_layer_rows(df):
    out = df.copy()
    bad_names = (
        out["layer_name"].str.contains("Average on|Sleep time|Throughput|Host Latency|GPU Compute", case=False, na=False)
        | out["layer_name"].isin(["loop_count", "num_threads", "powersave", "gpu_device", "cooling_down"])
    )
    bad_types = out["type"].isin(["=", ""])
    out = out[~(bad_names | bad_types)].copy()
    out = out.dropna(subset=["mean_ms"])
    out = out[out["mean_ms"] > 0]
    if "layer_index" not in out.columns:
        out["layer_index"] = range(1, len(out) + 1)
    return out


def norm_type(layer_type, layer_name=""):
    text = f"{layer_type} {layer_name}".lower()
    if "param min" in text or ".param min" in text:
        return "inference-summary"
    if any(tok in text for tok in ["batchnorm", "layernorm", "instancenorm", "groupnorm", "norm"]):
        return "norm"
    if any(tok in text for tok in ["relu", "gelu", "sigmoid", "hardswish", "hardsigmoid", "swish", "silu", "activation", "clip"]):
        return "act"
    if any(tok in text for tok in ["conv", "convolution", "fusedconv"]):
        return "conv"
    if any(tok in text for tok in ["linear", "gemm", "matmul", "innerproduct", "fully_connected", "fc"]):
        return "fc"
    if any(tok in text for tok in ["pool", "mean", "globalaveragepool"]):
        return "pool"
    if any(tok in text for tok in ["add", "mul", "eltwise", "binary"]):
        return "eltwise"
    if any(tok in text for tok in ["concat", "split"]):
        return "concat/split"
    if any(tok in text for tok in ["reshape", "transpose", "permute", "flatten", "shuffle"]):
        return "shape"
    if "softmax" in text:
        return "softmax"
    return str(layer_type) if str(layer_type) and str(layer_type) != "nan" else "op"


def parent_key(layer_name, runtime):
    name = str(layer_name)
    if runtime.startswith("pytorch"):
        parts = name.split(".")
        if len(parts) >= 2 and re.fullmatch(r"\d+|activation|scale_activation|avgpool|fc\d+", parts[-1]):
            return ".".join(parts[:-1])
        return ".".join(parts[:-1]) if len(parts) > 1 else name

    if runtime.startswith("onnxrt"):
        cleaned = re.sub(r"_(kernel_time|fence_before|fence_after)$", "", name)
        cleaned = cleaned.strip("/")
        parts = [p for p in cleaned.split("/") if p]
        return "/".join(parts[:-1]) if len(parts) > 1 else cleaned

    if runtime == "tflite_cpu":
        parts = re.split(r"/|:", name)
        return "/".join(parts[:-1]) if len(parts) > 1 else name

    if runtime == "ncnn_vulkan":
        if name.endswith(".param"):
            return "model inference"
        return re.sub(r"(_\d+)?$", "", name)

    if runtime.startswith("tensorrt"):
        cleaned = re.sub(r"^\[[^]]+\]\s*", "", name)
        return cleaned.split(":")[0].strip()

    return name


def is_blank(value):
    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def first_value(series):
    for value in series:
        if not is_blank(value):
            return str(value)
    return ""


def last_value(series):
    for value in reversed(list(series)):
        if not is_blank(value):
            return str(value)
    return ""


def format_params(value):
    if pd.isna(value):
        return "0"
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    if number >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if number >= 1_000:
        return f"{number / 1_000:.1f}K"
    return str(number)


def metadata_label(base_label, input_shape="", output_shape="", param_count=0):
    parts = [base_label]
    meta = []
    if input_shape:
        meta.append(f"in {input_shape}")
    if output_shape:
        meta.append(f"out {output_shape}")
    meta.append(f"params {format_params(param_count)}")
    parts.append(" | ".join(meta))
    return "\n".join(parts)


def pytorch_group_key(layer_name, layer_type, active_key):
    name = str(layer_name)
    op_type = norm_type(layer_type, name)

    if op_type in {"conv", "fc", "pool"}:
        match = re.match(r"^(?P<prefix>.+)\.(?P<kind>conv|bn|fc)(?P<num>\d+)$", name)
        if match:
            return f"{match.group('prefix')}.{match.group('kind')}{match.group('num')}", True
        match = re.match(r"^(?P<prefix>.+)\.(?P<num>\d+)$", name)
        if match:
            return f"{match.group('prefix')}.{match.group('num')}", True
        return name, True

    if op_type == "norm":
        match = re.match(r"^(?P<prefix>.+)\.bn(?P<num>\d+)$", name)
        if match:
            return f"{match.group('prefix')}.conv{match.group('num')}", False
        match = re.match(r"^(?P<prefix>.+)\.(?P<num>\d+)$", name)
        if match:
            previous = int(match.group("num")) - 1
            return f"{match.group('prefix')}.{previous}", False
        return active_key or name, False

    if op_type == "act":
        return active_key or name, False

    if "downsample.1" in name:
        return name.replace("downsample.1", "downsample.0"), False

    return name, True


def summarize_ops(types):
    order = ["conv", "norm", "act", "pool", "fc", "eltwise", "concat/split", "shape", "softmax"]
    unique = []
    for item in types:
        if item not in unique:
            unique.append(item)
    ordered = [item for item in order if item in unique]
    ordered += [item for item in unique if item not in ordered]
    if ordered[:3] == ["conv", "norm", "act"]:
        return "conv-norm-act"
    if ordered[:2] == ["conv", "act"]:
        return "conv-act"
    if ordered[:2] == ["conv", "norm"]:
        return "conv-norm"
    return "-".join(ordered[:4])


def group_layers(layer_df):
    if layer_df.empty:
        return pd.DataFrame()

    rows = []
    for (model, runtime), sub in layer_df.groupby(["model", "runtime"], sort=False):
        sub = sub.sort_values("layer_index").copy()
        if str(runtime).startswith("tensorrt"):
            for group_index, (_, row) in enumerate(sub.iterrows(), start=1):
                raw_name = str(row["layer_name"])
                cleaned_name = re.sub(r"^\[[^]]+\]\s*\[[^]]+\]\s*", "", raw_name).strip()
                if not cleaned_name or cleaned_name == raw_name:
                    cleaned_name = f"TensorRT layer {group_index:03d}"
                label_type = norm_type(row.get("type", "TensorRTLayer"), cleaned_name)
                rows.append({
                    "model": model,
                    "runtime": runtime,
                    "group_index": group_index,
                    "group_key": cleaned_name,
                    "group_label": f"{group_index:03d}. {shorten_label(cleaned_name)} [{label_type}]",
                    "group_type": label_type,
                    "mean_ms": row["mean_ms"],
                    "power_gpu_mw": row.get("power_gpu_mw", np.nan),
                    "power_total_mw": row.get("power_total_mw", np.nan),
                    "ops": 1,
                })
            continue

        if str(runtime).startswith("pytorch"):
            rows.extend(group_pytorch_layers(model, runtime, sub))
            continue

        sub["op_group"] = [parent_key(name, runtime) for name in sub["layer_name"]]
        sub["op_type"] = [norm_type(t, n) for t, n in zip(sub["type"], sub["layer_name"])]
        for group_index, (key, group) in enumerate(sub.groupby("op_group", sort=False), start=1):
            op_types = list(group["op_type"])
            label_type = summarize_ops(op_types)
            param_count = group["param_count"].sum() if "param_count" in group else 0
            label = metadata_label(
                f"{group_index:02d}. {shorten_label(key)} [{label_type}]",
                first_value(group["input_shape"]) if "input_shape" in group else "",
                last_value(group["output_shape"]) if "output_shape" in group else "",
                param_count,
            )
            rows.append({
                "model": model,
                "runtime": runtime,
                "group_index": group_index,
                "group_key": key,
                "group_label": label,
                "group_type": label_type,
                "mean_ms": group["mean_ms"].sum(),
                "power_gpu_mw": group["power_gpu_mw"].mean() if "power_gpu_mw" in group else np.nan,
                "power_total_mw": group["power_total_mw"].mean() if "power_total_mw" in group else np.nan,
                "ops": len(group),
                "input_shape": first_value(group["input_shape"]) if "input_shape" in group else "",
                "output_shape": last_value(group["output_shape"]) if "output_shape" in group else "",
                "param_count": param_count,
            })
    return pd.DataFrame(rows)


def group_pytorch_layers(model, runtime, sub):
    rows = []
    active_key = None
    sub = sub.sort_values("layer_index").copy()
    group_keys = []

    for _, row in sub.iterrows():
        key, starts_new = pytorch_group_key(row["layer_name"], row["type"], active_key)
        if starts_new or active_key is None:
            active_key = key
        group_keys.append(key)

    sub["op_group"] = group_keys
    sub["op_type"] = [norm_type(t, n) for t, n in zip(sub["type"], sub["layer_name"])]

    for group_index, (key, group) in enumerate(sub.groupby("op_group", sort=False), start=1):
        op_types = list(group["op_type"])
        label_type = summarize_ops(op_types)
        param_count = group["param_count"].sum() if "param_count" in group else 0
        label = metadata_label(
            f"{group_index:03d}. {shorten_label(key)} [{label_type}]",
            first_value(group["input_shape"]) if "input_shape" in group else "",
            last_value(group["output_shape"]) if "output_shape" in group else "",
            param_count,
        )
        rows.append({
            "model": model,
            "runtime": runtime,
            "group_index": group_index,
            "group_key": key,
            "group_label": label,
            "group_type": label_type,
            "mean_ms": group["mean_ms"].sum(),
            "power_gpu_mw": group["power_gpu_mw"].mean() if "power_gpu_mw" in group else np.nan,
            "power_total_mw": group["power_total_mw"].mean() if "power_total_mw" in group else np.nan,
            "ops": len(group),
            "input_shape": first_value(group["input_shape"]) if "input_shape" in group else "",
            "output_shape": last_value(group["output_shape"]) if "output_shape" in group else "",
            "param_count": param_count,
        })
    return rows


def shorten_label(label, max_len=58):
    text = str(label).strip("/") or "model"
    text = re.sub(r"^features/features\.", "features.", text)
    text = re.sub(r"/+", "/", text)
    if len(text) <= max_len:
        return text
    return "..." + text[-(max_len - 3):]


def save_fig(fig, path_base):
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png_path = path_base.with_suffix(".png")
    pdf_path = path_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] {png_path}")


def bar_values(ax, values, horizontal=False, fmt="{:.2f}"):
    for idx, value in enumerate(values):
        if pd.isna(value):
            continue
        if horizontal:
            ax.text(value, idx, f" {fmt.format(value)}", va="center", fontsize=8)
        else:
            ax.text(idx, value, fmt.format(value), ha="center", va="bottom", fontsize=8, rotation=0)


def plot_model_runtime_bars(model, model_df, out_dir):
    sub = ordered_runtime_frame(model_df.dropna(subset=["mean_ms"]))
    if sub.empty:
        return
    labels = [runtime_label(rt) for rt in sub["runtime"]]
    x = np.arange(len(sub))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(sub)))

    axes[0].bar(x, sub["mean_ms"], color=colors)
    axes[0].set_title(f"{model_label(model)} Runtime Latency")
    axes[0].set_ylabel("Mean latency (ms)")
    axes[0].grid(axis="y", alpha=0.25)
    bar_values(axes[0], list(sub["mean_ms"]))

    power_col = "power_total_mw_mean"
    if power_col in sub.columns and sub[power_col].fillna(0).sum() > 0:
        power_values = sub[power_col]
        power_label = "Total power (mW)"
    elif "gpu_util_pct_mean" in sub.columns:
        power_values = sub["gpu_util_pct_mean"]
        power_label = "GPU utilization (%) - power unavailable"
    else:
        power_values = pd.Series([np.nan] * len(sub))
        power_label = "Power unavailable"

    axes[1].bar(x, power_values, color=colors)
    axes[1].set_title(f"{model_label(model)} Runtime Power")
    axes[1].set_ylabel(power_label)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    bar_values(axes[1], list(power_values))

    save_fig(fig, out_dir / "runtime_comparison")


def plot_layer_group_bars(model, runtime, grouped_df, out_dir):
    sub = grouped_df[(grouped_df["model"] == model) & (grouped_df["runtime"] == runtime)].copy()
    sub = sub.dropna(subset=["mean_ms"])
    if sub.empty:
        return

    sub = sub.sort_values("group_index", ascending=True)
    height = max(7, 0.28 * len(sub) + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(16, height), sharey=True)
    y = np.arange(len(sub))

    axes[0].barh(y, sub["mean_ms"], color="#4C78A8")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub["group_label"], fontsize=8)
    axes[0].set_xlabel("Grouped latency (ms)")
    axes[0].set_title("Layer group speed")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].invert_yaxis()
    bar_values(axes[0], list(sub["mean_ms"]), horizontal=True)

    power_col = "power_total_mw"
    if power_col in sub.columns and sub[power_col].fillna(0).sum() > 0:
        power_values = sub[power_col]
        x_label = "Grouped avg total power (mW)"
    elif "power_gpu_mw" in sub.columns and sub["power_gpu_mw"].fillna(0).sum() > 0:
        power_values = sub["power_gpu_mw"]
        x_label = "Grouped avg GPU power (mW)"
    else:
        power_values = pd.Series([0] * len(sub), index=sub.index)
        x_label = "Power mapped as 0/unavailable"

    axes[1].barh(y, power_values, color="#F58518")
    axes[1].set_xlabel(x_label)
    axes[1].set_title("Layer group power")
    axes[1].grid(axis="x", alpha=0.25)
    bar_values(axes[1], list(power_values), horizontal=True)

    fig.suptitle(f"{model_label(model)} / {runtime_label(runtime)} Layer Groups", y=1.01, fontsize=14)
    save_fig(fig, out_dir / f"layer_groups_{runtime}")


def write_model_summary(model, model_df, grouped_df, out_dir):
    summary_path = out_dir / "summary.csv"
    ordered_runtime_frame(model_df).drop(columns=["runtime_order"], errors="ignore").to_csv(summary_path, index=False)
    if not grouped_df.empty:
        grouped_df[grouped_df["model"] == model].to_csv(out_dir / "layer_groups.csv", index=False)
    print(f"[Summary] {summary_path}")


def visualize(df, layer_df, output_dir):
    figures_root = output_dir / "figures_by_model"
    figures_root.mkdir(parents=True, exist_ok=True)
    grouped_df = group_layers(layer_df)

    for model in sorted(df["model"].dropna().unique()):
        model_out = figures_root / str(model)
        model_out.mkdir(parents=True, exist_ok=True)
        model_df = df[df["model"] == model].copy()
        write_model_summary(model, model_df, grouped_df, model_out)
        plot_model_runtime_bars(model, model_df, model_out)

        if grouped_df.empty:
            continue
        model_layers = grouped_df[grouped_df["model"] == model]
        for runtime in [rt for rt in RUNTIME_ORDER if rt in set(model_layers["runtime"])]:
            plot_layer_group_bars(model, runtime, grouped_df, model_out)

    if not grouped_df.empty:
        grouped_df.to_csv(output_dir / "layer_groups_all.csv", index=False)
    df.to_csv(output_dir / "summary_all_models.csv", index=False)
    print(f"[Summary] {output_dir / 'summary_all_models.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Create per-model Jetson benchmark visualizations")
    parser.add_argument("--results-dir", default="./results", help="Directory containing benchmark CSV files")
    parser.add_argument("--output-dir", default="./results", help="Directory for generated figures")
    parser.add_argument("--top-layers", type=int, default=0, help="Deprecated; all layer groups are always plotted")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "grid.color": "#DDDDDD",
    })

    benchmark_df = load_benchmarks(results_dir)
    layer_df = load_layer_profiles(results_dir)

    if benchmark_df.empty:
        print(f"[Warning] No benchmark_*.csv files found in {results_dir}")
        return

    if layer_df.empty:
        print(f"[Warning] No layer_power_*.csv files found in {results_dir}")

    visualize(benchmark_df, layer_df, output_dir)
    print("[Done] Per-model visualization complete.")


if __name__ == "__main__":
    main()
