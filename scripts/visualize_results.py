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
    "pytorch_cpu",
    "pytorch_cuda",
    "onnxrt_cpu",
    "tflite_cpu",
    "ncnn_vulkan",
    "tensorrt_fp32",
    "tensorrt_fp16",
    "tensorrt_int8",
]


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

    df["runtime"] = pd.Categorical(df["runtime"], categories=RUNTIME_ORDER, ordered=True)
    return df.sort_values(["model", "runtime"]).reset_index(drop=True)


def load_layer_profiles(results_dir):
    frames = []
    pattern = re.compile(r"^layer_power_(?P<model>.+)_(?P<runtime>pytorch_cpu|pytorch_cuda|onnxrt_cpu|tflite_cpu|ncnn_vulkan|tensorrt_fp32|tensorrt_fp16|tensorrt_int8)\.csv$")
    for path in sorted(results_dir.glob("layer_power_*.csv")):
        match = pattern.match(path.name)
        if not match or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["model"] = match.group("model")
        df["runtime"] = match.group("runtime")
        df["layer_index"] = range(1, len(df) + 1)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for col in ["mean_ms", "power_gpu_mw", "power_total_mw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_current(figures_dir, name):
    png_path = figures_dir / f"{name}.png"
    pdf_path = figures_dir / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[Figure] {png_path}")


def plot_latency_heatmap(df, figures_dir):
    if df.empty or "mean_ms" not in df.columns:
        return
    table = df.pivot_table(index="model", columns="runtime", values="mean_ms", aggfunc="mean", observed=False)
    table = table.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if table.empty:
        return

    plt.figure(figsize=(12, max(4, 0.6 * len(table))))
    values = table.to_numpy(dtype=float)
    im = plt.imshow(values, cmap="viridis_r", aspect="auto")
    plt.colorbar(im, label="Mean Latency (ms)")
    plt.xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
    plt.yticks(range(len(table.index)), table.index)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            if not np.isnan(values[y, x]):
                plt.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.title("Mean Latency by Model and Runtime")
    plt.xlabel("Runtime")
    plt.ylabel("Model")
    save_current(figures_dir, "latency_heatmap")


def plot_power_latency_scatter(df, figures_dir):
    required = {"mean_ms", "power_total_mw_mean", "runtime", "model"}
    if df.empty or not required.issubset(df.columns):
        return
    plot_df = df.dropna(subset=["mean_ms", "power_total_mw_mean"])
    if plot_df.empty:
        return

    plt.figure(figsize=(10, 6))
    runtimes = list(plot_df["runtime"].astype(str).dropna().unique())
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    for idx, runtime in enumerate(runtimes):
        sub = plot_df[plot_df["runtime"].astype(str) == runtime]
        plt.scatter(
            sub["mean_ms"],
            sub["power_total_mw_mean"],
            s=90,
            marker=markers[idx % len(markers)],
            label=runtime,
            alpha=0.85,
        )
        for _, row in sub.iterrows():
            plt.annotate(str(row["model"]), (row["mean_ms"], row["power_total_mw_mean"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.title("Power-Latency Trade-off")
    plt.xlabel("Mean Latency (ms)")
    plt.ylabel("Total Power VDD_IN (mW)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current(figures_dir, "power_latency_tradeoff")


def plot_co2_bars(df, figures_dir):
    required = {"model", "runtime", "co2_kg"}
    if df.empty or not required.issubset(df.columns):
        return
    plot_df = df.dropna(subset=["co2_kg"])
    if plot_df.empty:
        return
    plot_df = plot_df.copy()
    plot_df["co2_g"] = plot_df["co2_kg"] * 1000.0

    models = sorted(plot_df["model"].dropna().unique())
    height = max(4, 2.6 * len(models))
    fig, axes = plt.subplots(len(models), 1, figsize=(12, height), squeeze=False)
    for ax, model in zip(axes.flat, models):
        sub = plot_df[plot_df["model"] == model]
        sub = sub.set_index(sub["runtime"].astype(str)).reindex(RUNTIME_ORDER).dropna(subset=["co2_g"])
        ax.bar(sub.index.astype(str), sub["co2_g"])
        ax.set_title(model)
        ax.set_xlabel("")
        ax.set_ylabel("gCO2eq")
        ax.tick_params(axis="x", rotation=30)
    save_current(figures_dir, "co2_by_runtime")


def plot_tensorrt_efficiency_curve(df, figures_dir):
    required = {"model", "runtime", "mean_ms", "accuracy_top1"}
    if df.empty or not required.issubset(df.columns):
        return
    plot_df = df[df["runtime"].astype(str).isin(["tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8"])]
    plot_df = plot_df.dropna(subset=["mean_ms", "accuracy_top1"])
    if plot_df.empty:
        return

    plt.figure(figsize=(10, 6))
    for model, sub in plot_df.groupby("model"):
        sub = sub.set_index(sub["runtime"].astype(str)).reindex(["tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8"]).dropna(subset=["mean_ms", "accuracy_top1"])
        if sub.empty:
            continue
        plt.plot(sub["mean_ms"], sub["accuracy_top1"], marker="o", label=model)
        for runtime, row in sub.iterrows():
            plt.annotate(runtime.replace("tensorrt_", ""), (row["mean_ms"], row["accuracy_top1"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.title("TensorRT Accuracy-Efficiency Curve")
    plt.xlabel("Mean Latency (ms)")
    plt.ylabel("Top-1 Accuracy or mAP")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current(figures_dir, "tensorrt_accuracy_efficiency")


def plot_layer_waterfall(layer_df, figures_dir):
    required = {"model", "runtime", "layer_name", "mean_ms", "layer_index"}
    if layer_df.empty or not required.issubset(layer_df.columns):
        return
    row = layer_df.dropna(subset=["mean_ms"]).head(1)
    if row.empty:
        return
    model = row.iloc[0]["model"]
    runtime = row.iloc[0]["runtime"]
    sub = layer_df[(layer_df["model"] == model) & (layer_df["runtime"] == runtime)].dropna(subset=["mean_ms"])
    if sub.empty:
        return

    sub = sub.sort_values("layer_index").copy()
    sub["cum_ms"] = sub["mean_ms"].cumsum()
    plt.figure(figsize=(12, 6))
    plt.bar(sub["layer_index"], sub["mean_ms"])
    plt.plot(sub["layer_index"], sub["cum_ms"], color="black", marker="o", linewidth=1.5)
    plt.title(f"Layer Latency Waterfall: {model} / {runtime}")
    plt.xlabel("Layer Index")
    plt.ylabel("Latency (ms)")
    plt.grid(True, axis="y", alpha=0.3)
    save_current(figures_dir, f"layer_waterfall_{model}_{runtime}")


def plot_layer_type_bars(layer_df, figures_dir):
    required = {"runtime", "type", "mean_ms"}
    if layer_df.empty or not required.issubset(layer_df.columns):
        return
    plot_df = layer_df.dropna(subset=["mean_ms", "type"])
    if plot_df.empty:
        return

    table = plot_df.pivot_table(index="type", columns="runtime", values="mean_ms", aggfunc="mean", observed=False)
    table = table.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if table.empty:
        return

    plt.figure(figsize=(12, 6))
    x = np.arange(len(table.index))
    width = 0.8 / max(1, len(table.columns))
    for idx, runtime in enumerate(table.columns):
        offsets = x - 0.4 + width / 2 + idx * width
        plt.bar(offsets, table[runtime], width=width, label=str(runtime))
    plt.title("Mean Layer Latency by Layer Type")
    plt.xlabel("Layer Type")
    plt.ylabel("Mean Latency (ms)")
    plt.xticks(x, table.index, rotation=30, ha="right")
    plt.legend()
    save_current(figures_dir, "layer_type_latency")


def save_summary(df, output_dir):
    if df.empty:
        return
    summary_cols = [
        "model",
        "runtime",
        "precision",
        "power_mode",
        "mean_ms",
        "std_ms",
        "median_ms",
        "power_total_mw_mean",
        "power_gpu_mw_mean",
        "ram_used_mb_mean",
        "gpu_util_pct_mean",
        "gpu_temp_c_mean",
        "co2_kg",
        "accuracy_top1",
    ]
    cols = [col for col in summary_cols if col in df.columns]
    summary = df[cols].copy()
    path = output_dir / "summary_all_models.csv"
    summary.to_csv(path, index=False)
    print(f"[Summary] {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Jetson Nano benchmark results")
    parser.add_argument("--results-dir", default="./results", help="Directory containing benchmark CSV files")
    parser.add_argument("--output-dir", default="./results", help="Directory for summary CSV and figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("ggplot")
    df = load_benchmarks(results_dir)
    layer_df = load_layer_profiles(results_dir)

    if df.empty:
        print(f"[Warning] No benchmark_*.csv files found in {results_dir}")
    else:
        save_summary(df, output_dir)
        plot_latency_heatmap(df, figures_dir)
        plot_power_latency_scatter(df, figures_dir)
        plot_co2_bars(df, figures_dir)
        plot_tensorrt_efficiency_curve(df, figures_dir)

    if layer_df.empty:
        print(f"[Warning] No layer_power_*.csv files found in {results_dir}")
    else:
        plot_layer_waterfall(layer_df, figures_dir)
        plot_layer_type_bars(layer_df, figures_dir)

    print("[Done] Visualization complete.")


if __name__ == "__main__":
    main()
