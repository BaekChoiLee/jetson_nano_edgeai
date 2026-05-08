import numpy as np
np.typeDict = np.sctypeDict
#!/usr/bin/env python3
"""Generate 6 benchmark visualization charts.

Charts:
1. Runtime speed heatmap (latency by runtime × model/power)
2. Power-speed tradeoff scatter plot
3. Layer-wise latency waterfall (top-15 layers)
4. Carbon emissions bar chart
5. TRT quantization comparison (placeholder if no TRT data)
6. 5W vs 10W performance comparison

Reads from:
  - results/results/all_results.csv
  - results/results/carbon_summary.csv
  - results/results/*_layers.csv

Outputs to: charts/ directory (PNG + PDF)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Style configuration
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (10, 6),
})
sns.set_theme(style="whitegrid", palette="Set2")

# Color palette for runtimes
RUNTIME_COLORS = {
    "pytorch_cpu": "#1f77b4",
    "pytorch_cuda": "#ff7f0e",
    "tensorrt_fp32": "#2ca02c",
    "tensorrt_fp16": "#d62728",
    "tensorrt_int8": "#9467bd",
    "onnxrt_cpu": "#bcbd22",
    "onnxrt_cuda": "#8c564b",
    "tflite_cpu": "#e377c2",
    "ncnn_python": "#7f7f7f",
}

# Short labels for readability in charts
RUNTIME_SHORT = {
    "pytorch_cpu": "PT-CPU",
    "pytorch_cuda": "PT-CUDA",
    "tensorrt_fp32": "TRT-FP32",
    "tensorrt_fp16": "TRT-FP16",
    "tensorrt_int8": "TRT-INT8",
    "onnxrt_cpu": "ORT-CPU",
    "onnxrt_cuda": "ORT-CUDA",
    "tflite_cpu": "TFLite",
    "ncnn_python": "ncnn",
}

MODEL_MARKERS = {
    "mobilenetv3_small": "o",
    "resnet50": "s",
}


def save_chart(fig, charts_dir, name):
    """Save chart as both PNG and PDF."""
    png_path = os.path.join(charts_dir, f"{name}.png")
    pdf_path = os.path.join(charts_dir, f"{name}.pdf")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {png_path}")


def chart1_latency_heatmap(df, charts_dir):
    """Chart 1: Runtime × Model/Power latency heatmap."""
    print("\n[Chart 1] Runtime Speed Heatmap")

    # Create pivot: rows=runtime, cols=model_power
    df_plot = df.copy()
    df_plot["model_power"] = df_plot["model"] + "\n" + df_plot["power_mode"]

    pivot = df_plot.pivot_table(
        values="latency_mean_ms",
        index="runtime",
        columns="model_power",
        aggfunc="first",
    )

    if pivot.empty:
        print("  [SKIP] No data for heatmap")
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.8)))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Latency (ms)"},
    )

    ax.set_title("Inference Latency by Runtime and Model (ms)")
    ax.set_xlabel("Model / Power Mode")
    ax.set_ylabel("Runtime")

    save_chart(fig, charts_dir, "01_latency_heatmap")


def chart2_power_speed_scatter(df, charts_dir):
    """Chart 2: Power vs Speed tradeoff scatter plot."""
    print("\n[Chart 2] Power-Speed Tradeoff Scatter")

    # Filter out entries with 0 power (incomplete tegrastats)
    df_plot = df[df["power_in_mean_mw"] > 0].copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    for _, row in df_plot.iterrows():
        runtime = row["runtime"]
        model = row["model"]
        color = RUNTIME_COLORS.get(runtime, "#333333")
        marker = MODEL_MARKERS.get(model, "^")
        edge = "black" if row["power_mode"] == "10w" else "gray"

        ax.scatter(
            row["latency_mean_ms"],
            row["power_in_mean_mw"],
            c=color,
            marker=marker,
            s=120,
            edgecolors=edge,
            linewidths=1.5,
            zorder=5,
        )

        short_rt = RUNTIME_SHORT.get(runtime, runtime)
        label_text = f"{short_rt}\n{row['power_mode']}"
        ax.annotate(
            label_text,
            (row["latency_mean_ms"], row["power_in_mean_mw"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=6,
            alpha=0.8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference Latency (ms, log scale)")
    ax.set_ylabel("System Power (mW)")
    ax.set_title("Power-Speed Tradeoff: Lower-Left is Better")

    # Add legend for runtimes and models
    from matplotlib.lines import Line2D
    legend_elements = []
    for rt, color in RUNTIME_COLORS.items():
        short = RUNTIME_SHORT.get(rt, rt)
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=8, label=short)
        )
    legend_elements.append(Line2D([0], [0], color="w", label=""))  # spacer
    for model, marker in MODEL_MARKERS.items():
        legend_elements.append(
            Line2D([0], [0], marker=marker, color="w", markerfacecolor="gray",
                   markersize=10, label=model)
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=10, label="10W mode")
    )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="lightgray", markersize=10, label="5W mode")
    )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7, ncol=2)

    ax.grid(True, alpha=0.3)
    save_chart(fig, charts_dir, "02_power_speed_scatter")


def chart3_layer_waterfall(layers_dir, charts_dir):
    """Chart 3: Layer-wise latency waterfall (top-15 layers per model)."""
    print("\n[Chart 3] Layer Latency Waterfall")

    layer_files = [
        ("mobilenetv3_small", os.path.join(layers_dir, "mobilenetv3_small_layers.csv")),
        ("resnet50", os.path.join(layers_dir, "resnet50_layers.csv")),
    ]

    for model_name, csv_path in layer_files:
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {csv_path} not found")
            continue

        df_layers = pd.read_csv(csv_path)
        df_top = df_layers.nlargest(15, "mean_ms")

        fig, ax = plt.subplots(figsize=(10, 7))

        # Horizontal bar chart
        y_pos = range(len(df_top))
        bars = ax.barh(
            y_pos,
            df_top["mean_ms"].values,
            xerr=df_top["std_ms"].values,
            color=sns.color_palette("viridis", len(df_top)),
            edgecolor="white",
            linewidth=0.5,
            capsize=3,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{name}\n({ltype})" for name, ltype in
             zip(df_top["layer_name"], df_top["layer_type"])],
            fontsize=7,
        )
        ax.invert_yaxis()
        ax.set_xlabel("Latency (ms)")
        ax.set_title(f"Top-15 Slowest Layers: {model_name}")

        # Add value labels
        for bar, val in zip(bars, df_top["mean_ms"].values):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=7)

        save_chart(fig, charts_dir, f"03_layer_waterfall_{model_name}")


def chart4_carbon_bar(carbon_path, charts_dir):
    """Chart 4: Carbon emissions per inference by runtime (split by model)."""
    print("\n[Chart 4] Carbon Emissions Bar Chart")

    if not os.path.exists(carbon_path):
        print(f"  [SKIP] {carbon_path} not found")
        return

    df = pd.read_csv(carbon_path)
    # Filter out entries with 0 energy (incomplete data)
    df = df[df["energy_per_inference_mj"] > 0].copy()

    models = df["model"].unique()
    fig, axes = plt.subplots(len(models), 1, figsize=(14, 6 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, sorted(models)):
        model_df = df[df["model"] == model].copy()
        short_model = model.replace("mobilenetv3_small", "MNv3-S")
        model_df["label"] = model_df["runtime"].map(
            lambda r: RUNTIME_SHORT.get(r, r)
        ) + "\n" + model_df["power_mode"]

        colors = [RUNTIME_COLORS.get(r, "#333") for r in model_df["runtime"]]
        x = range(len(model_df))
        bars = ax.bar(x, model_df["energy_per_inference_mj"].values,
                      color=colors, edgecolor="white")

        ax.set_xticks(list(x))
        ax.set_xticklabels(model_df["label"].values, fontsize=8)
        ax.set_ylabel("Energy per Inference (mJ)")
        ax.set_title(f"Energy Consumption per Inference: {short_model}")

        # Add energy value on each bar
        for bar, energy in zip(bars, model_df["energy_per_inference_mj"].values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{energy:.1f}",
                    ha="center", va="bottom", fontsize=7)

        # Secondary axis for power
        ax2 = ax.twinx()
        ax2.plot(list(x), model_df["power_mean_mw"].values, "k--o",
                 alpha=0.5, label="Power (mW)")
        ax2.set_ylabel("Mean Power (mW)")
        ax2.legend(loc="upper left")

    fig.tight_layout()
    save_chart(fig, charts_dir, "04_carbon_emissions")


def chart5_trt_quantization(df, charts_dir):
    """Chart 5: TRT quantization comparison (FP32 vs FP16 vs INT8).

    If no TRT data, create a placeholder noting missing data.
    """
    print("\n[Chart 5] TRT Quantization Comparison")

    trt_data = df[df["runtime"].str.startswith("tensorrt_")]

    if trt_data.empty:
        # Create placeholder chart
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5,
                "TensorRT data not available\nin current results.\n\n"
                "Run benchmark on Jetson with\nTRT engine files to populate.",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax.transAxes)
        ax.set_title("TRT Quantization: FP32 vs FP16 vs INT8")
        ax.set_axis_off()
        save_chart(fig, charts_dir, "05_trt_quantization")
        return

    # Group by model for side-by-side comparison
    models = trt_data["model"].unique()
    precisions = ["tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8"]

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_data = trt_data[trt_data["model"] == model]
        x = np.arange(len(precisions))
        width = 0.35

        for i, pm in enumerate(["10w", "5w"]):
            pm_data = model_data[model_data["power_mode"] == pm]
            values = []
            for p in precisions:
                match = pm_data[pm_data["runtime"] == p]
                values.append(match["latency_mean_ms"].iloc[0] if len(match) > 0 else 0)
            offset = -width / 2 + i * width
            bars = ax.bar(x + offset, values, width, label=pm)
            for bar, v in zip(bars, values):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{v:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(["FP32", "FP16", "INT8"])
        ax.set_title(model)
        ax.set_ylabel("Latency (ms)")
        ax.legend()

    fig.suptitle("TensorRT Quantization Comparison", fontsize=14)
    save_chart(fig, charts_dir, "05_trt_quantization")


def chart6_5w_vs_10w(df, charts_dir):
    """Chart 6: 5W vs 10W performance comparison with speedup ratio."""
    print("\n[Chart 6] 5W vs 10W Comparison")

    # Pivot to get 5w and 10w side by side
    models = df["model"].unique()
    runtimes = df["runtime"].unique()

    labels = []
    latency_10w = []
    latency_5w = []

    for model in sorted(models):
        for runtime in sorted(runtimes):
            data_10w = df[(df["model"] == model) & (df["runtime"] == runtime) & (df["power_mode"] == "10w")]
            data_5w = df[(df["model"] == model) & (df["runtime"] == runtime) & (df["power_mode"] == "5w")]

            if data_10w.empty or data_5w.empty:
                continue

            l10 = data_10w["latency_mean_ms"].iloc[0]
            l5 = data_5w["latency_mean_ms"].iloc[0]
            if pd.isna(l10) or pd.isna(l5) or l10 <= 0 or l5 <= 0:
                continue
            short_model = model.replace("mobilenetv3_small", "MNv3-S")
            short_rt = RUNTIME_SHORT.get(runtime, runtime)
            labels.append(f"{short_model}\n{short_rt}")
            latency_10w.append(l10)
            latency_5w.append(l5)

    if not labels:
        print("  [SKIP] Not enough paired data")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]})

    x = np.arange(len(labels))
    width = 0.35

    bars_10w = ax1.bar(x - width / 2, latency_10w, width, label="10W (MAXN)",
                       color="#ff7f0e", edgecolor="white")
    bars_5w = ax1.bar(x + width / 2, latency_5w, width, label="5W",
                      color="#1f77b4", edgecolor="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("5W vs 10W Mode: Inference Latency Comparison")
    ax1.legend()
    ax1.set_yscale("log")

    # Add value labels
    for bar in list(bars_10w) + list(bars_5w):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height,
                 f"{height:.1f}", ha="center", va="bottom", fontsize=7)

    # Speedup ratio subplot
    speedup = [l5 / l10 for l5, l10 in zip(latency_5w, latency_10w)]
    colors = ["#2ca02c" if s > 1 else "#d62728" for s in speedup]
    ax2.bar(x, speedup, 0.6, color=colors, edgecolor="white")
    ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Slowdown Ratio\n(5W / 10W)")
    ax2.set_title("Performance Impact of Power Mode (>1 = slower at 5W)")

    for i, s in enumerate(speedup):
        ax2.text(i, s, f"{s:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    save_chart(fig, charts_dir, "06_5w_vs_10w")


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    results_dir = os.path.join(base_dir, "results", "results")
    charts_dir = os.path.join(base_dir, "charts")

    os.makedirs(charts_dir, exist_ok=True)

    # Load consolidated data
    csv_path = os.path.join(results_dir, "all_results.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found. Run consolidate_results.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} experiments from {csv_path}")

    carbon_path = os.path.join(results_dir, "carbon_summary.csv")
    # Layer CSV files are in results_jetson_full directory
    layers_dir_full = os.path.join(base_dir, "results", "results_jetson_full")
    layers_dir = layers_dir_full if os.path.exists(layers_dir_full) else results_dir

    # Generate all charts
    chart1_latency_heatmap(df, charts_dir)
    chart2_power_speed_scatter(df, charts_dir)
    chart3_layer_waterfall(layers_dir, charts_dir)
    chart4_carbon_bar(carbon_path, charts_dir)
    chart5_trt_quantization(df, charts_dir)
    chart6_5w_vs_10w(df, charts_dir)

    print(f"\nAll charts saved to: {charts_dir}")
    print(f"Files: {sorted(os.listdir(charts_dir))}")


if __name__ == "__main__":
    main()
