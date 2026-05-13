#!/usr/bin/env python3
"""Build dashboard-ready CSVs and layer-runtime charts from my_results."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


MODELS = [
    "mobilenetv3_small",
    "resnet50",
    "efficientnet_b0",
    "shufflenet_v2_x1_0",
    "yolov8n",
    "ssd_mobilenet_v2",
]

TRT_RUNTIMES = {"tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8"}
POWER_ORDER = {"MAXN": 0, "10w": 0, "5W": 1, "5w": 1}
RUNTIME_ORDER = {
    "pytorch_cpu": 0,
    "pytorch_cuda": 1,
    "tensorrt_fp32": 2,
    "tensorrt_fp16": 3,
    "tensorrt_int8": 4,
    "onnxrt_cuda": 5,
    "onnxrt_trt": 6,
    "tflite_cpu": 7,
    "tflite_gpu": 8,
    "ncnn_cpu": 9,
    "ncnn_vulkan": 10,
}

RUNTIME_LABELS = {
    "pytorch_cpu": "PyTorch CPU",
    "pytorch_cuda": "PyTorch CUDA",
    "tensorrt_fp32": "TRT FP32",
    "tensorrt_fp16": "TRT FP16",
    "tensorrt_int8": "TRT INT8",
    "onnxrt_cuda": "ORT CUDA",
    "onnxrt_trt": "ORT TRT",
    "tflite_cpu": "TFLite CPU",
    "tflite_gpu": "TFLite GPU",
    "ncnn_cpu": "ncnn CPU",
    "ncnn_vulkan": "ncnn Vulkan",
}


def is_trt_runtime(runtime: object) -> bool:
    return str(runtime) in TRT_RUNTIMES


def normalize_power_mode(value: object) -> str:
    text = str(value)
    if text.lower() == "10w":
        return "MAXN"
    if text.lower() == "5w":
        return "5W"
    return text


def model_from_engine_or_dir(row: pd.Series, summary_path: Path) -> str:
    dirname = summary_path.parent.name
    for model in MODELS:
        if dirname.startswith(f"{model}_"):
            return model

    raw_model = str(row.get("model", ""))
    for suffix in ("_fp32.engine", "_fp16.engine", "_int8.engine"):
        if raw_model.endswith(suffix):
            return raw_model[: -len(suffix)]
    return raw_model


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def align_columns(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    columns: list[str] = []
    seen = set()
    for frame in frames:
        for col in frame.columns:
            if col not in seen:
                seen.add(col)
                columns.append(col)
    return [frame.reindex(columns=columns) for frame in frames]


def build_latency(clean_dir: Path, trt_dir: Path, output_dir: Path) -> pd.DataFrame:
    clean_path = clean_dir / "consolidated_latency.csv"
    clean_df = read_csv_if_exists(clean_path)
    if clean_df.empty:
        raise FileNotFoundError(f"Missing or empty latency CSV: {clean_path}")

    clean_df = clean_df[~clean_df["runtime"].map(is_trt_runtime)].copy()
    clean_df["power_mode"] = clean_df["power_mode"].map(normalize_power_mode)
    clean_df["dashboard_source"] = "clean_v2_without_tensorrt"

    trt_frames = []
    for summary_path in sorted((trt_dir / "latency").glob("*_*_*/summary.csv")):
        df = pd.read_csv(summary_path)
        if df.empty or "runtime" not in df.columns:
            continue
        df = df[df["runtime"].map(is_trt_runtime)].copy()
        if df.empty:
            continue
        df["model"] = df.apply(lambda row: model_from_engine_or_dir(row, summary_path), axis=1)
        df["power_mode"] = df["power_mode"].map(normalize_power_mode)
        df["source_path"] = str(summary_path.parent)
        df["dashboard_source"] = "tensorrt_only"
        if "final_class" not in df.columns:
            df["final_class"] = "measured"
        if "status_reason" not in df.columns:
            df["status_reason"] = df.get("status", "ok")
        if "energy_per_inference_mj" not in df.columns:
            df["energy_per_inference_mj"] = pd.NA
        mask = (
            df["energy_per_inference_mj"].isna()
            & df.get("mean_ms", pd.Series(index=df.index)).notna()
            & df.get("power_avg_mw", pd.Series(index=df.index)).notna()
        )
        if mask.any():
            df.loc[mask, "energy_per_inference_mj"] = (
                pd.to_numeric(df.loc[mask, "mean_ms"], errors="coerce")
                * pd.to_numeric(df.loc[mask, "power_avg_mw"], errors="coerce")
                / 1000.0
            )
        trt_frames.append(df)

    frames = [clean_df] + trt_frames
    frames = [frame for frame in frames if not frame.empty]
    frames = align_columns(frames)
    combined = pd.concat(frames, ignore_index=True)
    combined["power_sort"] = combined["power_mode"].map(lambda x: POWER_ORDER.get(str(x), 99))
    combined["runtime_sort"] = combined["runtime"].map(lambda x: RUNTIME_ORDER.get(str(x), 99))
    combined = combined.sort_values(["model", "power_sort", "runtime_sort"]).drop(
        columns=["power_sort", "runtime_sort"]
    )

    out_path = output_dir / "consolidated_latency.csv"
    combined.to_csv(out_path, index=False)
    return combined


def build_accuracy(clean_dir: Path, output_dir: Path) -> pd.DataFrame:
    accuracy = read_csv_if_exists(clean_dir / "consolidated_accuracy.csv")
    if not accuracy.empty:
        accuracy.to_csv(output_dir / "consolidated_accuracy.csv", index=False)
    return accuracy


def load_layer_csvs(layer_dir: Path, include_trt: bool) -> list[pd.DataFrame]:
    frames = []
    for csv_path in sorted(layer_dir.glob("*.csv")):
        if csv_path.name.startswith("consolidated_"):
            continue
        match = re.match(r"(.+)_(pytorch_cpu|pytorch_cuda|tensorrt_fp32|tensorrt_fp16|tensorrt_int8|onnxrt_cuda|onnxrt_trt|tflite_cpu|tflite_gpu|ncnn_cpu|ncnn_vulkan)\.csv$", csv_path.name)
        if not match:
            continue
        runtime = match.group(2)
        if is_trt_runtime(runtime) != include_trt:
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["sequence"] = range(1, len(df) + 1)
        if "model" not in df.columns:
            df["model"] = match.group(1)
        if "runtime" not in df.columns:
            df["runtime"] = runtime
        df["dashboard_source"] = "tensorrt_only" if include_trt else "clean_v2_without_tensorrt"
        df["source_file"] = str(csv_path)
        frames.append(df)
    return frames


def build_layer_runtime(clean_dir: Path, trt_dir: Path, output_dir: Path) -> pd.DataFrame:
    clean_frames = load_layer_csvs(clean_dir / "layer_runtime", include_trt=False)
    trt_frames = load_layer_csvs(trt_dir / "detailed_layers" / "layer_runtime", include_trt=True)
    frames = clean_frames + trt_frames
    if not frames:
        raise FileNotFoundError("No layer runtime CSV files found")

    frames = align_columns(frames)
    combined = pd.concat(frames, ignore_index=True)
    if "rank" in combined.columns:
        combined["rank"] = pd.to_numeric(combined["rank"], errors="coerce")
    if "sequence" in combined.columns:
        combined["sequence"] = pd.to_numeric(combined["sequence"], errors="coerce")
    if "mean_ms" in combined.columns:
        combined["mean_ms"] = pd.to_numeric(combined["mean_ms"], errors="coerce")
    combined["runtime_sort"] = combined["runtime"].map(lambda x: RUNTIME_ORDER.get(str(x), 99))
    sort_cols = ["model", "runtime_sort", "sequence"]
    combined = combined.sort_values(sort_cols).drop(columns=["runtime_sort"])
    combined.to_csv(output_dir / "consolidated_layer_runtime.csv", index=False)

    split_dir = output_dir / "layer_runtime"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for (model, runtime), group in combined.groupby(["model", "runtime"], sort=False):
        group.to_csv(split_dir / f"{model}_{runtime}.csv", index=False)
    return combined


def clean_label(label: object, max_chars: int = 90) -> str:
    text = str(label)
    text = text.replace(" + ", "\n+ ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def plot_layer_runtime(layer_df: pd.DataFrame, output_dir: Path, top_n: int) -> int:
    image_count = 0
    required = {"model", "runtime", "layer_name", "mean_ms"}
    if not required.issubset(layer_df.columns):
        return image_count

    for (model, runtime), group in layer_df.groupby(["model", "runtime"], sort=False):
        group = group.copy()
        group["mean_ms"] = pd.to_numeric(group["mean_ms"], errors="coerce")
        group = group.dropna(subset=["mean_ms"])
        group = group[group["mean_ms"] > 0]
        if group.empty:
            continue

        if "sequence" in group.columns and group["sequence"].notna().any():
            plot_df = group.sort_values("sequence").head(top_n)
        else:
            plot_df = group.head(top_n)
        plot_df = plot_df.iloc[::-1].copy()
        plot_df["label"] = plot_df["layer_name"].map(clean_label)

        height = max(7.0, min(22.0, len(plot_df) * 0.34 + 2.5))
        fig, ax = plt.subplots(figsize=(14, height))
        color = "#31a354" if is_trt_runtime(runtime) else "#3182bd"
        ax.barh(plot_df["label"], plot_df["mean_ms"], color=color, alpha=0.88)
        ax.set_xlabel("Mean runtime per layer/op (ms)")
        ax.set_ylabel("")
        ax.set_title(f"{model} / {runtime} layer runtime")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)

        for y, value in enumerate(plot_df["mean_ms"]):
            ax.text(value, y, f" {value:.3f}", va="center", fontsize=8)

        fig.tight_layout()
        fig.savefig(output_dir / f"{model}_{runtime}.png", dpi=150)
        plt.close(fig)
        image_count += 1
    return image_count


def plot_framework_comparisons(latency_df: pd.DataFrame, output_dir: Path) -> int:
    required = {"model", "runtime", "power_mode", "mean_ms"}
    if not required.issubset(latency_df.columns):
        return 0

    metrics = [
        ("power_avg_mw", "Avg Power (mW)", "lower"),
        ("power_max_mw", "Max Power (mW)", "lower"),
        ("memory_mb", "Memory (MB)", "lower"),
        ("mean_ms", "Runtime Latency (ms)", "lower"),
    ]

    df = latency_df.copy()
    if "status" in df.columns:
        df = df[df["status"].fillna("ok") == "ok"]
    df["runtime_sort"] = df["runtime"].map(lambda x: RUNTIME_ORDER.get(str(x), 99))
    df = df.sort_values(["model", "runtime_sort", "power_mode"])

    for col, _, _ in metrics:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    image_count = 0
    for model, model_df in df.groupby("model", sort=False):
        runtimes = [
            rt
            for rt in sorted(
                model_df["runtime"].dropna().unique(),
                key=lambda rt: RUNTIME_ORDER.get(str(rt), 99),
            )
        ]
        if not runtimes:
            continue

        labels = [RUNTIME_LABELS.get(str(rt), str(rt)) for rt in runtimes]
        power_modes = [
            p
            for p in sorted(
                model_df["power_mode"].dropna().unique(),
                key=lambda p: POWER_ORDER.get(str(p), 99),
            )
        ]
        y = list(range(len(runtimes)))
        height = 0.36 if len(power_modes) > 1 else 0.55
        offsets = (
            [0.0]
            if len(power_modes) == 1
            else [(-height / 2), (height / 2)]
            if len(power_modes) == 2
            else [height * (i - (len(power_modes) - 1) / 2) for i in range(len(power_modes))]
        )

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        for ax, (metric_col, title, direction) in zip(axes, metrics):
            for idx, power in enumerate(power_modes):
                subset = model_df[model_df["power_mode"] == power]
                values = []
                for runtime in runtimes:
                    row = subset[subset["runtime"] == runtime]
                    values.append(float(row[metric_col].iloc[0]) if not row.empty and pd.notna(row[metric_col].iloc[0]) else 0.0)
                positions = [v + offsets[idx] for v in y]
                bars = ax.barh(positions, values, height=height, label=str(power), alpha=0.88)
                for bar, value in zip(bars, values):
                    if value <= 0:
                        continue
                    label = f"{value:.1f}" if value >= 10 else f"{value:.2f}"
                    ax.text(
                        bar.get_width(),
                        bar.get_y() + bar.get_height() / 2,
                        label,
                        ha="left",
                        va="center",
                        fontsize=7,
                    )

            ax.set_title(f"{title} ({direction} is better)")
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.25)
            ax.set_axisbelow(True)
            ax.legend(title="Power")
            xmax = ax.get_xlim()[1]
            ax.set_xlim(0, xmax * 1.18 if xmax > 0 else 1)

        fig.suptitle(f"{model} framework comparison", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output_dir / f"{model}_framework_comparison.png", dpi=150)
        plt.close(fig)
        image_count += 1

    return image_count


def write_manifest(output_dir: Path, clean_dir: Path, trt_dir: Path, counts: dict[str, int]) -> None:
    manifest = {
        "clean_source": str(clean_dir),
        "tensorrt_source": str(trt_dir),
        "output_dir": str(output_dir),
        "policy": {
            "clean_v2_excluded_runtimes": sorted(TRT_RUNTIMES),
            "tensorrt_only_included_runtimes": sorted(TRT_RUNTIMES),
        },
        "counts": counts,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-dir",
        default=str(project_dir / "my_results" / "clean_v2(without_tensorRT)"),
    )
    parser.add_argument(
        "--tensorrt-dir",
        default=str(project_dir / "my_results" / "tensorrt_only"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "my_results" / "visualizations"),
    )
    parser.add_argument("--top-n", type=int, default=45)
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    trt_dir = Path(args.tensorrt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latency = build_latency(clean_dir, trt_dir, output_dir)
    accuracy = build_accuracy(clean_dir, output_dir)
    layer_runtime = build_layer_runtime(clean_dir, trt_dir, output_dir)
    image_count = plot_layer_runtime(layer_runtime, output_dir, args.top_n)
    comparison_count = plot_framework_comparisons(latency, output_dir)

    counts = {
        "latency_rows": int(len(latency)),
        "accuracy_rows": int(len(accuracy)),
        "layer_runtime_rows": int(len(layer_runtime)),
        "layer_runtime_images": int(image_count),
        "framework_comparison_images": int(comparison_count),
    }
    write_manifest(output_dir, clean_dir, trt_dir, counts)

    print(f"[dashboard-data] output_dir={output_dir}")
    for key, value in counts.items():
        print(f"[dashboard-data] {key}={value}")


if __name__ == "__main__":
    main()
