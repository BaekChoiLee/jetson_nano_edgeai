#!/usr/bin/env python3
# =============================================================================
# generate_charts_v2.py — 6 모델 × 12 런타임 + 레이어 전력 차트 생성
# =============================================================================
# 역할: consolidate_v2.py 가 만든 통합 CSV/JSON 을 입력으로 받아 시각화 산출.
#       기존 generate_charts.py 와 충돌하지 않는 별도 진입점.
#
# 생성 차트:
#   1) latency_matrix_MAXN.png    — 6×12 히트맵 (mean ms)
#   2) latency_matrix_5W.png      — 6×12 히트맵 (mean ms)
#   3) accuracy_matrix.png        — 6×12 히트맵 (top1 또는 mAP@0.5)
#   4) layer_power_<model>.png    — 모델당 top-20 에너지 바 차트
#   5) coverage_summary.png       — 6×12 셀 상태(ok/error/missing) 그리드
#
# ⚠️ 차트 캡션:
#   - 모든 layer_power_<model>.png 차트에 "독립 실행 격리 전력 — in-context
#     추론과 차이 가능" 캡션을 명시.
#
# 사용법:
#   python analysis/generate_charts_v2.py \
#     --results-dir results/clean_v2_<TS> \
#     --output-dir results/clean_v2_<TS>/charts
# =============================================================================

"""Chart generator for 6 model × 12 runtime + layer power profiles."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict


CAVEAT_TEXT = (
    "⚠️ 독립 실행 격리 전력 (Layer Amplification Loop) — "
    "실제 추론 in-context 전력과 차이 가능"
)


def _lazy_matplotlib():
    """matplotlib 없으면 친절한 에러 메시지."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        return matplotlib, plt, np
    except ImportError as e:
        print(f"[ERROR] matplotlib 필요: pip install matplotlib numpy ({e})")
        sys.exit(1)


# =============================================================================
# 1. Latency 매트릭스 히트맵
# =============================================================================
def plot_latency_matrix(summary, output_path, power_mode="MAXN"):
    matplotlib, plt, np = _lazy_matplotlib()

    models = summary["models"]
    runtimes = summary["runtimes"]
    matrix = summary["matrix"]

    key = f"latency_{power_mode}_mean_ms"
    data = np.full((len(models), len(runtimes)), np.nan)
    for i, m in enumerate(models):
        for j, r in enumerate(runtimes):
            v = matrix.get(m, {}).get(r, {}).get(key)
            if v is not None:
                data[i, j] = v

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(runtimes)))
    ax.set_xticklabels(runtimes, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title(f"Latency (mean ms) — {power_mode}")

    # 셀에 값 표시
    for i in range(len(models)):
        for j in range(len(runtimes)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v > np.nanmean(data) else "black",
                        fontsize=8)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        color="red", fontsize=7)

    plt.colorbar(im, ax=ax, label="latency (ms)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {output_path}")


# =============================================================================
# 2. Accuracy 매트릭스 히트맵
# =============================================================================
def plot_accuracy_matrix(summary, output_path):
    matplotlib, plt, np = _lazy_matplotlib()

    models = summary["models"]
    runtimes = summary["runtimes"]
    matrix = summary["matrix"]

    # 분류는 top1, 검출은 map_50 사용
    data = np.full((len(models), len(runtimes)), np.nan)
    is_detection = []
    for i, m in enumerate(models):
        det = "yolov8n" in m or "ssd_mobilenet_v2" in m
        is_detection.append(det)
        for j, r in enumerate(runtimes):
            cell = matrix.get(m, {}).get(r, {})
            v = cell.get("accuracy_map_50") if det else cell.get("accuracy_top1")
            if v is not None:
                data[i, j] = v

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(runtimes)))
    ax.set_xticklabels(runtimes, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    labels = [
        f"{m} (mAP@.5)" if det else f"{m} (top-1)"
        for m, det in zip(models, is_detection)
    ]
    ax.set_yticklabels(labels)
    ax.set_title("Accuracy Matrix (top-1 for classification, mAP@0.5 for detection)")

    for i in range(len(models)):
        for j in range(len(runtimes)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="black", fontsize=8)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        color="gray", fontsize=7)

    plt.colorbar(im, ax=ax, label="accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {output_path}")


# =============================================================================
# 3. 레이어 전력 바 차트 (top-20 에너지)
# =============================================================================
def plot_layer_power(layer_csv, output_path, model_name, top_n=20):
    matplotlib, plt, np = _lazy_matplotlib()

    if not os.path.exists(layer_csv):
        print(f"  [skip] {layer_csv} not found")
        return

    rows = []
    with open(layer_csv, "r") as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["energy_per_run_mj"] = float(r.get("energy_per_run_mj") or 0.0)
                r["avg_power_mw"] = float(r.get("avg_power_mw") or 0.0)
                r["avg_latency_ms"] = float(r.get("avg_latency_ms") or 0.0)
                rows.append(r)
            except (TypeError, ValueError):
                continue

    if not rows:
        print(f"  [skip] no valid rows in {layer_csv}")
        return

    rows.sort(key=lambda x: x["energy_per_run_mj"], reverse=True)
    top = rows[:top_n]
    names = [r.get("layer_name", "?")[:35] for r in top]
    energies = [r["energy_per_run_mj"] for r in top]
    powers = [r["avg_power_mw"] for r in top]
    latencies = [r["avg_latency_ms"] for r in top]

    fig, ax1 = plt.subplots(figsize=(12, max(6, top_n * 0.35)))
    y_pos = np.arange(len(names))
    bars = ax1.barh(y_pos, energies, color="tab:orange", alpha=0.8,
                     label="energy/run (mJ)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("energy per run (mJ)")
    ax1.set_title(f"{model_name} — Top {top_n} layers by energy")

    # 보조 축: latency
    ax2 = ax1.twiny()
    ax2.plot(latencies, y_pos, "o-", color="tab:blue", label="latency (ms)")
    ax2.set_xlabel("latency (ms)")

    # 캡션 (caveat)
    fig.text(0.5, -0.02, CAVEAT_TEXT, ha="center", fontsize=8,
             color="darkred", style="italic")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {output_path}")


# =============================================================================
# 4. Coverage 그리드 (ok/error/missing)
# =============================================================================
def plot_coverage(summary, output_path):
    matplotlib, plt, np = _lazy_matplotlib()

    models = summary["models"]
    runtimes = summary["runtimes"]
    matrix = summary["matrix"]

    # 0=missing, 1=error, 2=ok
    status_map = {"ok": 2, "error": 1, "missing": 0}
    data = np.zeros((len(models), len(runtimes)))
    for i, m in enumerate(models):
        for j, r in enumerate(runtimes):
            data[i, j] = status_map.get(
                matrix.get(m, {}).get(r, {}).get("status", "missing"), 0
            )

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = matplotlib.colors.ListedColormap(
        ["lightgray", "tomato", "mediumseagreen"]
    )
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(range(len(runtimes)))
    ax.set_xticklabels(runtimes, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title(
        f"Coverage — ok: {summary['summary']['ok']}, "
        f"error: {summary['summary']['error']}, "
        f"missing: {summary['summary']['missing']} "
        f"(coverage {summary['summary']['coverage_pct']}%)"
    )

    for i in range(len(models)):
        for j in range(len(runtimes)):
            v = data[i, j]
            label = {0: "—", 1: "✗", 2: "✓"}.get(v, "?")
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if v == 1 else "black",
                    fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {output_path}")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate 6×12 matrix charts + layer power profiles"
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.results_dir, "charts")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(args.results_dir, "consolidated_summary.json")
    if not os.path.exists(summary_path):
        print(f"[ERROR] {summary_path} 없음. consolidate_v2.py 를 먼저 실행하세요.")
        sys.exit(1)

    with open(summary_path, "r") as f:
        summary = json.load(f)

    print(f"[1/5] latency matrix MAXN...")
    plot_latency_matrix(
        summary, os.path.join(out_dir, "latency_matrix_MAXN.png"), "MAXN"
    )

    print(f"[2/5] latency matrix 5W...")
    plot_latency_matrix(
        summary, os.path.join(out_dir, "latency_matrix_5W.png"), "5W"
    )

    print(f"[3/5] accuracy matrix...")
    plot_accuracy_matrix(
        summary, os.path.join(out_dir, "accuracy_matrix.png")
    )

    print(f"[4/5] layer power per-model...")
    layer_dir = os.path.join(args.results_dir, "layer_power")
    for model in summary["models"]:
        csv_path = os.path.join(layer_dir, f"{model}_layer_power.csv")
        out_png = os.path.join(out_dir, f"layer_power_{model}.png")
        plot_layer_power(csv_path, out_png, model, top_n=20)

    print(f"[5/5] coverage grid...")
    plot_coverage(
        summary, os.path.join(out_dir, "coverage_summary.png")
    )

    print(f"\n[DONE] charts saved to {out_dir}")
    print(f"⚠️  레이어 전력 차트 캡션: {CAVEAT_TEXT}")


if __name__ == "__main__":
    main()
