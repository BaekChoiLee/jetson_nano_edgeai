#!/usr/bin/env python3
# =============================================================================
# consolidate_v2.py — 6 모델 × 11 런타임 결과 통합 (신규)
# =============================================================================
# 역할: rebenchmark_clean_v2.sh 가 생성한 결과 트리를 읽어
#       (지연 + 정확도 + 레이어 런타임/전력) 통합 CSV/JSON 매트릭스를 만든다.
#       기존 analysis/consolidate_results.py 와 충돌하지 않도록 별도 진입점.
#
# 입력 트리:
#   results/clean_v2_<TS>/
#     latency/{MAXN,5W}/<model>_<runtime>.json
#     accuracy/<model>_classification.json   (분류 모델, 11 runtimes)
#     accuracy/<model>_detection.json        (검출 모델, 11 runtimes)
#     layer_runtime/<model>_<runtime>.csv
#     layer_runtime/<model>_<runtime>.json
#     layer_power/<model>_layer_power.csv   (legacy)
#     layer_power/<model>_layer_power.json  (legacy)
#
# 출력:
#   consolidated_latency.csv   — model × runtime × power_mode 매트릭스
#   consolidated_accuracy.csv  — model × runtime × metric 매트릭스
#   consolidated_layer_runtime.csv — runtime별 layer latency 로우데이터
#   consolidated_summary.json      — 6×11 풀 매트릭스 (NA 셀 표시)
# =============================================================================

"""Consolidate 6 model × 11 runtime results into unified CSV/JSON."""

import argparse
import csv
import json
import os
import sys
from glob import glob


CLASSIFICATION_MODELS = [
    "mobilenetv3_small",
    "resnet50",
    "efficientnet_b0",
    "shufflenet_v2_x1_0",
]

DETECTION_MODELS = [
    "yolov8n",
    "ssd_mobilenet_v2",
]

ALL_MODELS = CLASSIFICATION_MODELS + DETECTION_MODELS

ALL_RUNTIMES = [
    "pytorch_cpu", "pytorch_cuda",
    "tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8",
    "onnxrt_cuda", "onnxrt_trt",
    "tflite_cpu", "tflite_gpu",
    "ncnn_cpu", "ncnn_vulkan",
]

POWER_MODES = ["MAXN", "5W"]
CARBON_INTENSITY_G_PER_KWH = 459.0
STATUS_PRIORITY = {
    "missing": 0,
    "measured": 1,
    "na": 2,
    "invalid": 3,
    "error": 4,
}


def _safe_load_json(path):
    """JSON 로드 실패 시 None 반환."""
    if not os.path.exists(path):
        return None

    # benchmark.py를 --output-dir <...>.json 형태로 호출하면
    # <...>.json/ 내부에 타임스탬프 디렉토리와 results.json이 생긴다.
    # 이 경우 최신 results.json을 자동으로 찾아 로드한다.
    if os.path.isdir(path):
        candidates = sorted(glob(os.path.join(path, "**", "results.json"), recursive=True))
        if not candidates:
            return []
        path = candidates[-1]

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [warn] failed to read {path}: {e}")
        return {"error": f"json_read_failed: {e}"}


def _status_rank(status):
    return STATUS_PRIORITY.get(status or "missing", 0)


def _combine_status(current, incoming):
    if _status_rank(incoming) > _status_rank(current):
        return incoming
    return current


def _classify_error_message(error_message):
    msg = (error_message or "").strip()
    low = msg.lower()
    if not msg:
        return "error", "runtime_error"
    if "n/a:" in low:
        return "na", "explicit_na"
    if "flexsize" in low or "select tensorflow op" in low or "flex delegate" in low:
        return "na", "tflite_flex_required"
    if "unsupported/unstable" in low or "unsupported" in low:
        return "na", "runtime_unsupported"
    if "not found" in low or "file missing" in low or "no such file" in low:
        return "na", "artifact_missing"
    return "error", "runtime_error"


def _load_latency_payload(latency_dir, model, runtime):
    candidates = [
        os.path.join(latency_dir, f"{model}_{runtime}"),
        os.path.join(latency_dir, f"{model}_{runtime}.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return _safe_load_json(path), path
    return None, candidates[0]


def _classify_latency_payload(data):
    if data is None:
        return "na", "file_missing", {}, "file missing"

    if isinstance(data, list):
        if len(data) == 0:
            return "invalid", "empty_results_list", {}, ""
        data = data[0]

    if not isinstance(data, dict):
        return "invalid", "invalid_payload_type", {}, ""

    error_msg = (data.get("error") or "").strip()
    if error_msg:
        status, reason = _classify_error_message(error_msg)
        return status, reason, data, error_msg

    latency = data.get("latency", {})
    mean_ms = _round(latency.get("mean")) if isinstance(latency, dict) else None
    if mean_ms is None:
        return "invalid", "missing_latency", data, ""
    if mean_ms <= 0:
        return "invalid", "nonpositive_latency", data, ""
    return "measured", "ok", data, ""


def _classify_accuracy_record(task, rec):
    explicit = (rec.get("status") or "").strip().lower()
    if explicit in ("measured", "invalid", "error", "na"):
        return explicit, rec.get("status_reason", "from_record_status")

    error_msg = (rec.get("error") or "").strip()
    if error_msg:
        return _classify_error_message(error_msg)

    if task == "classification":
        top1 = _round(rec.get("top1") or rec.get("top_1"))
        top5 = _round(rec.get("top5") or rec.get("top_5"))
        n = rec.get("n")
        try:
            n = int(n) if n is not None else None
        except (TypeError, ValueError):
            n = None
        if top1 is None or top5 is None:
            return "invalid", "missing_topk"
        if top5 < top1:
            return "invalid", "top5_lt_top1"
        if n is not None and n >= 50 and top1 <= 0.2 and top5 <= 0.2:
            return "invalid", "suspicious_dummy_accuracy"
        return "measured", "ok"

    map_50 = _round(rec.get("map_50"))
    map_50_95 = _round(rec.get("map_50_95"))
    n = rec.get("n")
    try:
        n = int(n) if n is not None else None
    except (TypeError, ValueError):
        n = None
    if map_50 is None or map_50_95 is None:
        return "invalid", "missing_map"
    if n is not None and n >= 50 and map_50 == 0.0 and map_50_95 == 0.0:
        return "invalid", "zero_map_with_samples"
    return "measured", "ok"


def consolidate_latency(results_dir, output_csv):
    """latency 결과를 단일 CSV 로 합친다.

    각 행: model, runtime, power_mode, mean_ms, std_ms, p50, p95, p99, model_size_mb
    """
    rows = []
    for power_mode in POWER_MODES:
        latency_dir = os.path.join(results_dir, "latency", power_mode)
        for model in ALL_MODELS:
            for runtime in ALL_RUNTIMES:
                data, src_path = _load_latency_payload(latency_dir, model, runtime)
                final_class, status_reason, payload, payload_error = _classify_latency_payload(data)
                lat = payload.get("latency", {}) if isinstance(payload, dict) else {}
                tegra = payload.get("tegrastats", {}) if isinstance(payload, dict) else {}
                power_in = tegra.get("pom_5v_in_current_mw", {}) if isinstance(tegra, dict) else {}
                power_avg_mw = _round(power_in.get("mean"))
                power_max_mw = _round(power_in.get("max"))
                mean_ms = _round(lat.get("mean"))
                energy_per_mj = None
                co2_per_mg = None
                total_energy_mj = None
                total_co2_mg = None
                num_runs = payload.get("num_runs") if isinstance(payload, dict) else None
                if power_avg_mw is not None and mean_ms is not None and mean_ms > 0:
                    # mW * ms / 1000 = mJ
                    energy_per_mj = (power_avg_mw * mean_ms) / 1000.0
                    # mgCO2 = Wh * gCO2/kWh = (mJ/3_600_000) * carbon_intensity
                    co2_per_mg = (energy_per_mj / 3_600_000.0) * CARBON_INTENSITY_G_PER_KWH
                    if isinstance(num_runs, int) and num_runs > 0:
                        total_energy_mj = energy_per_mj * num_runs
                        total_co2_mg = co2_per_mg * num_runs
                rows.append({
                    "model": model, "runtime": runtime,
                    "power_mode": power_mode,
                    "status": "ok" if final_class == "measured" else final_class,
                    "final_class": final_class,
                    "status_reason": status_reason,
                    "source_path": src_path,
                    "mean_ms": mean_ms,
                    "std_ms":  _round(lat.get("std")),
                    "p50_ms":  _round(lat.get("p50")),
                    "p95_ms":  _round(lat.get("p95")),
                    "p99_ms":  _round(lat.get("p99")),
                    "memory_mb": _round(payload.get("memory_mb")) if isinstance(payload, dict) else None,
                    "model_size_mb": _round(payload.get("model_size_mb")) if isinstance(payload, dict) else None,
                    "power_avg_mw": power_avg_mw,
                    "power_max_mw": power_max_mw,
                    "energy_per_inference_mj": _round(energy_per_mj, 6),
                    "co2_per_inference_mg": _round(co2_per_mg, 6),
                    "total_energy_mj": _round(total_energy_mj, 4),
                    "total_co2_mg": _round(total_co2_mg, 4),
                    "num_runs": num_runs,
                    "error": payload_error,
                })

    _write_csv(output_csv, rows)
    return rows


def consolidate_accuracy(results_dir, output_csv):
    """accuracy 결과를 단일 CSV 로 합친다.

    분류: top1, top5
    검출: map_50, map_50_95
    """
    rows = []
    accuracy_dir = os.path.join(results_dir, "accuracy")

    # 분류
    for model in CLASSIFICATION_MODELS:
        path = os.path.join(accuracy_dir, f"{model}_classification.json")
        data = _safe_load_json(path)
        # accuracy_eval.py 가 list[dict] 또는 dict 어느 쪽이든 받을 수 있게
        records = _normalize_records(data, model)
        for rec in records:
            rt = rec.get("runtime", "unknown")
            final_class, status_reason = _classify_accuracy_record("classification", rec)
            rows.append({
                "model": model, "runtime": rt,
                "task": "classification",
                "top1": _round(rec.get("top1") or rec.get("top_1")),
                "top5": _round(rec.get("top5") or rec.get("top_5")),
                "map_50": None, "map_50_95": None,
                "n": rec.get("n"),
                "error": rec.get("error", ""),
                "final_class": final_class,
                "status_reason": status_reason,
            })

    # 검출
    for model in DETECTION_MODELS:
        path = os.path.join(accuracy_dir, f"{model}_detection.json")
        data = _safe_load_json(path)
        records = _normalize_records(data, model)
        for rec in records:
            rt = rec.get("runtime", "unknown")
            final_class, status_reason = _classify_accuracy_record("detection", rec)
            rows.append({
                "model": model, "runtime": rt,
                "task": "detection",
                "top1": None, "top5": None,
                "map_50":    _round(rec.get("map_50")),
                "map_50_95": _round(rec.get("map_50_95")),
                "n": rec.get("n"),
                "error": rec.get("error", ""),
                "final_class": final_class,
                "status_reason": status_reason,
            })

    _write_csv(output_csv, rows)
    return rows


def consolidate_layer_power(results_dir, output_csv):
    """레이어 전력 CSV 들을 하나로 합치고 모델 컬럼을 추가."""
    layer_dir = os.path.join(results_dir, "layer_power")
    rows = []
    for model in ALL_MODELS:
        csv_path = os.path.join(layer_dir, f"{model}_layer_power.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r") as f:
            # 첫 줄 caveat 주석 스킵
            first = f.readline()
            if not first.startswith("#"):
                f.seek(0)
            reader = csv.DictReader(f)
            for row in reader:
                row["model"] = model
                rows.append(row)
    if rows:
        _write_csv(output_csv, rows)
    return rows


def consolidate_layer_runtime(results_dir, output_csv):
    """런타임별 레이어 CSV/JSON 결과를 하나로 합친다.

    CSV가 있는 경우: 레이어 행을 모두 수집
    CSV가 비어있거나 없는 경우: JSON status/error를 1행으로 기록
    """
    layer_dir = os.path.join(results_dir, "layer_runtime")
    rows = []
    if not os.path.isdir(layer_dir):
        return rows

    for model in ALL_MODELS:
        for runtime in ALL_RUNTIMES:
            csv_path = os.path.join(layer_dir, f"{model}_{runtime}.csv")
            json_path = os.path.join(layer_dir, f"{model}_{runtime}.json")
            meta = _safe_load_json(json_path) or {}

            appended_layer_rows = 0
            if os.path.exists(csv_path):
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row["model"] = model
                        row["runtime"] = runtime
                        row["status"] = meta.get("status", "ok")
                        if "error" not in row:
                            row["error"] = meta.get("error", "")
                        if "backend" not in row or not row.get("backend"):
                            row["backend"] = meta.get("backend", "")
                        if "granularity" not in row or not row.get("granularity"):
                            row["granularity"] = meta.get("granularity", "")
                        rows.append(row)
                        appended_layer_rows += 1

            if appended_layer_rows == 0:
                rows.append({
                    "model": model,
                    "runtime": runtime,
                    "layer_name": "",
                    "backend": meta.get("backend", ""),
                    "granularity": meta.get("granularity", ""),
                    "mean_ms": None,
                    "std_ms": None,
                    "min_ms": None,
                    "max_ms": None,
                    "samples": 0,
                    "status": meta.get("status", "missing"),
                    "error": meta.get("error", "file missing"),
                })

    _write_csv(output_csv, rows)
    return rows


def consolidate_carbon(latency_rows, output_csv):
    """latency rows에서 탄소/에너지 행만 추출."""
    rows = []
    for r in latency_rows:
        if r.get("final_class") != "measured" and r.get("status") != "ok":
            continue
        if r.get("energy_per_inference_mj") is None and r.get("co2_per_inference_mg") is None:
            continue
        rows.append({
            "model": r.get("model"),
            "runtime": r.get("runtime"),
            "power_mode": r.get("power_mode"),
            "num_runs": r.get("num_runs"),
            "latency_mean_ms": r.get("mean_ms"),
            "power_avg_mw": r.get("power_avg_mw"),
            "energy_per_inference_mj": r.get("energy_per_inference_mj"),
            "co2_per_inference_mg": r.get("co2_per_inference_mg"),
            "total_energy_mj": r.get("total_energy_mj"),
            "total_co2_mg": r.get("total_co2_mg"),
        })
    _write_csv(output_csv, rows)
    return rows


def build_summary_matrix(latency_rows, accuracy_rows):
    """6×12 매트릭스 풀 셀 표 (JSON)."""
    lat_index = {}
    for r in latency_rows:
        lat_index[(r.get("model"), r.get("runtime"), r.get("power_mode"))] = r

    acc_index = {}
    for r in accuracy_rows:
        key = (r.get("model"), r.get("runtime"))
        prev = acc_index.get(key)
        if prev is None:
            acc_index[key] = r
            continue
        prev_class = prev.get("final_class", "missing")
        cur_class = r.get("final_class", "missing")
        if _status_rank(cur_class) > _status_rank(prev_class):
            acc_index[key] = r

    matrix = {}
    for model in ALL_MODELS:
        matrix[model] = {}
        for runtime in ALL_RUNTIMES:
            cell = {
                "latency_MAXN_mean_ms": None,
                "latency_5W_mean_ms": None,
                "latency_MAXN_status": "missing",
                "latency_5W_status": "missing",
                "latency_MAXN_reason": "no_latency_rows",
                "latency_5W_reason": "no_latency_rows",
                "accuracy_top1": None,
                "accuracy_top5": None,
                "accuracy_map_50": None,
                "accuracy_map_50_95": None,
                "accuracy_status": "missing",
                "accuracy_reason": "no_accuracy_row",
                "status": "missing",
                "status_reason": "no_latency_rows",
            }

            maxn_row = lat_index.get((model, runtime, "MAXN"))
            if maxn_row:
                cell["latency_MAXN_mean_ms"] = maxn_row.get("mean_ms")
                cell["latency_MAXN_status"] = maxn_row.get("final_class", "missing")
                cell["latency_MAXN_reason"] = maxn_row.get("status_reason", "")

            w5_row = lat_index.get((model, runtime, "5W"))
            if w5_row:
                cell["latency_5W_mean_ms"] = w5_row.get("mean_ms")
                cell["latency_5W_status"] = w5_row.get("final_class", "missing")
                cell["latency_5W_reason"] = w5_row.get("status_reason", "")

            mode_statuses = [cell["latency_MAXN_status"], cell["latency_5W_status"]]
            if "error" in mode_statuses:
                cell["status"] = "error"
                cell["status_reason"] = (
                    cell["latency_MAXN_reason"]
                    if cell["latency_MAXN_status"] == "error"
                    else cell["latency_5W_reason"]
                )
            elif "invalid" in mode_statuses:
                cell["status"] = "invalid"
                cell["status_reason"] = (
                    cell["latency_MAXN_reason"]
                    if cell["latency_MAXN_status"] == "invalid"
                    else cell["latency_5W_reason"]
                )
            elif mode_statuses[0] == "measured" and mode_statuses[1] == "measured":
                cell["status"] = "measured"
                cell["status_reason"] = "both_power_modes_measured"
            elif "measured" in mode_statuses and ("na" in mode_statuses or "missing" in mode_statuses):
                cell["status"] = "invalid"
                cell["status_reason"] = "partial_power_mode_coverage"
            elif "na" in mode_statuses:
                cell["status"] = "na"
                cell["status_reason"] = (
                    cell["latency_MAXN_reason"]
                    if cell["latency_MAXN_status"] == "na"
                    else cell["latency_5W_reason"]
                )

            acc = acc_index.get((model, runtime))
            if acc:
                cell["accuracy_top1"] = acc.get("top1")
                cell["accuracy_top5"] = acc.get("top5")
                cell["accuracy_map_50"] = acc.get("map_50")
                cell["accuracy_map_50_95"] = acc.get("map_50_95")
                cell["accuracy_status"] = acc.get("final_class", "missing")
                cell["accuracy_reason"] = acc.get("status_reason", "")

                # latency가 측정됐더라도 accuracy가 명확히 invalid/error이면 셀도 강등한다.
                if cell["status"] == "measured" and cell["accuracy_status"] in ("error", "invalid"):
                    cell["status"] = cell["accuracy_status"]
                    cell["status_reason"] = f"accuracy_{cell['accuracy_reason']}"

            cell["final_class"] = cell["status"]
            matrix[model][runtime] = cell

    # 요약 통계
    n_total = len(ALL_MODELS) * len(ALL_RUNTIMES)
    n_measured = sum(
        1 for m in ALL_MODELS for r in ALL_RUNTIMES
        if matrix[m][r]["status"] == "measured"
    )
    n_error = sum(
        1 for m in ALL_MODELS for r in ALL_RUNTIMES
        if matrix[m][r]["status"] == "error"
    )
    n_invalid = sum(
        1 for m in ALL_MODELS for r in ALL_RUNTIMES
        if matrix[m][r]["status"] == "invalid"
    )
    n_na = sum(
        1 for m in ALL_MODELS for r in ALL_RUNTIMES
        if matrix[m][r]["status"] == "na"
    )
    n_missing = n_total - n_measured - n_error - n_invalid - n_na

    return {
        "models": ALL_MODELS,
        "runtimes": ALL_RUNTIMES,
        "matrix": matrix,
        "summary": {
            "total_cells": n_total,
            "measured": n_measured,
            "ok": n_measured,
            "error": n_error,
            "invalid": n_invalid,
            "na": n_na,
            "missing": n_missing,
            "coverage_pct": round(n_measured / n_total * 100, 1) if n_total else 0.0,
        },
    }


# =============================================================================
# 헬퍼
# =============================================================================
def _normalize_records(data, model):
    """JSON 파일이 dict 인지 list 인지 모를 때 안전하게 list[dict] 로 변환."""
    if data is None:
        return [{"runtime": rt, "error": "file missing"} for rt in ALL_RUNTIMES]
    if isinstance(data, list):
        if len(data) == 0:
            return [{"runtime": rt, "error": "empty results"} for rt in ALL_RUNTIMES]
        return data
    if isinstance(data, dict):
        # accuracy_eval.py 가 dict 형태로 결과를 넣을 수도 있음
        # {"runtime_x": {...}, "runtime_y": {...}} 케이스
        if all(isinstance(v, dict) for v in data.values()):
            return [{"runtime": k, **v} for k, v in data.items()]
        return [data]
    return []


def _round(v, ndigits=4):
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None


def _write_csv(path, rows):
    if not rows:
        print(f"  [warn] no rows for {path}")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  [saved] {path} ({len(rows)} rows)")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Consolidate 6 model × 12 runtime benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="results/clean_v2_<TS> 경로",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="출력 디렉토리 (default: results-dir 아래)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/6] consolidate latency...")
    lat_rows = consolidate_latency(
        args.results_dir,
        os.path.join(out_dir, "consolidated_latency.csv"),
    )

    print(f"[2/6] consolidate accuracy...")
    acc_rows = consolidate_accuracy(
        args.results_dir,
        os.path.join(out_dir, "consolidated_accuracy.csv"),
    )

    print(f"[3/6] consolidate layer runtime...")
    consolidate_layer_runtime(
        args.results_dir,
        os.path.join(out_dir, "consolidated_layer_runtime.csv"),
    )

    print(f"[4/6] consolidate carbon...")
    consolidate_carbon(
        lat_rows,
        os.path.join(out_dir, "consolidated_carbon.csv"),
    )

    print(f"[5/6] consolidate layer power (legacy)...")
    consolidate_layer_power(
        args.results_dir,
        os.path.join(out_dir, "consolidated_layer_power.csv"),
    )

    print(f"[6/6] build summary matrix...")
    summary = build_summary_matrix(lat_rows, acc_rows)
    summary_path = os.path.join(out_dir, "consolidated_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [saved] {summary_path}")

    print()
    print("=" * 60)
    print(f"6 × 12 = {summary['summary']['total_cells']} cells:")
    print(f"  measured:{summary['summary']['measured']} "
          f"({summary['summary']['coverage_pct']}%)")
    print(f"  error:   {summary['summary']['error']}")
    print(f"  invalid: {summary['summary']['invalid']}")
    print(f"  na:      {summary['summary']['na']}")
    print(f"  missing: {summary['summary']['missing']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
