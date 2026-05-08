#!/bin/bash
# =============================================================================
# rebenchmark_clean.sh — Jetson Nano 전체 재벤치마크 (순차 실행)
# =============================================================================
# 목적: 모든 런타임을 개별 프로세스로, swap/cache 초기화 후 순차 실행
#       → 열/메모리 오염 없는 순도 100% 데이터 수집
#
# 사용법:
#   bash rebenchmark_clean.sh              # 전체 실행
#   bash rebenchmark_clean.sh --dry-run    # 계획만 출력
#
# 예상 소요: ~4-5시간
# =============================================================================

set -uo pipefail

# --- 설정 ---
BENCH_DIR="$HOME/jetson-benchmark"
BENCH_PY="$BENCH_DIR/benchmark/benchmark.py"
ACCURACY_PY="$BENCH_DIR/benchmark/accuracy_eval.py"
LAYER_PY="$BENCH_DIR/benchmark/layer_analyzer.py"
FAILED_PY="$BENCH_DIR/run_failed_runtimes.py"  # ncnn_vulkan_fixed 전용

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="$BENCH_DIR/results/clean_${TIMESTAMP}"
LATENCY_DIR="${OUTPUT_BASE}/latency"
ACCURACY_DIR="${OUTPUT_BASE}/accuracy"
LAYER_DIR="${OUTPUT_BASE}/layers"
LOG_FILE="${OUTPUT_BASE}/rebenchmark.log"

MODELS=("mobilenetv3_small" "resnet50")
POWER_MODES=("10w" "5w")

# 런타임 순서: 가벼운 것 → 무거운 것
RUNTIMES=(
    "tensorrt_fp16"
    "tensorrt_fp32"
    "tensorrt_int8"
    "ncnn_cpu"
    "ncnn_vulkan"
    "onnxrt_cuda"
    "onnxrt_trt"
    "tflite_cpu"
    "tflite_gpu"
    "pytorch_cuda"
    "pytorch_cpu"
)

COOLDOWN_SEC=0
NUM_WARMUP=10
NUM_RUNS=50
STD_THRESHOLD=10

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY-RUN] 실제 실행 없이 계획만 출력합니다."
fi

# --- 유틸리티 함수 ---

log() {
    local msg="[$(date '+%H:%M:%S')] $1"
    echo "$msg"
    if [[ "$DRY_RUN" == false ]]; then
        mkdir -p "$(dirname "$LOG_FILE")"
        echo "$msg" >> "$LOG_FILE"
    fi
}

clean_system() {
    log "  시스템 초기화: swap 재설정 + 캐시 드롭"
    if [[ "$DRY_RUN" == false ]]; then
        sudo swapoff -a 2>/dev/null || true
        sudo swapon -a 2>/dev/null || true
        sync
        echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
        sleep 5
    fi
}

set_power_mode() {
    local mode=$1
    log "  전력 모드 설정: $mode"
    if [[ "$DRY_RUN" == false ]]; then
        if [[ "$mode" == "5w" ]]; then
            sudo nvpmodel -m 1
        else
            sudo nvpmodel -m 0
        fi
        sleep 3
    fi
}

check_system_state() {
    if [[ "$DRY_RUN" == true ]]; then return 0; fi
    local swap_used
    swap_used=$(free -m | awk '/Swap:/ {print $3}')
    local ram_free
    ram_free=$(free -m | awk '/Mem:/ {print $4}')
    local temp_cpu
    temp_cpu=$(cat /sys/devices/virtual/thermal/thermal_zone1/temp 2>/dev/null || echo "0")
    temp_cpu=$((temp_cpu / 1000))
    log "    RAM free: ${ram_free}MB | Swap: ${swap_used}MB | CPU: ${temp_cpu}C"

    if [[ "$swap_used" -gt 50 ]]; then
        log "    [WARN] Swap ${swap_used}MB > 50MB — 재초기화"
        clean_system
    fi
    if [[ "$temp_cpu" -gt 55 ]]; then
        log "    [WARN] CPU ${temp_cpu}C > 55C — 추가 대기 60초"
        sleep 60
    fi
}

check_quality() {
    local result_dir=$1
    local runtime=$2
    if [[ "$DRY_RUN" == true ]]; then return 0; fi

    # 가장 최근 생성된 results.json 찾기 (최신 타임스탬프 폴더)
    local json_file
    json_file=$(find "$result_dir" -name "results.json" -type f -newer "$LOG_FILE" 2>/dev/null | tail -1)
    if [[ -z "$json_file" ]]; then
        json_file=$(find "$result_dir" -name "results.json" -type f 2>/dev/null | sort | tail -1)
    fi
    if [[ -z "$json_file" ]]; then
        log "    [QUALITY] results.json 없음 — 스킵"
        return 0
    fi

    python3 -c "
import json, sys
try:
    with open('${json_file}') as f:
        data = json.load(f)
    if isinstance(data, dict):
        for rt_name, rt_data in data.items():
            avg = rt_data.get('avg_ms', 0) or 0
            std = rt_data.get('std_ms', 0) or 0
            if avg > 0:
                pct = std / avg * 100
                status = 'OK' if pct <= ${STD_THRESHOLD} else 'WARN'
                print(f'{status}: {rt_name} avg={avg:.2f}ms std={pct:.1f}%')
    elif isinstance(data, list):
        for item in data:
            lat = item.get('latency', {})
            avg = lat.get('mean', 0) or 0
            std = lat.get('std', 0) or 0
            if avg > 0:
                pct = std / avg * 100
                status = 'OK' if pct <= ${STD_THRESHOLD} else 'WARN'
                print(f'{status}: {item[\"runtime\"]} avg={avg:.2f}ms std={pct:.1f}%')
except Exception as e:
    print(f'QUALITY-ERROR: {e}')
" 2>&1 | while read -r line; do log "    $line"; done || true
}

run_single_benchmark() {
    local model=$1
    local power=$2
    local runtime=$3
    local count=$4
    local total=$5

    log ""
    log "[$count/$total] $model / $power / $runtime"
    log "--- [$model] [$power] [$runtime] ---"

    check_system_state
    clean_system

    log "  쿨다운 ${COOLDOWN_SEC}초..."
    if [[ "$DRY_RUN" == false ]]; then
        sleep "$COOLDOWN_SEC"
    fi

    log "  벤치마크 시작"
    if [[ "$DRY_RUN" == false ]]; then
        python3 "$BENCH_PY" \
            --model "$model" \
            --runtimes "$runtime" \
            --power-mode "$power" \
            --num-warmup "$NUM_WARMUP" \
            --num-runs "$NUM_RUNS" \
            --cool-down 5 \
            --model-dir "$BENCH_DIR/models" \
            --output-dir "$LATENCY_DIR" \
            2>&1 | tee -a "$LOG_FILE"

        check_quality "$LATENCY_DIR" "$runtime"
    fi

    log "  완료: $runtime"
}

run_ncnn_vulkan_fixed() {
    local model=$1
    local power=$2
    local count=$3
    local total=$4

    log ""
    log "[$count/$total] $model / $power / ncnn_vulkan_fixed"
    log "--- [$model] [$power] [ncnn_vulkan_fixed] (run_failed_runtimes.py) ---"

    check_system_state
    clean_system

    log "  쿨다운 ${COOLDOWN_SEC}초..."
    if [[ "$DRY_RUN" == false ]]; then
        sleep "$COOLDOWN_SEC"
    fi

    log "  ncnn_vulkan_fixed 시작 (create_gpu_instance + fp16)"
    if [[ "$DRY_RUN" == false ]]; then
        python3 "$FAILED_PY" \
            --model "$model" \
            --power-mode "$power" \
            --num-warmup "$NUM_WARMUP" \
            --num-runs "$NUM_RUNS" \
            --cool-down 5 \
            --model-dir "$BENCH_DIR/models" \
            --output-dir "$LATENCY_DIR" \
            --skip-tflite \
            2>&1 | tee -a "$LOG_FILE"

        check_quality "$LATENCY_DIR" "ncnn_vulkan_fixed"
    fi

    log "  완료: ncnn_vulkan_fixed"
}

run_accuracy() {
    local model=$1

    log ""
    log "=== 정확도 평가: $model (MKLDNN disabled 내장, n=1000) ==="

    clean_system

    if [[ "$DRY_RUN" == false ]]; then
        python3 "$ACCURACY_PY" \
            --model "$model" \
            --data-dir "$BENCH_DIR/data/imagenet" \
            --max-images 1000 \
            --model-dir "$BENCH_DIR/models" \
            --output-dir "$ACCURACY_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    fi

    log "  정확도 평가 완료: $model"
}

run_layer_analysis() {
    local model=$1

    log ""
    log "=== 레이어 분석: $model ==="

    clean_system

    if [[ "$DRY_RUN" == false ]]; then
        python3 "$LAYER_PY" \
            --model "$model" \
            --device cuda \
            --num-runs 20 \
            --output-dir "$LAYER_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    fi

    log "  레이어 분석 완료: $model"
}

generate_summary() {
    log ""
    log "=== 최종 요약 생성 ==="
    if [[ "$DRY_RUN" == true ]]; then return; fi

    python3 << 'PYEOF'
import json, os, csv, glob

output_base = os.environ.get("OUTPUT_BASE", ".")
latency_dir = os.path.join(output_base, "latency")
accuracy_dir = os.path.join(output_base, "accuracy")

summary = {"timestamp": os.path.basename(output_base), "latency": {}, "accuracy": {}}

# 레이턴시 결과 수집 (benchmark.py 출력: dict 형태)
for jf in sorted(glob.glob(os.path.join(latency_dir, "*", "results.json"))):
    try:
        with open(jf) as f:
            data = json.load(f)

        dirname = os.path.basename(os.path.dirname(jf))

        if isinstance(data, dict):
            # benchmark.py format: {runtime_name: {avg_ms, std_ms, ...}}
            for rt_name, rt_data in data.items():
                key = dirname  # e.g. mobilenetv3_small_10w_20260408_120000
                parts = dirname.rsplit("_", 1)[0]  # remove timestamp
                if parts not in summary["latency"]:
                    summary["latency"][parts] = {}
                summary["latency"][parts][rt_name] = {
                    "avg_ms": rt_data.get("avg_ms"),
                    "std_ms": rt_data.get("std_ms"),
                    "p50_ms": rt_data.get("p50_ms"),
                    "p95_ms": rt_data.get("p95_ms"),
                    "source_dir": dirname
                }

        elif isinstance(data, list):
            # run_failed_runtimes.py format: [{runtime, latency: {mean, std, ...}}]
            for item in data:
                model = item.get("model", "unknown")
                power = item.get("power_mode", "unknown")
                key = f"{model}_{power}"
                if key not in summary["latency"]:
                    summary["latency"][key] = {}
                lat = item["latency"]
                summary["latency"][key][item["runtime"]] = {
                    "avg_ms": lat["mean"],
                    "std_ms": lat["std"],
                    "p50_ms": lat["p50"],
                    "p95_ms": lat["p95"],
                    "source_dir": dirname
                }

    except Exception as e:
        print(f"  WARN: {jf}: {e}")

# 정확도 결과 수집
for model in ["mobilenetv3_small", "resnet50"]:
    acc_file = os.path.join(accuracy_dir, f"{model}_accuracy.csv")
    if os.path.exists(acc_file):
        summary["accuracy"][model] = {}
        with open(acc_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                summary["accuracy"][model][row["runtime"]] = {
                    "top1": float(row["top1"]),
                    "top5": float(row["top5"]),
                    "n": int(row["n"])
                }

out_path = os.path.join(output_base, "clean_summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved: {out_path}")

# 요약 테이블 출력
print(f"\n{'='*80}")
print(f"  CLEAN BENCHMARK SUMMARY")
print(f"{'='*80}")
for group_key in sorted(summary["latency"].keys()):
    print(f"\n  [{group_key}]")
    print(f"  {'Runtime':<25} {'Avg(ms)':<12} {'Std(ms)':<12} {'P95(ms)':<12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for rt, vals in sorted(summary["latency"][group_key].items()):
        avg = vals.get("avg_ms", 0) or 0
        std = vals.get("std_ms", 0) or 0
        p95 = vals.get("p95_ms", 0) or 0
        print(f"  {rt:<25} {avg:<12.2f} {std:<12.2f} {p95:<12.2f}")

if summary["accuracy"]:
    print(f"\n  [Accuracy]")
    for model, rts in summary["accuracy"].items():
        print(f"  {model}:")
        for rt, vals in rts.items():
            print(f"    {rt:<25} top1={vals['top1']:.1f}% top5={vals['top5']:.1f}% (n={vals['n']})")
PYEOF
}

# =============================================================================
# 메인 실행
# =============================================================================

main() {
    # 일반 런타임 11개 + ncnn_vulkan_fixed 1개 = 12 per model/power
    # 2 models × 2 powers × 12 = 48 레이턴시 측정
    local total=$((${#MODELS[@]} * ${#POWER_MODES[@]} * (${#RUNTIMES[@]} + 1)))

    log "=============================================="
    log " Jetson Nano 전체 재벤치마크 (순차)"
    log "=============================================="
    log " 출력: $OUTPUT_BASE"
    log " 모델: ${MODELS[*]}"
    log " 전력: ${POWER_MODES[*]}"
    log " 런타임: ${#RUNTIMES[@]}개 + ncnn_vulkan_fixed = $total 측정"
    log " 설정: warmup=${NUM_WARMUP}, runs=${NUM_RUNS}, cooldown=${COOLDOWN_SEC}s"
    log ""
    log " Phase 1: 레이턴시 ($total 측정, ~4시간)"
    log " Phase 2: 정확도 (2 모델, ~30분)"
    log " Phase 3: 레이어 분석 (2 모델, ~10분)"
    log "=============================================="

    if [[ "$DRY_RUN" == false ]]; then
        mkdir -p "$LATENCY_DIR" "$ACCURACY_DIR" "$LAYER_DIR"
        export OUTPUT_BASE
    fi

    local count=0

    # --- Phase 1: 레이턴시 (전력 모드별 그룹) ---
    for power in "${POWER_MODES[@]}"; do
        log ""
        log "====== Phase 1: 레이턴시 [$power] ======"
        set_power_mode "$power"

        for model in "${MODELS[@]}"; do
            # 일반 런타임 11개
            for runtime in "${RUNTIMES[@]}"; do
                count=$((count + 1))
                run_single_benchmark "$model" "$power" "$runtime" "$count" "$total"
            done

            # ncnn_vulkan_fixed (run_failed_runtimes.py 사용)
            count=$((count + 1))
            run_ncnn_vulkan_fixed "$model" "$power" "$count" "$total"
        done
    done

    # --- Phase 2: 정확도 (10W) ---
    log ""
    log "====== Phase 2: 정확도 평가 ======"
    set_power_mode "10w"
    clean_system

    for model in "${MODELS[@]}"; do
        run_accuracy "$model"
    done

    # --- Phase 3: 레이어 분석 (10W) ---
    log ""
    log "====== Phase 3: 레이어 분석 ======"
    clean_system

    for model in "${MODELS[@]}"; do
        run_layer_analysis "$model"
    done

    # --- Phase 4: 요약 ---
    generate_summary

    log ""
    log "=============================================="
    log " 전체 재벤치마크 완료!"
    log " 결과: $OUTPUT_BASE"
    log "=============================================="
}

main
