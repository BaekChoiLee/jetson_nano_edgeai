#!/bin/bash
# =============================================================================
# rebenchmark_clean_v2.sh — 6 모델 × 11 런타임 통합 벤치마크 wrapper
# =============================================================================
# 역할: 기존 rebenchmark_clean.sh (2 모델) 를 손대지 않고, 6 모델 풀 커버리지
#       파이프라인을 별도 진입점으로 제공한다. 각 단계를 명시적으로 분기하므로
#       Phase 5 에서 사용자가 직접 한 줄로 돌릴 수 있다.
#
# 흐름:
#   1) clean_system     — 캐시·임시파일·이전 결과 정리
#   2) check_models     — 6 모델 아티팩트 존재 검증
#   3) bench_detect     — 2 검출 모델 × 11 런타임 (run_detection_latency.py 호출)
#   4) bench_classify   — 4 분류 모델 × 11 런타임 (기존 benchmark.py 호출)
#   5) accuracy_detect   — COCO mAP (run_detection_accuracy.py)
#   6) accuracy_classify — 분류 mAP 대신 top-1/top-5 (기존 accuracy_eval.py)
#   7) layer_runtime     — 6 모델 × 11 런타임 레이어 latency (PyTorch/TRT, others N/A)
#   8) consolidate       — 결과 통합 (기존 consolidate_results.py 호출)
#
# 중요:
#   - **Claude 가 실행하지 않음** — Phase 5 사용자 전담
#   - 각 단계는 idempotent: 결과 디렉토리가 있으면 skip
#   - 실패한 (model, runtime) 조합은 N/A 로 기록 후 다음 진행
#
# 사용법:
#   bash rebenchmark_clean_v2.sh                # 전체 실행
#   bash rebenchmark_clean_v2.sh --skip-classify # 분류 스킵
#   bash rebenchmark_clean_v2.sh --only-power   # 레이어 전력만
# =============================================================================

# 실행 중 원본 파일이 갱신되면 EOF parse 오류가 날 수 있어 임시 복사본에서 실행.
if [ -z "${RB_V2_LOCKED_COPY:-}" ] && command -v mktemp >/dev/null 2>&1; then
    _src="$0"
    _src_dir="$(cd "$(dirname "$_src")" && pwd)"
    _copy="$(mktemp /tmp/rebenchmark_clean_v2.XXXXXX.sh 2>/dev/null || true)"
    if [ -n "$_copy" ] && cp "$_src" "$_copy" 2>/dev/null; then
        chmod +x "$_copy" 2>/dev/null || true
        export RB_V2_LOCKED_COPY=1
        export RB_V2_ORIG_DIR="$_src_dir"
        exec bash "$_copy" "$@"
    fi
fi

set -uo pipefail
# pipefail 만 사용 (errexit 비활성화) — 런타임 실패 시 다음 단계로 진행하기 위해

# 스크립트 위치 기준 (잠금 복사 실행 시 원본 경로 우선)
SCRIPT_DIR="${RB_V2_ORIG_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"

# =============================================================================
# 설정
# =============================================================================
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT_DEFAULT="results/clean_v2_${TIMESTAMP}"
RESULTS_ROOT_ARG=""
RESULTS_ROOT=""
LOG_FILE=""

CLASSIFICATION_MODELS=(
    "mobilenetv3_small"
    "resnet50"
    "efficientnet_b0"
    "shufflenet_v2_x1_0"
)

DETECTION_MODELS=(
    "yolov8n"
    "ssd_mobilenet_v2"
)

ALL_RUNTIMES=(
    "pytorch_cpu" "pytorch_cuda"
    "tensorrt_fp32" "tensorrt_fp16" "tensorrt_int8"
    "onnxrt_cuda" "onnxrt_trt"
    "tflite_cpu" "tflite_gpu"
    "ncnn_cpu" "ncnn_vulkan"
)

POWER_MODES=("MAXN" "5W")

# 스택 제약으로 불가한 셀: TF 2.4.1 MLIR 버그로 변환 불가 (JetPack 4.6.1)
skip_cell() {
    case "$1:$2" in
        shufflenet_v2_x1_0:tflite_cpu|shufflenet_v2_x1_0:tflite_gpu) return 0 ;;
    esac
    return 1
}
NUM_WARMUP=20
NUM_RUNS=200
COOLDOWN_SEC=60
DEFAULT_TFLITE_BENCHMARK_MODEL_BIN="$HOME/jetson-benchmark/tools/benchmark_model"
DEFAULT_NCNN_BENCHNCNN_BIN="$HOME/ncnn/build_bench/benchmark/benchncnn"
TFLITE_BENCHMARK_MODEL_BIN="${TFLITE_BENCHMARK_MODEL_BIN:-$DEFAULT_TFLITE_BENCHMARK_MODEL_BIN}"
NCNN_BENCHNCNN_BIN="${NCNN_BENCHNCNN_BIN:-$DEFAULT_NCNN_BENCHNCNN_BIN}"

# 옵션 파싱
SKIP_CLASSIFY=0
SKIP_DETECT=0
SKIP_ACCURACY=0
SKIP_POWER=0
ONLY_POWER=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root)
            RESULTS_ROOT_ARG="$2"
            shift 2 ;;
        --skip-classify) SKIP_CLASSIFY=1; shift ;;
        --skip-detect)   SKIP_DETECT=1; shift ;;
        --skip-accuracy) SKIP_ACCURACY=1; shift ;;
        --skip-power)    SKIP_POWER=1; shift ;;
        --only-power)
            ONLY_POWER=1
            SKIP_CLASSIFY=1; SKIP_DETECT=1; SKIP_ACCURACY=1
            shift ;;
        -h|--help)
            grep "^# " "$0" | head -40
            echo "  --results-root <path>   기존 결과 경로를 이어서 실행"
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done

if [ -n "$RESULTS_ROOT_ARG" ]; then
    RESULTS_ROOT="$RESULTS_ROOT_ARG"
    # clean_v2_YYYYmmdd_HHMMSS 형태면 timestamp 표시를 동기화
    if [[ "$RESULTS_ROOT" =~ clean_v2_([0-9]{8}_[0-9]{6})$ ]]; then
        TIMESTAMP="${BASH_REMATCH[1]}"
    fi
else
    RESULTS_ROOT="$RESULTS_ROOT_DEFAULT"
fi
LOG_FILE="${RESULTS_ROOT}/run.log"

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "rebenchmark_clean_v2 — 6 model × 11 runtime"
echo "  timestamp: $TIMESTAMP"
echo "  results:   $RESULTS_ROOT"
echo "  models (cls): ${CLASSIFICATION_MODELS[*]}"
echo "  models (det): ${DETECTION_MODELS[*]}"
echo "  runtimes:  ${#ALL_RUNTIMES[@]} runtimes"
echo "  warmup×runs: ${NUM_WARMUP}×${NUM_RUNS}"
echo "============================================================"

# =============================================================================
# 헬퍼: 시스템 정리 + 쿨다운
# =============================================================================
clean_system() {
    echo "[clean] dropping caches, sleeping ${COOLDOWN_SEC}s..."
    sync || true
    if [ -w /proc/sys/vm/drop_caches ]; then
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    fi
    sleep "$COOLDOWN_SEC"
}

set_power_mode() {
    local mode="$1"
    if command -v nvpmodel >/dev/null 2>&1; then
        if [ "$mode" = "MAXN" ]; then
            sudo nvpmodel -m 0 || true
        else
            sudo nvpmodel -m 1 || true
        fi
        sudo jetson_clocks || true
    fi
    echo "[power] mode set to $mode"
}

has_results_json_tree() {
    local target_dir="$1"
    [ -d "$target_dir" ] || return 1
    find "$target_dir" -type f -name "results.json" -print -quit 2>/dev/null | grep -q .
}

# =============================================================================
# Step 1: 모델 아티팩트 존재 검증
# =============================================================================
check_models() {
    echo ""
    echo "[2/8] check_models — 6 모델 아티팩트 존재 검증"
    local missing=0
    for m in "${CLASSIFICATION_MODELS[@]}"; do
        if [ ! -f "models/${m}.onnx" ]; then
            echo "  [MISS] models/${m}.onnx"
            missing=$((missing+1))
        else
            echo "  [OK]   models/${m}.onnx"
        fi
    done
    for m in "${DETECTION_MODELS[@]}"; do
        if [ "$m" = "yolov8n" ] && [ -f "models/${m}.onnx" ]; then
            echo "  [OK]   models/${m}.onnx"
        elif [ "$m" = "ssd_mobilenet_v2" ] && [ -f "models/${m}_raw.onnx" ]; then
            echo "  [OK]   models/${m}_raw.onnx"
        else
            echo "  [MISS] detection artifact for ${m}"
            missing=$((missing+1))
        fi
    done
    if [ "$missing" -gt 0 ]; then
        echo "  [WARN] $missing artifacts missing — affected runtimes will be N/A"
    fi
}

# =============================================================================
# Step 1.5: 검출 아티팩트 준비 (YOLO/SSD 공통)
# =============================================================================
prepare_detection_artifacts_v2() {
    echo ""
    echo "[2.5/8] prepare_detection_artifacts — detect 전용 아티팩트 보강"
    if [ ! -f "./rebenchmark_detection_2models.sh" ]; then
        echo "  [skip] rebenchmark_detection_2models.sh 없음"
        return
    fi
    # 준비 단계만 호출: latency/accuracy 실제 측정은 본 스크립트에서 수행
    bash ./rebenchmark_detection_2models.sh \
        --results-root "${RESULTS_ROOT}/_prep_detection" \
        --skip-latency \
        --skip-accuracy \
        || echo "  [warn] detection artifact prep failed"
}

# =============================================================================
# Step 2: 분류 모델 12 런타임 지연
# =============================================================================
bench_classify() {
    if [ "$SKIP_CLASSIFY" -eq 1 ]; then
        echo "[4/8] bench_classify — SKIPPED"
        return
    fi
    echo ""
    echo "[4/8] bench_classify — 4 모델 × 11 런타임 × 2 전력모드"
    for mode in "${POWER_MODES[@]}"; do
        set_power_mode "$mode"
        for model in "${CLASSIFICATION_MODELS[@]}"; do
            for rt in "${ALL_RUNTIMES[@]}"; do
                if skip_cell "$model" "$rt"; then echo "  [stack-skip] $model/$rt"; continue; fi
                local out_dir="${RESULTS_ROOT}/latency/${mode}/${model}_${rt}"
                local legacy_out="${RESULTS_ROOT}/latency/${mode}/${model}_${rt}.json"
                if [ -f "$legacy_out" ]; then
                    echo "  [skip] $legacy_out (legacy file)"
                    continue
                fi
                if has_results_json_tree "$out_dir"; then
                    echo "  [skip] $out_dir (results.json exists)"
                    continue
                fi
                mkdir -p "$out_dir"
                clean_system
                echo "  [run] $mode / $model / $rt"
                python3 benchmark/benchmark.py \
                    --model "$model" \
                    --runtimes "$rt" \
                    --num-warmup "$NUM_WARMUP" \
                    --num-runs "$NUM_RUNS" \
                    --output-dir "$out_dir" \
                    || echo "    [fail] $model / $rt"
            done
        done
    done
}

# =============================================================================
# Step 3: 검출 모델 12 런타임 지연
# =============================================================================
bench_detect() {
    if [ "$SKIP_DETECT" -eq 1 ]; then
        echo "[3/8] bench_detect — SKIPPED"
        return
    fi
    echo ""
    echo "[3/8] bench_detect — 2 모델 × 11 런타임 × 2 전력모드"
    for mode in "${POWER_MODES[@]}"; do
        set_power_mode "$mode"
        for model in "${DETECTION_MODELS[@]}"; do
            for rt in "${ALL_RUNTIMES[@]}"; do
                local out="${RESULTS_ROOT}/latency/${mode}/${model}_${rt}.json"
                local tegra="${RESULTS_ROOT}/latency/${mode}/${model}_${rt}.tegra.log"
                if [ -f "$out" ]; then
                    echo "  [skip] $out"
                    continue
                fi
                mkdir -p "$(dirname "$out")"
                clean_system
                echo "  [run] $mode / $model / $rt"
                python3 benchmark/run_detection_latency.py \
                    --model "$model" \
                    --runtime "$rt" \
                    --num-warmup "$NUM_WARMUP" \
                    --num-runs "$NUM_RUNS" \
                    --tegra-log "$tegra" \
                    --output "$out" \
                    || echo "    [fail] $model / $rt"
            done
        done
    done
}

# =============================================================================
# Step 4: 분류 정확도 (top-1/top-5)
# =============================================================================
accuracy_classify() {
    if [ "$SKIP_ACCURACY" -eq 1 ]; then
        echo "[6/8] accuracy_classify — SKIPPED"
        return
    fi
    echo ""
    echo "[6/8] accuracy_classify — 4 모델 × 11 런타임 ImageNet top-1/5"
    for model in "${CLASSIFICATION_MODELS[@]}"; do
        local out_dir="${RESULTS_ROOT}/accuracy"
        local out="${RESULTS_ROOT}/accuracy/${model}_classification.json"
        if [ -f "$out" ]; then
            echo "  [skip] $out"
            continue
        fi
        mkdir -p "$out_dir"
        echo "  [run] $model"
        # 기존 accuracy_eval.py 가 12 런타임을 모두 지원해야 함
        # (지원하지 않는 런타임은 N/A 로 기록되어야 함)
        # OMP_NUM_THREADS=1: aarch64 NEON SIMD 안정성 (pytorch_cpu 정확도 버그 방지)
        OMP_NUM_THREADS=1 python3 benchmark/accuracy_eval.py \
            --model "$model" \
            --data-dir "data/imagenet_val" \
            --model-dir "models" \
            --max-images 500 \
            --output-dir "$out_dir" \
            || echo "    [fail] $model"
    done
}

# =============================================================================
# Step 5: 검출 정확도 (COCO mAP)
# =============================================================================
accuracy_detect() {
    if [ "$SKIP_ACCURACY" -eq 1 ]; then
        echo "[5/8] accuracy_detect — SKIPPED"
        return
    fi
    echo ""
    echo "[5/8] accuracy_detect — 2 모델 × 11 런타임 COCO mAP"

    # COCO 데이터 검증
    if [ ! -f "data/coco_val/annotations/instances_val2017_subset500.json" ]; then
        echo "  [WARN] COCO subset 없음 → bash data/coco_val/download_subset.sh 먼저 실행"
        return
    fi

    for model in "${DETECTION_MODELS[@]}"; do
        local out="${RESULTS_ROOT}/accuracy/${model}_detection.json"
        if [ -f "$out" ]; then
            echo "  [skip] $out"
            continue
        fi
        mkdir -p "$(dirname "$out")"
        echo "  [run] $model"
        python3 benchmark/run_detection_accuracy.py \
            --model "$model" \
            --output "$out" \
            || echo "    [fail] $model"
    done
}

# =============================================================================
# Step 6: 런타임별 레이어 latency
# =============================================================================
layer_power() {
    if [ "$SKIP_POWER" -eq 1 ]; then
        echo "[7/8] layer_runtime — SKIPPED"
        return
    fi
    echo ""
    echo "[7/8] layer_runtime — 6 모델 × 11 런타임 레이어 latency"
    echo "  [tool] TFLite benchmark_model: $TFLITE_BENCHMARK_MODEL_BIN"
    echo "  [tool] ncnn benchncnn:        $NCNN_BENCHNCNN_BIN"
    set_power_mode "MAXN"
    local out_dir="${RESULTS_ROOT}/layer_runtime"
    mkdir -p "$out_dir"
    for model in "${CLASSIFICATION_MODELS[@]}" "${DETECTION_MODELS[@]}"; do
        for rt in "${ALL_RUNTIMES[@]}"; do
            local out_json="${out_dir}/${model}_${rt}.json"
            local out_csv="${out_dir}/${model}_${rt}.csv"
            if [ -f "$out_json" ]; then
                echo "  [skip] ${model} / ${rt}"
                continue
            fi
            clean_system
            echo "  [run] $model / $rt"
            python3 benchmark/layer_runtime_analyzer.py \
                --model "$model" \
                --runtime "$rt" \
                --model-dir "models" \
                --num-warmup 10 \
                --num-runs 50 \
                --tflite-benchmark-bin "$TFLITE_BENCHMARK_MODEL_BIN" \
                --ncnn-benchmark-bin "$NCNN_BENCHNCNN_BIN" \
                --output-json "$out_json" \
                --output-csv "$out_csv" \
                || echo "    [fail] layer_runtime for $model / $rt"
        done
    done
}

# =============================================================================
# Step 7: 결과 통합
# =============================================================================
consolidate() {
    echo ""
    echo "[8/8] consolidate — 결과 통합"
    if [ -f "analysis/consolidate_v2.py" ]; then
        python3 analysis/consolidate_v2.py \
            --results-dir "$RESULTS_ROOT" \
            --output-dir "$RESULTS_ROOT" \
            || echo "  [warn] consolidate failed"
    else
        echo "  [skip] analysis/consolidate_results.py 없음"
    fi
}

# =============================================================================
# 실행
# =============================================================================
echo ""
echo "[1/8] clean_system 초기화"
clean_system
check_models
prepare_detection_artifacts_v2
bench_detect
bench_classify
accuracy_detect
accuracy_classify
layer_power
consolidate

echo ""
echo "============================================================"
echo "[DONE] rebenchmark_clean_v2 완료"
echo "  results: $RESULTS_ROOT"
echo "============================================================"

# 결과 자동 압축
echo ""
echo "📦 결과를 자동으로 압축합니다..."
tar_name="${RESULTS_ROOT%/}.tar.gz"
tar -czvf "$tar_name" "$RESULTS_ROOT" > /dev/null 2>&1
echo "✅ 압축 완료: $tar_name (젯슨 나노에서 이 파일을 가져가세요!)"
