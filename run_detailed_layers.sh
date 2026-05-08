#!/bin/bash
# Detailed per-layer extraction entrypoint.
#
# Produces:
#   - architecture/<model>_layers.csv
#   - layer_runtime/<model>_<runtime>.csv
#   - layer_runtime/<model>_<runtime>.json
#   - consolidated_layer_runtime_detailed.csv
# Optional:
#   - layer_power/<model>_layer_power.csv/json
#   - layer_timelines/*.json

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MODEL_DIR="${MODEL_DIR:-models}"
RESULTS_ROOT="results/detailed_layers_$(date +%Y%m%d_%H%M%S)"
NUM_WARMUP="${NUM_WARMUP:-10}"
NUM_RUNS="${NUM_RUNS:-50}"
NUM_AMPLIFY="${NUM_AMPLIFY:-2000}"
RUN_ARCHITECTURE=1
RUN_RUNTIME=1
RUN_POWER=0
RUN_TIMELINES=0
DRY_RUN=0

TFLITE_BENCHMARK_MODEL_BIN="${TFLITE_BENCHMARK_MODEL_BIN:-$HOME/tensorflow-2.13.0/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model}"
NCNN_BENCHNCNN_BIN="${NCNN_BENCHNCNN_BIN:-$HOME/ncnn/build_bench/benchmark/benchncnn}"

MODELS=(
    mobilenetv3_small
    resnet50
    efficientnet_b0
    shufflenet_v2_x1_0
    yolov8n
    ssd_mobilenet_v2
)

RUNTIMES=(
    pytorch_cpu
    pytorch_cuda
    tensorrt_fp32
    tensorrt_fp16
    tensorrt_int8
    onnxrt_cuda
    onnxrt_trt
    tflite_cpu
    tflite_gpu
    ncnn_cpu
    ncnn_vulkan
)

if [ -n "${MODELS_OVERRIDE:-}" ]; then
    read -r -a MODELS <<< "$MODELS_OVERRIDE"
fi

if [ -n "${RUNTIMES_OVERRIDE:-}" ]; then
    read -r -a RUNTIMES <<< "$RUNTIMES_OVERRIDE"
fi

usage() {
    cat <<'EOF'
Usage: bash run_detailed_layers.sh [OPTIONS]

Options:
  --model MODEL              Profile only one model
  --runtime RUNTIME          Profile only one runtime
  --results-dir DIR          Output directory (default: results/detailed_layers_<timestamp>)
  --num-warmup N             Warmup runs for runtime-level profiling
  --num-runs N               Timed runs for runtime-level profiling
  --num-amplify N            Layer-power amplification count
  --skip-architecture        Skip PyTorch module/shape summary
  --skip-runtime             Skip 6 x 11 runtime-aware layer profiling
  --include-power            Also run layer_power_analyzer.py (slow)
  --include-timelines        Also export TRT/PyTorch timeline JSON
  --dry-run                  Print commands without executing
  --help                     Show this help

Environment:
  MODEL_DIR
  TFLITE_BENCHMARK_MODEL_BIN
  NCNN_BENCHNCNN_BIN
  NUM_WARMUP
  NUM_RUNS
  NUM_AMPLIFY
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODELS=("$2")
            shift 2
            ;;
        --runtime)
            RUNTIMES=("$2")
            shift 2
            ;;
        --results-dir)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --num-warmup)
            NUM_WARMUP="$2"
            shift 2
            ;;
        --num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --num-amplify)
            NUM_AMPLIFY="$2"
            shift 2
            ;;
        --skip-architecture)
            RUN_ARCHITECTURE=0
            shift
            ;;
        --skip-runtime)
            RUN_RUNTIME=0
            shift
            ;;
        --include-power)
            RUN_POWER=1
            shift
            ;;
        --include-timelines)
            RUN_TIMELINES=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

mkdir -p "$RESULTS_ROOT"

echo "=================================================="
echo "  Detailed Layer Extraction"
echo "=================================================="
echo "  project:        $PROJECT_DIR"
echo "  model_dir:      $MODEL_DIR"
echo "  results:        $RESULTS_ROOT"
echo "  models:         ${MODELS[*]}"
echo "  runtimes:       ${RUNTIMES[*]}"
echo "  warmup/runs:    $NUM_WARMUP / $NUM_RUNS"
echo "  include_power:  $RUN_POWER"
echo "  timelines:      $RUN_TIMELINES"
echo "=================================================="

write_manifest() {
    cat > "$RESULTS_ROOT/manifest.json" <<EOF
{
  "project_dir": "$PROJECT_DIR",
  "model_dir": "$MODEL_DIR",
  "results_root": "$RESULTS_ROOT",
  "models": "$(printf '%s ' "${MODELS[@]}" | sed 's/ $//')",
  "runtimes": "$(printf '%s ' "${RUNTIMES[@]}" | sed 's/ $//')",
  "num_warmup": $NUM_WARMUP,
  "num_runs": $NUM_RUNS,
  "num_amplify": $NUM_AMPLIFY,
  "include_power": $RUN_POWER,
  "include_timelines": $RUN_TIMELINES
}
EOF
}

run_or_print() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  [DRY] '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

write_manifest

if [ "$RUN_ARCHITECTURE" -eq 1 ]; then
    echo ""
    echo "[1/4] architecture-level PyTorch/TorchScript layer summary"
    ARCH_DIR="$RESULTS_ROOT/architecture"
    mkdir -p "$ARCH_DIR"
    for model in "${MODELS[@]}"; do
        echo "  [architecture] $model"
        run_or_print python3 benchmark/layer_analyzer.py \
            --model "$model" \
            --model-dir "$MODEL_DIR" \
            --output-dir "$ARCH_DIR" \
            || echo "    [warn] architecture profile failed for $model"
    done
else
    echo "[1/4] architecture-level summary skipped"
fi

if [ "$RUN_RUNTIME" -eq 1 ]; then
    echo ""
    echo "[2/4] runtime-aware per-layer/per-node profiling"
    RUNTIME_DIR="$RESULTS_ROOT/layer_runtime"
    mkdir -p "$RUNTIME_DIR"
    for model in "${MODELS[@]}"; do
        for runtime in "${RUNTIMES[@]}"; do
            out_json="$RUNTIME_DIR/${model}_${runtime}.json"
            out_csv="$RUNTIME_DIR/${model}_${runtime}.csv"
            echo "  [runtime] $model / $runtime"
            run_or_print python3 benchmark/layer_runtime_analyzer.py \
                --model "$model" \
                --runtime "$runtime" \
                --model-dir "$MODEL_DIR" \
                --num-warmup "$NUM_WARMUP" \
                --num-runs "$NUM_RUNS" \
                --tflite-benchmark-bin "$TFLITE_BENCHMARK_MODEL_BIN" \
                --ncnn-benchmark-bin "$NCNN_BENCHNCNN_BIN" \
                --output-json "$out_json" \
                --output-csv "$out_csv" \
                || echo "    [warn] runtime profile failed for $model / $runtime"
        done
    done

    CONSOLIDATED="$RESULTS_ROOT/consolidated_layer_runtime_detailed.csv"
    first=1
    : > "$CONSOLIDATED"
    for csv in "$RUNTIME_DIR"/*.csv; do
        [ -f "$csv" ] || continue
        if [ "$first" -eq 1 ]; then
            cat "$csv" >> "$CONSOLIDATED"
            first=0
        else
            tail -n +2 "$csv" >> "$CONSOLIDATED"
        fi
    done
    echo "  [saved] $CONSOLIDATED"
else
    echo "[2/4] runtime-aware profiling skipped"
fi

if [ "$RUN_POWER" -eq 1 ]; then
    echo ""
    echo "[3/4] layer power and energy profiling"
    POWER_DIR="$RESULTS_ROOT/layer_power"
    mkdir -p "$POWER_DIR"
    for model in "${MODELS[@]}"; do
        echo "  [power] $model"
        run_or_print python3 benchmark/layer_power_analyzer.py \
            --model "$model" \
            --device cuda \
            --num-amplify "$NUM_AMPLIFY" \
            --model-dir "$MODEL_DIR" \
            --output-dir "$POWER_DIR" \
            || echo "    [warn] layer power failed for $model"
    done
else
    echo "[3/4] layer power skipped"
fi

if [ "$RUN_TIMELINES" -eq 1 ]; then
    echo ""
    echo "[4/4] timeline traces"
    export LAYER_TIMELINE_OUT_DIR="$RESULTS_ROOT/layer_timelines"
    export MODELS_OVERRIDE="${MODELS[*]}"
    run_or_print bash run_timeline_profiler.sh \
        || echo "    [warn] timeline profiling failed"
else
    echo "[4/4] timeline traces skipped"
fi

echo ""
echo "[DONE] detailed layer data saved to: $RESULTS_ROOT"
find "$RESULTS_ROOT" -maxdepth 2 -type f | sort
