#!/bin/bash
# TensorRT-only debug/measurement entrypoint.
#
# This script intentionally reuses the existing measurement paths:
#   - latency/power: benchmark/benchmark.py
#   - detailed layers: run_detailed_layers.sh -> benchmark/layer_runtime_analyzer.py
#
# Usage:
#   bash run_tensorrt_only.sh
#   bash run_tensorrt_only.sh --model resnet50
#   bash run_tensorrt_only.sh --runtime tensorrt_fp16 --power-mode 10w
#   bash run_tensorrt_only.sh --skip-layers
#   bash run_tensorrt_only.sh --layers-only

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MODELS=(
    "mobilenetv3_small"
    "resnet50"
    "efficientnet_b0"
    "shufflenet_v2_x1_0"
    "yolov8n"
    "ssd_mobilenet_v2"
)
CLASSIFICATION_MODELS=("mobilenetv3_small" "resnet50" "efficientnet_b0" "shufflenet_v2_x1_0")
DETECTION_MODELS=("yolov8n" "ssd_mobilenet_v2")

POWER_MODES=("10w" "5w")
TRT_RUNTIMES=("tensorrt_fp32" "tensorrt_fp16" "tensorrt_int8")

MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results/tensorrt_only_$(date +%Y%m%d_%H%M%S)}"
LATENCY_DIR="$RESULTS_ROOT/latency"
DETAIL_DIR="$RESULTS_ROOT/detailed_layers"
LOG_FILE="$RESULTS_ROOT/run.log"

NUM_WARMUP="${NUM_WARMUP:-10}"
NUM_RUNS="${NUM_RUNS:-100}"
LAYER_NUM_WARMUP="${LAYER_NUM_WARMUP:-10}"
LAYER_NUM_RUNS="${LAYER_NUM_RUNS:-50}"
COOL_DOWN="${COOL_DOWN:-60}"
THERMAL_WAIT="${THERMAL_WAIT:-30}"

SKIP_CONVERT=false
SKIP_BENCHMARK=false
SKIP_LAYERS=false
LAYERS_ONLY=false
DRY_RUN=false
NO_POWER_SWITCH=false
NO_TEGRASTATS=false
TARGET_WEEK=""

usage() {
    cat <<'EOF'
Usage: bash run_tensorrt_only.sh [OPTIONS]

Options:
  --model MODEL              Run a single model
  --week N                   Run models assigned to week N (3, 4, 5, or 6)
  --runtime RUNTIME          Run one TensorRT runtime only
                             (tensorrt_fp32, tensorrt_fp16, tensorrt_int8)
  --include-onnxrt-trt       Also include onnxrt_trt for TensorRT EP checks
  --power-mode MODE          Run one power mode only (10w or 5w)
  --results-dir DIR          Output root directory
  --model-dir DIR            Model artifact directory
  --num-warmup N             Warmup runs for benchmark.py
  --num-runs N               Timed runs for benchmark.py
  --layer-num-warmup N       Warmup runs for detailed layer profiling
  --layer-num-runs N         Timed runs for detailed layer profiling
  --cool-down N              Seconds between runtime/model benchmark calls
  --thermal-wait N           Seconds after power mode change
  --skip-convert             Do not build missing TensorRT engines
  --skip-benchmark           Skip benchmark.py latency/power measurement
  --skip-layers              Skip detailed layer profiling
  --layers-only              Only run detailed layer profiling
  --no-power-switch          Do not call sudo nvpmodel / jetson_clocks
  --no-tegrastats            Pass --no-tegrastats to benchmark.py
  --dry-run                  Print commands without executing
  --help                     Show this help

Output:
  <results-dir>/run.log
  <results-dir>/latency/<model>_<power>_<timestamp>/{results.json,summary.csv}
  <results-dir>/detailed_layers/layer_runtime/*_{tensorrt_*}.{json,csv}
EOF
}

is_classification_model() {
    local model="$1"
    for m in "${CLASSIFICATION_MODELS[@]}"; do
        [ "$m" = "$model" ] && return 0
    done
    return 1
}

is_detection_model() {
    local model="$1"
    for m in "${DETECTION_MODELS[@]}"; do
        [ "$m" = "$model" ] && return 0
    done
    return 1
}

join_by_comma() {
    local IFS=,
    echo "$*"
}

run_or_print() {
    if [ "$DRY_RUN" = true ]; then
        printf '  [DRY] '
        printf '%q ' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODELS=("$2")
            shift 2
            ;;
        --week)
            TARGET_WEEK="$2"
            shift 2
            ;;
        --runtime)
            case "$2" in
                tensorrt_fp32|tensorrt_fp16|tensorrt_int8|onnxrt_trt)
                    TRT_RUNTIMES=("$2")
                    ;;
                *)
                    echo "[ERROR] Unsupported TensorRT runtime: $2" >&2
                    usage
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --include-onnxrt-trt)
            TRT_RUNTIMES+=("onnxrt_trt")
            shift
            ;;
        --power-mode)
            case "$2" in
                10w|5w) POWER_MODES=("$2") ;;
                *)
                    echo "[ERROR] Invalid power mode: $2" >&2
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --results-dir)
            RESULTS_ROOT="$2"
            LATENCY_DIR="$RESULTS_ROOT/latency"
            DETAIL_DIR="$RESULTS_ROOT/detailed_layers"
            LOG_FILE="$RESULTS_ROOT/run.log"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
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
        --layer-num-warmup)
            LAYER_NUM_WARMUP="$2"
            shift 2
            ;;
        --layer-num-runs)
            LAYER_NUM_RUNS="$2"
            shift 2
            ;;
        --cool-down)
            COOL_DOWN="$2"
            shift 2
            ;;
        --thermal-wait)
            THERMAL_WAIT="$2"
            shift 2
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            shift
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift
            ;;
        --skip-layers)
            SKIP_LAYERS=true
            shift
            ;;
        --layers-only)
            LAYERS_ONLY=true
            SKIP_BENCHMARK=true
            shift
            ;;
        --no-power-switch)
            NO_POWER_SWITCH=true
            shift
            ;;
        --no-tegrastats)
            NO_TEGRASTATS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
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

if [ -n "$TARGET_WEEK" ]; then
    case "$TARGET_WEEK" in
        3) MODELS=("mobilenetv3_small") ;;
        4) MODELS=("efficientnet_b0" "shufflenet_v2_x1_0") ;;
        5) MODELS=("resnet50") ;;
        6) MODELS=("yolov8n" "ssd_mobilenet_v2") ;;
        *)
            echo "[ERROR] Invalid week: $TARGET_WEEK. valid weeks: 3, 4, 5, 6." >&2
            exit 1
            ;;
    esac
fi

mkdir -p "$RESULTS_ROOT" "$LATENCY_DIR"
if [ "$DRY_RUN" = false ]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "=================================================="
echo "  TensorRT-only Debug Benchmark"
echo "=================================================="
echo "  project:        $PROJECT_DIR"
echo "  model_dir:      $MODEL_DIR"
echo "  results:        $RESULTS_ROOT"
echo "  models:         ${MODELS[*]}"
echo "  power_modes:    ${POWER_MODES[*]}"
echo "  runtimes:       ${TRT_RUNTIMES[*]}"
echo "  warmup/runs:    $NUM_WARMUP / $NUM_RUNS"
echo "  layer warm/runs:$LAYER_NUM_WARMUP / $LAYER_NUM_RUNS"
echo "  skip_convert:   $SKIP_CONVERT"
echo "  skip_benchmark: $SKIP_BENCHMARK"
echo "  skip_layers:    $SKIP_LAYERS"
echo "  dry_run:        $DRY_RUN"
echo "=================================================="

write_manifest() {
    cat > "$RESULTS_ROOT/manifest.json" <<EOF
{
  "project_dir": "$PROJECT_DIR",
  "model_dir": "$MODEL_DIR",
  "results_root": "$RESULTS_ROOT",
  "models": "$(printf '%s ' "${MODELS[@]}" | sed 's/ $//')",
  "power_modes": "$(printf '%s ' "${POWER_MODES[@]}" | sed 's/ $//')",
  "runtimes": "$(printf '%s ' "${TRT_RUNTIMES[@]}" | sed 's/ $//')",
  "num_warmup": $NUM_WARMUP,
  "num_runs": $NUM_RUNS,
  "layer_num_warmup": $LAYER_NUM_WARMUP,
  "layer_num_runs": $LAYER_NUM_RUNS,
  "skip_convert": $SKIP_CONVERT,
  "skip_benchmark": $SKIP_BENCHMARK,
  "skip_layers": $SKIP_LAYERS
}
EOF
}

resolve_onnx_path() {
    local model="$1"
    local onnx_path="$MODEL_DIR/${model}.onnx"
    if [ -f "$onnx_path" ]; then
        echo "$onnx_path"
        return 0
    fi
    for candidate in \
        "$MODEL_DIR/${model}_raw_fpinput.onnx" \
        "$MODEL_DIR/${model}_raw3_fpinput.onnx" \
        "$MODEL_DIR/${model}_raw.onnx" \
        "$MODEL_DIR/${model}_effnms.onnx"; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    echo "$onnx_path"
    return 1
}

engine_exists_for_precision() {
    local model="$1"
    local precision="$2"
    for candidate in \
        "$MODEL_DIR/${model}_${precision}.engine" \
        "$MODEL_DIR/${model}_raw_${precision}.engine" \
        "$MODEL_DIR/${model}_raw_fpinput_${precision}.engine" \
        "$MODEL_DIR/${model}_raw3_fpinput_${precision}.engine" \
        "$MODEL_DIR/${model}_effnms_fpinput_${precision}.engine" \
        "$MODEL_DIR/${model}_effnms_${precision}.engine"; do
        [ -s "$candidate" ] && return 0
    done
    return 1
}

prepare_trt_engines() {
    [ "$SKIP_CONVERT" = true ] && return 0

    echo ""
    echo "[1/3] TensorRT engine preparation"
    for model in "${MODELS[@]}"; do
        echo "  [model] $model"

        local need_build=false
        for precision in fp32 fp16 int8; do
            if ! engine_exists_for_precision "$model" "$precision"; then
                need_build=true
            fi
        done

        if [ "$need_build" = false ]; then
            echo "    engines already exist"
            continue
        fi

        local onnx_path
        if ! onnx_path="$(resolve_onnx_path "$model")"; then
            if is_classification_model "$model"; then
                echo "    ONNX missing; exporting classification model"
                run_or_print python3 convert/export_onnx.py --model "$model" --output-dir "$MODEL_DIR" || {
                    echo "    [warn] ONNX export failed for $model"
                    continue
                }
                onnx_path="$MODEL_DIR/${model}.onnx"
            else
                echo "    [warn] ONNX not found for detection model: $model"
                continue
            fi
        fi

        echo "    building from: $onnx_path"
        run_or_print bash convert/build_trt.sh "$onnx_path" || {
            echo "    [warn] TRT build failed for $model"
            continue
        }

        local onnx_stem
        onnx_stem="$(basename "$onnx_path" .onnx)"
        if [ "$onnx_stem" != "$model" ]; then
            for precision in fp32 fp16 int8; do
                local src="$MODEL_DIR/${onnx_stem}_${precision}.engine"
                local dst="$MODEL_DIR/${model}_${precision}.engine"
                if [ -f "$src" ] && [ ! -f "$dst" ]; then
                    run_or_print mv "$src" "$dst"
                    echo "    renamed: $(basename "$src") -> $(basename "$dst")"
                fi
            done
        fi
    done
}

set_power_mode() {
    local power="$1"
    if [ "$NO_POWER_SWITCH" = true ]; then
        echo "  [power] no-power-switch enabled; keeping current mode"
        return 0
    fi
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY] sudo nvpmodel -m $([ "$power" = "5w" ] && echo 1 || echo 0)"
        echo "  [DRY] sudo jetson_clocks"
        echo "  [DRY] sleep $THERMAL_WAIT"
        return 0
    fi
    if [ "$power" = "5w" ]; then
        sudo nvpmodel -m 1 || echo "  [warn] nvpmodel 5w failed"
    else
        sudo nvpmodel -m 0 || echo "  [warn] nvpmodel 10w failed"
    fi
    sudo jetson_clocks 2>/dev/null || true
    echo "  Waiting ${THERMAL_WAIT}s for thermal stabilization..."
    sleep "$THERMAL_WAIT"
}

run_latency_benchmarks() {
    [ "$SKIP_BENCHMARK" = true ] && return 0

    echo ""
    echo "[2/3] TensorRT latency/power benchmark"
    local runtime_csv
    runtime_csv="$(join_by_comma "${TRT_RUNTIMES[@]}")"
    local failures=0

    for model in "${MODELS[@]}"; do
        for power in "${POWER_MODES[@]}"; do
            echo ""
            echo "  [benchmark] $model / $power / $runtime_csv"
            set_power_mode "$power"

            local cmd=(
                python3 benchmark/benchmark.py
                --model "$model"
                --runtimes "$runtime_csv"
                --power-mode "$power"
                --num-warmup "$NUM_WARMUP"
                --num-runs "$NUM_RUNS"
                --cool-down "$COOL_DOWN"
                --model-dir "$MODEL_DIR"
                --output-dir "$LATENCY_DIR"
            )
            if [ "$NO_TEGRASTATS" = true ]; then
                cmd+=(--no-tegrastats)
            fi

            run_or_print "${cmd[@]}" || {
                echo "    [warn] benchmark.py failed for $model / $power"
                failures=$((failures + 1))
            }
        done
    done

    if [ "$failures" -gt 0 ]; then
        echo "  [warn] benchmark failures: $failures"
    fi
}

run_detailed_layers() {
    [ "$SKIP_LAYERS" = true ] && return 0

    echo ""
    echo "[3/3] TensorRT detailed layer profiling"
    mkdir -p "$DETAIL_DIR"

    local layer_runtimes=()
    for rt in "${TRT_RUNTIMES[@]}"; do
        case "$rt" in
            tensorrt_fp32|tensorrt_fp16|tensorrt_int8|onnxrt_trt)
                layer_runtimes+=("$rt")
                ;;
        esac
    done

    if [ "${#layer_runtimes[@]}" -eq 0 ]; then
        echo "  [skip] no detailed-layer compatible runtime selected"
        return 0
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY] MODELS_OVERRIDE='${MODELS[*]}' RUNTIMES_OVERRIDE='${layer_runtimes[*]}' bash run_detailed_layers.sh --results-dir '$DETAIL_DIR' --num-warmup '$LAYER_NUM_WARMUP' --num-runs '$LAYER_NUM_RUNS' --skip-architecture"
        return 0
    fi

    MODELS_OVERRIDE="${MODELS[*]}" \
    RUNTIMES_OVERRIDE="${layer_runtimes[*]}" \
    bash run_detailed_layers.sh \
        --results-dir "$DETAIL_DIR" \
        --num-warmup "$LAYER_NUM_WARMUP" \
        --num-runs "$LAYER_NUM_RUNS" \
        --skip-architecture || {
        echo "  [warn] detailed layer profiling failed"
        return 0
    }
}

write_manifest
if [ "$LAYERS_ONLY" = false ]; then
    prepare_trt_engines
fi
run_latency_benchmarks
run_detailed_layers

echo ""
echo "[DONE] TensorRT-only run complete"
echo "  results: $RESULTS_ROOT"
echo "  log:     $LOG_FILE"
find "$RESULTS_ROOT" -maxdepth 3 -type f | sort
