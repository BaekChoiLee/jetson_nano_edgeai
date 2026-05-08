#!/bin/bash
# Full Jetson Nano benchmark automation script
# Supports 6 models: 4 classification + 2 detection
# Usage: ./run_all.sh [--model MODEL] [--skip-convert] [--skip-setup] [--dry-run]
set -e

# ============================================
# Configuration
# ============================================
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
COOL_DOWN=60          # seconds between experiments
NUM_WARMUP=10
NUM_RUNS=100
LAYER_NUM_WARMUP="${LAYER_NUM_WARMUP:-10}"
LAYER_NUM_RUNS="${LAYER_NUM_RUNS:-50}"
THERMAL_WAIT=30       # seconds to wait after power mode change
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${PROJECT_DIR}/models"
RESULTS_DIR="${PROJECT_DIR}/results"
DATA_DIR="${PROJECT_DIR}/data/imagenet_val"
COCO_DIR="${PROJECT_DIR}/data/coco_val"
DETAILED_LAYERS=false
INCLUDE_LAYER_POWER=false
INCLUDE_LAYER_TIMELINES=false

# No skip combos — let all runtimes run; failures are recorded as error results
SKIP_COMBOS=()

# ============================================
# Helper Functions
# ============================================
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

should_skip_combo() {
    local model="$1"
    local runtime="$2"
    local combo="${model}:${runtime}"
    for skip in "${SKIP_COMBOS[@]}"; do
        [ "$skip" = "$combo" ] && return 0
    done
    return 1
}

# ============================================
# Argument Parsing
# ============================================
SKIP_CONVERT=false
SKIP_SETUP=false
DRY_RUN=false
SINGLE_MODEL=""
TARGET_WEEK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       SINGLE_MODEL="$2"; shift 2;;
        --week)        TARGET_WEEK="$2"; shift 2;;
        --skip-convert) SKIP_CONVERT=true; shift;;
        --skip-setup)  SKIP_SETUP=true; shift;;
        --detailed-layers) DETAILED_LAYERS=true; shift;;
        --include-layer-power) INCLUDE_LAYER_POWER=true; shift;;
        --include-layer-timelines) INCLUDE_LAYER_TIMELINES=true; shift;;
        --dry-run)     DRY_RUN=true; shift;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL     Run a single model"
            echo "  --week N          Run models assigned to week N (3, 4, 5, or 6)"
            echo "  --skip-convert    Skip model conversion step"
            echo "  --skip-setup      Skip environment verification"
            echo "  --detailed-layers Run 6x11 runtime-aware layer profiling after Step 4"
            echo "  --include-layer-power"
            echo "                    Include slow Layer Amplification power profiling"
            echo "  --include-layer-timelines"
            echo "                    Export TRT/PyTorch timeline JSON traces"
            echo "  --dry-run         Print what would be done without executing"
            echo "  --help            Show this help"
            echo ""
            echo "Models (6 total):"
            echo "  Classification: mobilenetv3_small, resnet50, efficientnet_b0, shufflenet_v2_x1_0"
            echo "  Detection:      yolov8n, ssd_mobilenet_v2"
            echo ""
            echo "Week assignments:"
            echo "  Week 3: mobilenetv3_small"
            echo "  Week 4: efficientnet_b0, shufflenet_v2_x1_0"
            echo "  Week 5: resnet50"
            echo "  Week 6: yolov8n, ssd_mobilenet_v2"
            exit 0;;
        *)
            echo "Unknown option: $1"
            exit 1;;
    esac
done

# Filter models if single model specified
if [ -n "$SINGLE_MODEL" ]; then
    MODELS=("$SINGLE_MODEL")
fi

# Filter models if week specified
if [ -n "$TARGET_WEEK" ]; then
    if [ "$TARGET_WEEK" = "3" ]; then
        MODELS=("mobilenetv3_small")
    elif [ "$TARGET_WEEK" = "4" ]; then
        MODELS=("efficientnet_b0" "shufflenet_v2_x1_0")
    elif [ "$TARGET_WEEK" = "5" ]; then
        MODELS=("resnet50")
    elif [ "$TARGET_WEEK" = "6" ]; then
        MODELS=("yolov8n" "ssd_mobilenet_v2")
    else
        echo "[ERROR] Invalid week: $TARGET_WEEK. valid weeks: 3, 4, 5, 6."
        exit 1
    fi
fi

cd "$PROJECT_DIR"

echo "============================================"
echo "  Jetson Nano Benchmark Suite (6 Models)"
echo "============================================"
echo "  Models:      ${MODELS[*]}"
echo "  Power modes: ${POWER_MODES[*]}"
echo "  Warmup:      ${NUM_WARMUP}"
echo "  Runs:        ${NUM_RUNS}"
echo "  Cool-down:   ${COOL_DOWN}s"
echo "  Project:     ${PROJECT_DIR}"
echo "  Dry run:     ${DRY_RUN}"
echo "  Detailed layers: ${DETAILED_LAYERS}"
echo "  Layer power: ${INCLUDE_LAYER_POWER}"
echo "  Layer timelines: ${INCLUDE_LAYER_TIMELINES}"
echo "  Skip combos: ${SKIP_COMBOS[*]}"
echo "============================================"
echo ""

# ============================================
# Pre-flight Checks
# ============================================
echo "=== Pre-flight Checks ==="

# Disk space (need at least 2GB free)
DISK_FREE=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
echo "  Disk free: ${DISK_FREE}GB"
if [ "${DISK_FREE}" -lt 2 ]; then
    echo "  [WARN] Less than 2GB free disk space!"
fi

# Memory
echo "  Memory: $(free -h | grep Mem | awk '{print $7}') available"
echo "  Swap:   $(free -h | grep Swap | awk '{print $3}') used / $(free -h | grep Swap | awk '{print $2}') total"

# Temperature
if [ -f /sys/devices/virtual/thermal/thermal_zone0/temp ]; then
    TEMP=$(cat /sys/devices/virtual/thermal/thermal_zone0/temp)
    TEMP_C=$((TEMP / 1000))
    echo "  Temperature: ${TEMP_C}°C"
    if [ "$TEMP_C" -gt 70 ]; then
        echo "  [WARN] Temperature is high! Wait for cooling."
    fi
fi

echo ""

# ============================================
# Step 0: Verify Environment
# ============================================
if [ "$SKIP_SETUP" = false ]; then
    echo "=== Step 0: Verifying Environment ==="
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY] Would run: python3 setup/verify_env.py"
    else
        python3 setup/verify_env.py || {
            echo "[ERROR] Environment verification failed!"
            echo "  Run setup scripts first:"
            echo "    bash setup/setup_env.sh"
            echo "    bash setup/install_onnxrt.sh"
            echo "    bash setup/install_ncnn.sh"
            echo "    bash setup/install_tflite.sh"
            exit 1
        }
    fi
    echo ""
fi

# ============================================
# Step 1: Model Conversion
# ============================================
if [ "$SKIP_CONVERT" = false ]; then
    echo "=== Step 1: Model Conversion ==="
    mkdir -p "$MODEL_DIR"

    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "--- Converting: $MODEL ---"

        # Resolve ONNX path: detection models may use _raw_fpinput.onnx or _raw.onnx
        # Prefer _fpinput variant (float32 input) for TRT/ORT compatibility
        ONNX_PATH="$MODEL_DIR/${MODEL}.onnx"
        if [ ! -f "$ONNX_PATH" ]; then
            if [ -f "$MODEL_DIR/${MODEL}_raw_fpinput.onnx" ]; then
                ONNX_PATH="$MODEL_DIR/${MODEL}_raw_fpinput.onnx"
            elif [ -f "$MODEL_DIR/${MODEL}_raw.onnx" ]; then
                ONNX_PATH="$MODEL_DIR/${MODEL}_raw.onnx"
            fi
        fi

        if [ "$DRY_RUN" = true ]; then
            if is_classification_model "$MODEL"; then
                echo "  [DRY] python3 convert/export_onnx.py --model $MODEL --output-dir $MODEL_DIR"
            else
                echo "  [DRY] Detection model — ONNX must be pre-built (${MODEL}.onnx or ${MODEL}_raw.onnx)"
            fi
            echo "  [DRY] bash convert/build_trt.sh $ONNX_PATH"
            echo "  [DRY] python3 convert/convert_tflite.py --onnx-path $ONNX_PATH"
            echo "  [DRY] bash convert/convert_ncnn.sh $ONNX_PATH"
            continue
        fi

        # ONNX export (classification only; detection ONNX is pre-built)
        if [ ! -f "$ONNX_PATH" ]; then
            if is_classification_model "$MODEL"; then
                python3 convert/export_onnx.py --model "$MODEL" --output-dir "$MODEL_DIR"
                ONNX_PATH="$MODEL_DIR/${MODEL}.onnx"
            else
                echo "  [WARN] Detection ONNX not found: ${MODEL}.onnx or ${MODEL}_raw.onnx"
                echo "  [WARN] Skipping conversion for $MODEL (provide pre-built ONNX)"
                continue
            fi
        else
            echo "  ONNX already exists: $(basename "$ONNX_PATH")"
        fi

        # TensorRT engines
        if [ ! -f "$MODEL_DIR/${MODEL}_fp32.engine" ]; then
            bash convert/build_trt.sh "$ONNX_PATH" || {
                echo "  [WARN] TRT build failed. Continuing..."
            }
            # Rename engines if built from a differently-named ONNX (e.g. _raw.onnx)
            ONNX_STEM="$(basename "$ONNX_PATH" .onnx)"
            if [ "$ONNX_STEM" != "$MODEL" ]; then
                for prec in fp32 fp16 int8; do
                    src="$MODEL_DIR/${ONNX_STEM}_${prec}.engine"
                    dst="$MODEL_DIR/${MODEL}_${prec}.engine"
                    if [ -f "$src" ] && [ ! -f "$dst" ]; then
                        mv "$src" "$dst"
                        echo "  Renamed: $(basename "$src") -> $(basename "$dst")"
                    fi
                done
            fi
        else
            echo "  TRT engines already exist, skipping."
        fi

        # TFLite
        if [ ! -f "$MODEL_DIR/${MODEL}.tflite" ]; then
            python3 convert/convert_tflite.py --onnx-path "$ONNX_PATH" --output-dir "$MODEL_DIR" || {
                echo "  [WARN] TFLite conversion failed. Continuing..."
            }
        else
            echo "  TFLite already exists, skipping."
        fi

        # ncnn
        if [ ! -f "$MODEL_DIR/${MODEL}.ncnn.param" ] && [ ! -f "$MODEL_DIR/${MODEL}.param" ]; then
            bash convert/convert_ncnn.sh "$ONNX_PATH" || {
                echo "  [WARN] ncnn conversion failed. Continuing..."
            }
        else
            echo "  ncnn already exists, skipping."
        fi
    done
    echo ""
fi

# ============================================
# Step 2: Benchmark (Model × Power Mode)
# ============================================
echo "=== Step 2: Running Benchmarks ==="

for MODEL in "${MODELS[@]}"; do
    for POWER in "${POWER_MODES[@]}"; do
        echo ""
        echo "========================================"
        echo "  Benchmarking: $MODEL @ $POWER"
        echo "========================================"

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY] sudo nvpmodel -m $([ "$POWER" = "5w" ] && echo 1 || echo 0)"
            echo "  [DRY] sudo jetson_clocks"
            echo "  [DRY] sleep $THERMAL_WAIT"
            echo "  [DRY] python3 benchmark/benchmark.py --model $MODEL --power-mode $POWER"
            # Show which runtimes would be skipped
            for skip in "${SKIP_COMBOS[@]}"; do
                skip_model="${skip%%:*}"
                skip_rt="${skip##*:}"
                if [ "$skip_model" = "$MODEL" ]; then
                    echo "  [DRY] [SKIP] $skip_rt (known incompatible: $skip)"
                fi
            done
            continue
        fi

        # Set power mode
        if [ "$POWER" = "5w" ]; then
            sudo nvpmodel -m 1
        else
            sudo nvpmodel -m 0
        fi
        sudo jetson_clocks 2>/dev/null || true

        # Wait for thermal stabilization
        echo "  Waiting ${THERMAL_WAIT}s for thermal stabilization..."
        sleep "$THERMAL_WAIT"

        # Build runtime exclusion list for this model
        EXCLUDE_RUNTIMES=""
        for skip in "${SKIP_COMBOS[@]}"; do
            skip_model="${skip%%:*}"
            skip_rt="${skip##*:}"
            if [ "$skip_model" = "$MODEL" ]; then
                echo "  [SKIP] $skip_rt (known incompatible: $skip)"
                if [ -n "$EXCLUDE_RUNTIMES" ]; then
                    EXCLUDE_RUNTIMES="${EXCLUDE_RUNTIMES},$skip_rt"
                else
                    EXCLUDE_RUNTIMES="$skip_rt"
                fi
            fi
        done

        # Run benchmark (exclude known-bad runtimes via filtered list)
        if [ -n "$EXCLUDE_RUNTIMES" ]; then
            # Build runtime list excluding bad combos
            ALL_RT="pytorch_cpu,pytorch_cuda,tensorrt_fp32,tensorrt_fp16,tensorrt_int8,onnxrt_cuda,onnxrt_trt,tflite_cpu,tflite_gpu,ncnn_cpu,ncnn_vulkan"
            FILTERED_RT=""
            IFS=',' read -ra ALL_ARRAY <<< "$ALL_RT"
            IFS=',' read -ra EXCL_ARRAY <<< "$EXCLUDE_RUNTIMES"
            for rt in "${ALL_ARRAY[@]}"; do
                excluded=false
                for excl in "${EXCL_ARRAY[@]}"; do
                    if [ "$rt" = "$excl" ]; then
                        excluded=true
                        break
                    fi
                done
                if [ "$excluded" = false ]; then
                    if [ -n "$FILTERED_RT" ]; then
                        FILTERED_RT="${FILTERED_RT},$rt"
                    else
                        FILTERED_RT="$rt"
                    fi
                fi
            done

            python3 benchmark/benchmark.py \
                --model "$MODEL" \
                --runtimes "$FILTERED_RT" \
                --power-mode "$POWER" \
                --num-warmup "$NUM_WARMUP" \
                --num-runs "$NUM_RUNS" \
                --cool-down "$COOL_DOWN" \
                --model-dir "$MODEL_DIR" \
                --output-dir "$RESULTS_DIR"
        else
            python3 benchmark/benchmark.py \
                --model "$MODEL" \
                --power-mode "$POWER" \
                --num-warmup "$NUM_WARMUP" \
                --num-runs "$NUM_RUNS" \
                --cool-down "$COOL_DOWN" \
                --model-dir "$MODEL_DIR" \
                --output-dir "$RESULTS_DIR"
        fi

        echo "  Cooling down for ${COOL_DOWN}s..."
        sleep "$COOL_DOWN"
    done
done

# ============================================
# Step 3: Accuracy Evaluation (Classification)
# ============================================
echo ""
echo "=== Step 3: Accuracy Evaluation (Classification: Top-1/Top-5) ==="

if [ ! -d "$DATA_DIR" ]; then
    echo "  [SKIP] ImageNet validation data not found at $DATA_DIR"
    echo "  To run accuracy eval, place validation images in:"
    echo "    $DATA_DIR/val/<class_name>/<images>"
else
    for MODEL in "${MODELS[@]}"; do
        if ! is_classification_model "$MODEL"; then
            continue
        fi
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY] OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 benchmark/accuracy_eval.py --model $MODEL --data-dir $DATA_DIR"
        else
            OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            python3 benchmark/accuracy_eval.py \
                --model "$MODEL" \
                --data-dir "$DATA_DIR" \
                --model-dir "$MODEL_DIR" \
                --output-dir "$RESULTS_DIR"
        fi
    done
fi

# ============================================
# Step 3b: Accuracy Evaluation (Detection: mAP)
# ============================================
echo ""
echo "=== Step 3b: Accuracy Evaluation (Detection: mAP) ==="

HAS_DETECTION=false
for MODEL in "${MODELS[@]}"; do
    is_detection_model "$MODEL" && HAS_DETECTION=true
done

if [ "$HAS_DETECTION" = true ]; then
    if [ ! -d "$COCO_DIR" ]; then
        echo "  [SKIP] COCO validation data not found at $COCO_DIR"
        echo "  To run detection accuracy, place COCO val2017 images in:"
        echo "    $COCO_DIR/val2017/ and $COCO_DIR/annotations/"
    else
        if [ -d "$COCO_DIR/images" ]; then
            COCO_IMAGES_DIR="$COCO_DIR/images"
        elif [ -d "$COCO_DIR/val2017" ]; then
            COCO_IMAGES_DIR="$COCO_DIR/val2017"
        else
            COCO_IMAGES_DIR="$COCO_DIR"
        fi
        COCO_ANNOTATIONS="$COCO_DIR/annotations/instances_val2017_subset500.json"
        if [ ! -f "$COCO_ANNOTATIONS" ]; then
            echo "  [SKIP] COCO annotations not found at $COCO_ANNOTATIONS"
            echo "  To run detection accuracy, provide instances_val2017_subset500.json"
            echo "  under $COCO_DIR/annotations/"
        else
            echo "  COCO images:      $COCO_IMAGES_DIR"
            echo "  COCO annotations: $COCO_ANNOTATIONS"
        fi
        for MODEL in "${MODELS[@]}"; do
            if ! is_detection_model "$MODEL"; then
                continue
            fi
            if [ ! -f "$COCO_ANNOTATIONS" ]; then
                continue
            fi
            DET_OUTPUT="$RESULTS_DIR/detection_accuracy_${MODEL}.json"
            if [ "$DRY_RUN" = true ]; then
                echo "  [DRY] python3 benchmark/run_detection_accuracy.py --model $MODEL --coco-dir $COCO_IMAGES_DIR --annotations $COCO_ANNOTATIONS --output $DET_OUTPUT"
            else
                python3 benchmark/run_detection_accuracy.py \
                    --model "$MODEL" \
                    --coco-dir "$COCO_IMAGES_DIR" \
                    --annotations "$COCO_ANNOTATIONS" \
                    --model-dir "$MODEL_DIR" \
                    --output "$DET_OUTPUT" || {
                    echo "  [WARN] Detection accuracy eval failed for $MODEL"
                }
            fi
        done
    fi
else
    echo "  [SKIP] No detection models in current run"
fi

# ============================================
# Step 4: Layer Analysis
# ============================================
echo ""
echo "=== Step 4: Layer Analysis ==="

for MODEL in "${MODELS[@]}"; do
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY] python3 benchmark/layer_analyzer.py --model $MODEL"
    else
        python3 benchmark/layer_analyzer.py \
            --model "$MODEL" \
            --model-dir "$MODEL_DIR" \
            --output-dir "$RESULTS_DIR" || {
            echo "  [WARN] Layer analysis failed for $MODEL"
        }
    fi
done

# Optional detailed layer extraction:
# - architecture/*.csv: module shapes/params summary
# - layer_runtime/*.csv/json: runtime-aware layer/node/operator latency for 11 runtimes
# - layer_power/*.csv/json: optional slow layer energy profiling
# - layer_timelines/*.json: optional Chrome trace / TRT profile export
if [ "$DETAILED_LAYERS" = true ] || [ "$INCLUDE_LAYER_POWER" = true ] || [ "$INCLUDE_LAYER_TIMELINES" = true ]; then
    echo ""
    echo "=== Step 4b: Detailed Layer Extraction ==="
    DETAIL_ARGS=(
        --results-dir "$RESULTS_DIR/detailed_layers"
        --num-warmup "$LAYER_NUM_WARMUP"
        --num-runs "$LAYER_NUM_RUNS"
    )

    if [ -n "$SINGLE_MODEL" ]; then
        DETAIL_ARGS+=(--model "$SINGLE_MODEL")
    fi
    if [ "$DETAILED_LAYERS" != true ]; then
        DETAIL_ARGS+=(--skip-runtime)
    fi
    if [ "$INCLUDE_LAYER_POWER" = true ]; then
        DETAIL_ARGS+=(--include-power)
    fi
    if [ "$INCLUDE_LAYER_TIMELINES" = true ]; then
        DETAIL_ARGS+=(--include-timelines)
    fi
    if [ "$DRY_RUN" = true ]; then
        DETAIL_ARGS+=(--dry-run)
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY] bash run_detailed_layers.sh ${DETAIL_ARGS[*]}"
    fi
    MODELS_OVERRIDE="${MODELS[*]}" bash run_detailed_layers.sh "${DETAIL_ARGS[@]}" || {
        echo "  [WARN] Detailed layer extraction failed"
    }
fi

# ============================================
# Done
# ============================================
echo ""
echo "============================================"
echo "  All experiments complete!"
echo "============================================"
echo "  Models tested: ${MODELS[*]}"
echo "  Results saved in: $RESULTS_DIR"
echo ""
echo "  File listing:"
ls -lh "$RESULTS_DIR"/ 2>/dev/null || echo "  (no results yet)"
echo ""
echo "  Next steps:"
echo "  1. Review results CSV files"
echo "  2. Compare across models and power modes"
echo "  3. Share results with team"

# 자동 압축
echo ""
echo "📦 결과를 자동으로 압축합니다..."
TAR_NAME="${RESULTS_DIR%/}.tar.gz"
tar -czvf "$TAR_NAME" "$RESULTS_DIR" > /dev/null 2>&1
echo "✅ 압축 완료: $TAR_NAME (젯슨 나노에서 이 파일을 PC로 가져가세요!)"
