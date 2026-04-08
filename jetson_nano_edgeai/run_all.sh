#!/bin/bash
# Full Jetson Nano benchmark automation script
# Usage: ./run_all.sh [--model MODEL] [--skip-convert] [--skip-setup] [--dry-run]
set -e

# ============================================
# Configuration
# ============================================
# MODELS=("mobilenetv3_small" "resnet50" "shufflenet_v2" "ssd_mobilenet_v2")  # 전체 모델
MODELS=("shufflenet_v2" "ssd_mobilenet_v2")
POWER_MODES=("10w" "5w")
COOL_DOWN=60          # seconds between experiments
NUM_WARMUP=10
NUM_RUNS=100
THERMAL_WAIT=30       # seconds to wait after power mode change
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${PROJECT_DIR}/models"
RESULTS_DIR="${PROJECT_DIR}/results"
DATA_DIR="${PROJECT_DIR}/data/imagenet_val"
COCO_DATA_DIR="${PROJECT_DIR}/data/coco_val"

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
        --dry-run)     DRY_RUN=true; shift;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL     Run only this model (mobilenetv3_small, resnet50, shufflenet_v2, ssd_mobilenet_v2)"
            echo "  --week N          Run models assigned to week N (3, 4, 5, or 6)"
            echo "  --skip-convert    Skip model conversion step"
            echo "  --skip-setup      Skip environment verification"
            echo "  --dry-run         Print what would be done without executing"
            echo "  --help            Show this help"
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
        MODELS=("efficientnet_b0" "shufflenet_v2")
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
echo "  Jetson Nano Benchmark Suite"
echo "============================================"
echo "  Models:      ${MODELS[*]}"
echo "  Power modes: ${POWER_MODES[*]}"
echo "  Warmup:      ${NUM_WARMUP}"
echo "  Runs:        ${NUM_RUNS}"
echo "  Cool-down:   ${COOL_DOWN}s"
echo "  Project:     ${PROJECT_DIR}"
echo "  Dry run:     ${DRY_RUN}"
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

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY] python3 convert/export_onnx.py --model $MODEL --output-dir $MODEL_DIR"
            echo "  [DRY] bash convert/build_trt.sh $MODEL_DIR/${MODEL}.onnx"
            echo "  [DRY] python3 convert/convert_tflite.py --onnx-path $MODEL_DIR/${MODEL}.onnx"
            echo "  [DRY] bash convert/convert_ncnn.sh $MODEL_DIR/${MODEL}.onnx"
            continue
        fi

        # ONNX export
        if [ ! -f "$MODEL_DIR/${MODEL}.onnx" ]; then
            python3 convert/export_onnx.py --model "$MODEL" --output-dir "$MODEL_DIR"
        else
            echo "  ONNX already exists, skipping."
        fi

        # TensorRT engines
        if [ ! -f "$MODEL_DIR/${MODEL}_fp32.engine" ]; then
            bash convert/build_trt.sh "$MODEL_DIR/${MODEL}.onnx"
        else
            echo "  TRT engines already exist, skipping."
        fi

        # TFLite
        if [ ! -f "$MODEL_DIR/${MODEL}.tflite" ]; then
            python3 convert/convert_tflite.py --onnx-path "$MODEL_DIR/${MODEL}.onnx" --output-dir "$MODEL_DIR" || {
                echo "  [WARN] TFLite conversion failed. Continuing..."
            }
        else
            echo "  TFLite already exists, skipping."
        fi

        # ncnn
        if [ ! -f "$MODEL_DIR/${MODEL}.param" ]; then
            bash convert/convert_ncnn.sh "$MODEL_DIR/${MODEL}.onnx" || {
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

        # Run benchmark
        python3 benchmark/benchmark.py \
            --model "$MODEL" \
            --power-mode "$POWER" \
            --num-warmup "$NUM_WARMUP" \
            --num-runs "$NUM_RUNS" \
            --cool-down "$COOL_DOWN" \
            --model-dir "$MODEL_DIR" \
            --output-dir "$RESULTS_DIR"

        echo "  Cooling down for ${COOL_DOWN}s..."
        sleep "$COOL_DOWN"
    done
done

# ============================================
# Step 3: Accuracy Evaluation
# ============================================
echo ""
echo "=== Step 3: Accuracy Evaluation ==="

if [ ! -d "$DATA_DIR" ]; then
    echo "  [SKIP] ImageNet validation data not found at $DATA_DIR"
    echo "  To run accuracy eval, place validation images in:"
    echo "    $DATA_DIR/val/<class_name>/<images>"
else
    for MODEL in "${MODELS[@]}"; do
        # Detection models use COCO dataset
        if [ "$MODEL" = "ssd_mobilenet_v2" ]; then
            EVAL_DATA_DIR="$COCO_DATA_DIR"
        else
            EVAL_DATA_DIR="$DATA_DIR"
        fi

        if [ ! -d "$EVAL_DATA_DIR" ]; then
            echo "  [SKIP] Data not found for $MODEL at $EVAL_DATA_DIR"
            continue
        fi

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY] python3 benchmark/accuracy_eval.py --model $MODEL --data-dir $EVAL_DATA_DIR"
        else
            python3 benchmark/accuracy_eval.py \
                --model "$MODEL" \
                --data-dir "$EVAL_DATA_DIR" \
                --model-dir "$MODEL_DIR" \
                --output-dir "$RESULTS_DIR"
        fi
    done
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
            --output-dir "$RESULTS_DIR" || {
            echo "  [WARN] Layer analysis failed for $MODEL"
        }
    fi
done

# ============================================
# Done
# ============================================
echo ""
echo "============================================"
echo "  All experiments complete!"
echo "============================================"
echo "  Results saved in: $RESULTS_DIR"
echo ""
echo "  File listing:"
ls -lh "$RESULTS_DIR"/ 2>/dev/null || echo "  (no results yet)"
echo ""
echo "  Next steps:"
echo "  1. Review results CSV files"
echo "  2. Compare across models and power modes"
echo "  3. Share results with team"
