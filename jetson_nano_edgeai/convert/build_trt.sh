#!/bin/bash
# Convert ONNX model to TensorRT engines (FP32, FP16, INT8)
set -e

ONNX_PATH="${1:?Usage: $0 <model.onnx> [--workspace-size MB]}"
WORKSPACE_SIZE="${3:-1024}"  # Default 1024 MB

# Parse optional args
while [[ $# -gt 1 ]]; do
    case "$2" in
        --workspace-size) WORKSPACE_SIZE="$3"; shift 2;;
        *) shift;;
    esac
done

# Derive names from ONNX path
MODEL_DIR="$(dirname "$ONNX_PATH")"
MODEL_NAME="$(basename "$ONNX_PATH" .onnx)"

# Find trtexec
TRTEXEC=$(which trtexec 2>/dev/null || echo "/usr/src/tensorrt/bin/trtexec")
if [ ! -f "$TRTEXEC" ]; then
    echo "[ERROR] trtexec not found!"
    echo "  Expected at: /usr/src/tensorrt/bin/trtexec"
    echo "  Or in PATH."
    exit 1
fi
echo "Using trtexec: $TRTEXEC"
echo "ONNX model: $ONNX_PATH"
echo "Workspace: ${WORKSPACE_SIZE} MB"
echo ""

# FP32 Engine
echo "=== Building FP32 Engine ==="
FP32_ENGINE="${MODEL_DIR}/${MODEL_NAME}_fp32.engine"
$TRTEXEC \
    --onnx="$ONNX_PATH" \
    --saveEngine="$FP32_ENGINE" \
    --workspace="$WORKSPACE_SIZE" \
    --verbose 2>&1 | tail -5
echo "  Saved: $FP32_ENGINE"
echo ""

# FP16 Engine
echo "=== Building FP16 Engine ==="
FP16_ENGINE="${MODEL_DIR}/${MODEL_NAME}_fp16.engine"
$TRTEXEC \
    --onnx="$ONNX_PATH" \
    --fp16 \
    --saveEngine="$FP16_ENGINE" \
    --workspace="$WORKSPACE_SIZE" \
    --verbose 2>&1 | tail -5
echo "  Saved: $FP16_ENGINE"
echo ""

# INT8 Engine
echo "=== Building INT8 Engine ==="
INT8_ENGINE="${MODEL_DIR}/${MODEL_NAME}_int8.engine"
echo "  NOTE: INT8 requires calibration data for best accuracy."
echo "  Using --best flag as fallback (mixed precision)."
echo "  For proper INT8, provide --calib with calibration cache."
$TRTEXEC \
    --onnx="$ONNX_PATH" \
    --int8 \
    --saveEngine="$INT8_ENGINE" \
    --workspace="$WORKSPACE_SIZE" \
    --verbose 2>&1 | tail -5
echo "  Saved: $INT8_ENGINE"
echo ""

# Summary
echo "=== TensorRT Engine Summary ==="
echo "  Model: $MODEL_NAME"
for ENGINE in "$FP32_ENGINE" "$FP16_ENGINE" "$INT8_ENGINE"; do
    if [ -f "$ENGINE" ]; then
        SIZE=$(du -h "$ENGINE" | cut -f1)
        echo "  $(basename $ENGINE): $SIZE"
    else
        echo "  $(basename $ENGINE): FAILED"
    fi
done
echo ""
echo "To benchmark, use: python3 benchmark/run_tensorrt.py --engine <engine_path>"
