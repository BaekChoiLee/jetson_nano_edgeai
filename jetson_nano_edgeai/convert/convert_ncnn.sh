#!/bin/bash
# Convert ONNX model to ncnn format (.param + .bin)
set -e

ONNX_PATH="${1:?Usage: $0 <model.onnx>}"
MODEL_DIR="$(dirname "$ONNX_PATH")"
MODEL_NAME="$(basename "$ONNX_PATH" .onnx)"

# Find ncnn tools
NCNN_DIR="${NCNN_DIR:-$HOME/ncnn}"
ONNX2NCNN="${NCNN_DIR}/build/tools/onnx/onnx2ncnn"
NCNNOPTIMIZE="${NCNN_DIR}/build/tools/ncnnoptimize"

# Check for onnx2ncnn
if [ ! -f "$ONNX2NCNN" ]; then
    ONNX2NCNN=$(which onnx2ncnn 2>/dev/null || true)
    if [ -z "$ONNX2NCNN" ]; then
        echo "[ERROR] onnx2ncnn not found!"
        echo "  Set NCNN_DIR or add to PATH."
        echo "  Expected: \$NCNN_DIR/build/tools/onnx/onnx2ncnn"
        exit 1
    fi
fi

# Check for ncnnoptimize
if [ ! -f "$NCNNOPTIMIZE" ]; then
    NCNNOPTIMIZE=$(which ncnnoptimize 2>/dev/null || true)
fi

PARAM_PATH="${MODEL_DIR}/${MODEL_NAME}.param"
BIN_PATH="${MODEL_DIR}/${MODEL_NAME}.bin"

# Step 1: ONNX → ncnn
echo "=== Converting ONNX to ncnn ==="
echo "  Input:  $ONNX_PATH"
echo "  Output: $PARAM_PATH, $BIN_PATH"
"$ONNX2NCNN" "$ONNX_PATH" "$PARAM_PATH" "$BIN_PATH"
echo "  [OK] Conversion complete."

# Step 2: Optimize (if ncnnoptimize available)
if [ -n "$NCNNOPTIMIZE" ] && [ -f "$NCNNOPTIMIZE" ]; then
    OPT_PARAM="${MODEL_DIR}/${MODEL_NAME}_opt.param"
    OPT_BIN="${MODEL_DIR}/${MODEL_NAME}_opt.bin"
    echo ""
    echo "=== Optimizing ncnn model ==="
    "$NCNNOPTIMIZE" "$PARAM_PATH" "$BIN_PATH" "$OPT_PARAM" "$OPT_BIN" 0
    # Replace original with optimized
    mv "$OPT_PARAM" "$PARAM_PATH"
    mv "$OPT_BIN" "$BIN_PATH"
    echo "  [OK] Optimized model saved."
else
    echo ""
    echo "  [INFO] ncnnoptimize not found, skipping optimization."
fi

# Summary
echo ""
echo "=== ncnn Model Summary ==="
if [ -f "$PARAM_PATH" ]; then
    PARAM_SIZE=$(du -h "$PARAM_PATH" | cut -f1)
    echo "  ${MODEL_NAME}.param: $PARAM_SIZE"
fi
if [ -f "$BIN_PATH" ]; then
    BIN_SIZE=$(du -h "$BIN_PATH" | cut -f1)
    echo "  ${MODEL_NAME}.bin:   $BIN_SIZE"
fi
echo ""
echo "To benchmark:"
echo "  python3 benchmark/run_ncnn.py --param $PARAM_PATH --bin $BIN_PATH"
