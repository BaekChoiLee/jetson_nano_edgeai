#!/bin/bash
# ============================================================================
# convert_ncnn.sh - ONNX/TorchScript -> NCNN (.param/.bin)
# ============================================================================

set -euo pipefail

ONNX_PATH="${1:?Usage: $0 <model.onnx>}"
MODEL_DIR="$(dirname "$ONNX_PATH")"
MODEL_NAME="$(basename "$ONNX_PATH" .onnx)"

PARAM_PATH="${MODEL_DIR}/${MODEL_NAME}.param"
BIN_PATH="${MODEL_DIR}/${MODEL_NAME}.bin"
LOG_PATH="${MODEL_DIR}/${MODEL_NAME}.ncnn.convert.log"
STATUS_PATH="${MODEL_DIR}/${MODEL_NAME}.ncnn.status.json"

NCNN_DIR="${NCNN_DIR:-$HOME/ncnn}"
ONNX2NCNN_CANDIDATES=(
    "${NCNN_DIR}/build/tools/onnx/onnx2ncnn"
    "${NCNN_DIR}/build/onnx2ncnn"
    "${NCNN_DIR}/build/tools/onnx2ncnn"
)
NCNNOPTIMIZE_CANDIDATES=(
    "${NCNN_DIR}/build/tools/ncnnoptimize"
    "${NCNN_DIR}/build/ncnnoptimize"
)
PNNX_CANDIDATES=(
    "$HOME/.local/bin/pnnx"
    "/usr/local/bin/pnnx"
)

ONNX2NCNN=""
NCNNOPTIMIZE=""
PNNX_BIN=""

write_status() {
    local ok="$1"
    local phase="$2"
    local reason="$3"
    python3 - "$STATUS_PATH" "$ok" "$phase" "$reason" "$PARAM_PATH" "$BIN_PATH" <<'PY'
import json, os, sys
path, ok, phase, reason, param_path, bin_path = sys.argv[1:]
payload = {
    "ok": ok == "1",
    "phase": phase,
    "reason": reason,
    "param_path": param_path,
    "bin_path": bin_path,
    "param_exists": os.path.exists(param_path),
    "param_size": os.path.getsize(param_path) if os.path.exists(param_path) else 0,
    "bin_exists": os.path.exists(bin_path),
    "bin_size": os.path.getsize(bin_path) if os.path.exists(bin_path) else 0,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
PY
}

pair_ready() {
    [ -s "$PARAM_PATH" ] && [ -s "$BIN_PATH" ]
}

smoke_ncnn() {
    local hw=224
    if [[ "$MODEL_NAME" == yolov8n* ]]; then
        hw=640
    elif [[ "$MODEL_NAME" == ssd_mobilenet_v2* ]]; then
        hw=320
    fi

    set +e
    python3 - "$PARAM_PATH" "$BIN_PATH" "$MODEL_NAME" "$hw" >>"$LOG_PATH" 2>&1 <<'PY'
import sys
import numpy as np
import ncnn

param_path, bin_path, _, hw = sys.argv[1:]
hw = int(hw)

net = ncnn.Net()
net.opt.use_vulkan_compute = False

if net.load_param(param_path) != 0:
    raise RuntimeError("load_param failed")
if net.load_model(bin_path) != 0:
    raise RuntimeError("load_model failed")

in_names = list(net.input_names())
out_names = list(net.output_names())
if not in_names or not out_names:
    raise RuntimeError("empty ncnn io names")

x = np.random.rand(3, hw, hw).astype(np.float32)
mat = ncnn.Mat(x).clone()

ex = net.create_extractor()
ex.input(in_names[0], mat)
ret, out = ex.extract(out_names[0])
if ret != 0:
    raise RuntimeError(f"extract failed ret={ret}")
arr = np.array(out)
print(f"[smoke] ok output_shape={arr.shape}")
PY
    local rc=$?
    set -e
    return $rc
}

for p in "${ONNX2NCNN_CANDIDATES[@]}"; do
    if [ -f "$p" ]; then
        ONNX2NCNN="$p"
        break
    fi
done
if [ -z "$ONNX2NCNN" ]; then
    ONNX2NCNN="$(command -v onnx2ncnn 2>/dev/null || true)"
fi

for p in "${NCNNOPTIMIZE_CANDIDATES[@]}"; do
    if [ -f "$p" ]; then
        NCNNOPTIMIZE="$p"
        break
    fi
done
if [ -z "$NCNNOPTIMIZE" ]; then
    NCNNOPTIMIZE="$(command -v ncnnoptimize 2>/dev/null || true)"
fi

for p in "${PNNX_CANDIDATES[@]}"; do
    if [ -f "$p" ]; then
        PNNX_BIN="$p"
        break
    fi
done
if [ -z "$PNNX_BIN" ]; then
    PNNX_BIN="$(command -v pnnx 2>/dev/null || true)"
fi

echo "=== Converting to ncnn ===" | tee "$LOG_PATH"
echo "  Input:  $ONNX_PATH" | tee -a "$LOG_PATH"
echo "  Output: $PARAM_PATH, $BIN_PATH" | tee -a "$LOG_PATH"

onnx_ok=0
if [ -n "$ONNX2NCNN" ] && [ -f "$ONNX2NCNN" ]; then
    echo "  [try] onnx2ncnn: $ONNX2NCNN" | tee -a "$LOG_PATH"
    set +e
    "$ONNX2NCNN" "$ONNX_PATH" "$PARAM_PATH" "$BIN_PATH" >>"$LOG_PATH" 2>&1
    rc=$?
    set -e
    if [ "$rc" -eq 0 ] && pair_ready; then
        onnx_ok=1
    else
        echo "  [warn] onnx2ncnn failed rc=$rc or empty artifacts" | tee -a "$LOG_PATH"
    fi
else
    echo "  [warn] onnx2ncnn not found" | tee -a "$LOG_PATH"
fi

if [ "$onnx_ok" -eq 1 ] && [ -n "$NCNNOPTIMIZE" ] && [ -f "$NCNNOPTIMIZE" ]; then
    OPT_PARAM="${MODEL_DIR}/${MODEL_NAME}_opt.param"
    OPT_BIN="${MODEL_DIR}/${MODEL_NAME}_opt.bin"
    set +e
    "$NCNNOPTIMIZE" "$PARAM_PATH" "$BIN_PATH" "$OPT_PARAM" "$OPT_BIN" 0 >>"$LOG_PATH" 2>&1
    rc=$?
    set -e
    if [ "$rc" -eq 0 ] && [ -s "$OPT_PARAM" ] && [ -s "$OPT_BIN" ]; then
        mv "$OPT_PARAM" "$PARAM_PATH"
        mv "$OPT_BIN" "$BIN_PATH"
        echo "  [ok] ncnnoptimize applied" | tee -a "$LOG_PATH"
    else
        echo "  [warn] ncnnoptimize failed rc=$rc (keeping original)" | tee -a "$LOG_PATH"
        rm -f "$OPT_PARAM" "$OPT_BIN"
    fi
fi

if [ "$onnx_ok" -eq 0 ] && [[ "$MODEL_NAME" == ssd_mobilenet_v2* ]] && [ -n "$PNNX_BIN" ]; then
    echo "  [fallback] try pnnx from TorchScript" | tee -a "$LOG_PATH"
    TS_CANDS=(
        "${MODEL_DIR}/ssd_mobilenet_v2.torchscript"
        "${MODEL_DIR}/${MODEL_NAME}.torchscript"
        "${MODEL_DIR}/ssd_mobilenet_v2_raw.torchscript"
    )
    TS_PATH=""
    for t in "${TS_CANDS[@]}"; do
        if [ -s "$t" ]; then
            TS_PATH="$t"
            break
        fi
    done

    if [ -n "$TS_PATH" ]; then
        TMP_PREFIX="${MODEL_DIR}/${MODEL_NAME}_pnnx_tmp"
        TMP_NCNN_PARAM="${TMP_PREFIX}.ncnn.param"
        TMP_NCNN_BIN="${TMP_PREFIX}.ncnn.bin"
        set +e
        "$PNNX_BIN" "$TS_PATH" \
            "inputshape=[1,3,320,320]" \
            "pnnxparam=${TMP_PREFIX}.pnnx.param" \
            "pnnxbin=${TMP_PREFIX}.pnnx.bin" \
            "pnnxpy=${TMP_PREFIX}_pnnx.py" \
            "ncnnparam=${TMP_NCNN_PARAM}" \
            "ncnnbin=${TMP_NCNN_BIN}" \
            "ncnnpy=${TMP_PREFIX}_ncnn.py" >>"$LOG_PATH" 2>&1
        rc=$?
        set -e
        if [ "$rc" -eq 0 ] && [ -s "$TMP_NCNN_PARAM" ] && [ -s "$TMP_NCNN_BIN" ]; then
            cp -f "$TMP_NCNN_PARAM" "$PARAM_PATH"
            cp -f "$TMP_NCNN_BIN" "$BIN_PATH"
            onnx_ok=1
            echo "  [ok] pnnx fallback artifact generated" | tee -a "$LOG_PATH"
        else
            echo "  [warn] pnnx fallback failed rc=$rc" | tee -a "$LOG_PATH"
        fi
    else
        echo "  [warn] TorchScript not found for pnnx fallback" | tee -a "$LOG_PATH"
    fi
fi

if pair_ready && [[ "$MODEL_NAME" == ssd_mobilenet_v2* ]]; then
    # SSD 경로 해석기 호환용 canonical 복사본
    cp -f "$PARAM_PATH" "${MODEL_DIR}/ssd_mobilenet_v2.ncnn.param" || true
    cp -f "$BIN_PATH" "${MODEL_DIR}/ssd_mobilenet_v2.ncnn.bin" || true
fi

if pair_ready; then
    if ! smoke_ncnn; then
        write_status 0 "smoke" "runtime_smoke_failed_or_segfault"
        echo "[ERROR] ncnn smoke test failed. log: $LOG_PATH" >&2
        exit 1
    fi

    write_status 1 "convert" "ok"
    echo ""
    echo "=== ncnn Model Summary ==="
    echo "  $(basename "$PARAM_PATH"): $(du -h "$PARAM_PATH" | cut -f1)"
    echo "  $(basename "$BIN_PATH"):   $(du -h "$BIN_PATH" | cut -f1)"
    exit 0
fi

write_status 0 "convert" "artifact_not_generated"
echo "[ERROR] ncnn conversion failed. log: $LOG_PATH" >&2
exit 1
