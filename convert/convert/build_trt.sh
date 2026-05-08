#!/bin/bash
# ============================================================================
# build_trt.sh - ONNX -> TensorRT engine builder (FP32/FP16/INT8)
# ============================================================================

set -euo pipefail

ONNX_PATH=""
WORKSPACE_SIZE=1024

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace-size)
            WORKSPACE_SIZE="${2:-1024}"
            shift 2
            ;;
        *)
            if [[ -z "$ONNX_PATH" ]]; then
                ONNX_PATH="$1"
                shift
            else
                echo "[ERROR] Unknown option: $1" >&2
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$ONNX_PATH" ]]; then
    echo "Usage: $0 <model.onnx> [--workspace-size MB]" >&2
    exit 1
fi
if [[ ! -f "$ONNX_PATH" ]]; then
    echo "[ERROR] ONNX not found: $ONNX_PATH" >&2
    exit 1
fi

MODEL_DIR="$(dirname "$ONNX_PATH")"
MODEL_NAME="$(basename "$ONNX_PATH" .onnx)"
ENGINE_STEM="$MODEL_NAME"
if [[ "$MODEL_NAME" == ssd_mobilenet_v2* ]]; then
    # SSD 변형(raw/fpinput 등)도 런타임 로더가 찾는 canonical 이름으로 저장
    ENGINE_STEM="ssd_mobilenet_v2"
fi

TRTEXEC="$(command -v trtexec 2>/dev/null || true)"
if [[ -z "$TRTEXEC" ]]; then
    TRTEXEC="/usr/src/tensorrt/bin/trtexec"
fi
if [[ ! -f "$TRTEXEC" ]]; then
    echo "[ERROR] trtexec not found." >&2
    exit 1
fi

INPUT_META="$(python3 - "$ONNX_PATH" <<'PY'
import onnx, sys
p = sys.argv[1]
m = onnx.load(p)
inp = m.graph.input[0]
print(inp.name)
print(int(inp.type.tensor_type.elem_type))
PY
)"
INPUT_NAME="$(echo "$INPUT_META" | sed -n '1p')"
INPUT_DTYPE="$(echo "$INPUT_META" | sed -n '2p')"

TRT_ONNX_PATH="$ONNX_PATH"
if [[ "$INPUT_DTYPE" == "2" ]]; then
    # TRT 8.0.1 parser는 UINT8 input 을 직접 받지 못하므로 input dtype을 FLOAT로 보정
    PATCHED_ONNX="${MODEL_DIR}/${MODEL_NAME}_fpinput.onnx"
    python3 - "$ONNX_PATH" "$PATCHED_ONNX" <<'PY'
import onnx, sys
from onnx import TensorProto
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
m.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
onnx.save(m, dst)
print(dst)
PY
    TRT_ONNX_PATH="$PATCHED_ONNX"
    echo "[INFO] Patched UINT8 input ONNX -> $TRT_ONNX_PATH"
fi

SHAPE_ARGS=()
if [[ "$ENGINE_STEM" == "ssd_mobilenet_v2" ]]; then
    # SSD raw ONNX는 dynamic NHWC 입력이라 shape를 고정해 1x1x1x3 자동강등을 방지
    SHAPE_ARGS=( "--shapes=${INPUT_NAME}:1x320x320x3" )
fi

echo "Using trtexec: $TRTEXEC"
echo "ONNX model: $TRT_ONNX_PATH"
echo "Workspace: ${WORKSPACE_SIZE} MB"
echo ""

build_one() {
    local precision="$1"
    shift
    local flags=("$@")
    local engine="${MODEL_DIR}/${ENGINE_STEM}_${precision}.engine"
    local log="${MODEL_DIR}/${ENGINE_STEM}_${precision}.trtexec.log"

    echo "=== Building ${precision^^} Engine ==="
    set +e
    "$TRTEXEC" \
        --onnx="$TRT_ONNX_PATH" \
        "${flags[@]}" \
        --saveEngine="$engine" \
        --workspace="$WORKSPACE_SIZE" \
        "${SHAPE_ARGS[@]}" \
        --verbose 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ $rc -ne 0 || ! -s "$engine" ]]; then
        echo "  [FAIL] ${precision^^} engine build failed (rc=$rc)"
        echo "  [FAIL] log: $log"
        return 1
    fi

    echo "  [OK] Saved: $engine"
    return 0
}

ok_fp32=0
ok_fp16=0
ok_int8=0

if build_one fp32; then ok_fp32=1; fi
if build_one fp16 --fp16; then ok_fp16=1; fi
if build_one int8 --int8; then ok_int8=1; fi

echo ""
echo "=== TensorRT Engine Summary ==="
echo "  Model: $ENGINE_STEM (from $(basename "$TRT_ONNX_PATH"))"
for precision in fp32 fp16 int8; do
    engine="${MODEL_DIR}/${ENGINE_STEM}_${precision}.engine"
    if [[ -s "$engine" ]]; then
        size="$(du -h "$engine" | cut -f1)"
        echo "  $(basename "$engine"): $size"
    else
        echo "  $(basename "$engine"): FAILED"
    fi
done

if [[ $ok_fp32 -eq 0 && $ok_fp16 -eq 0 && $ok_int8 -eq 0 ]]; then
    exit 1
fi
