#!/bin/bash
set -uo pipefail

# Build helper tools required for tflite/ncnn layer-level profiling.
# - benchmark_model (TFLite op profiling)
# - benchncnn with NCNN_BENCHMARK=ON (ncnn per-layer timing)
#
# Usage:
#   bash setup/build_layer_tools.sh
# Optional env:
#   MAKE_JOBS=2

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/results/tool_build_logs"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_NCNN="${LOG_DIR}/build_ncnn_bench_${TS}.log"
LOG_TFLITE="${LOG_DIR}/build_tflite_benchmark_${TS}.log"

MAKE_JOBS="${MAKE_JOBS:-2}"
NCNN_SRC="${HOME}/ncnn"
NCNN_BUILD_BENCH="${NCNN_SRC}/build_bench"
TF_SRC="${HOME}/tensorflow-2.13.0"
TFLITE_BENCH_BIN="${TF_SRC}/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model"

echo "[tools] root=$ROOT_DIR"
echo "[tools] logs=$LOG_DIR"
echo "[tools] make_jobs=$MAKE_JOBS"

build_ncnn_bench() {
    echo "[ncnn] start -> $LOG_NCNN"
    if [ ! -d "$NCNN_SRC" ]; then
        echo "[ncnn] missing source: $NCNN_SRC" | tee -a "$LOG_NCNN"
        return 1
    fi
    mkdir -p "$NCNN_BUILD_BENCH"
    (
        set -e
        cd "$NCNN_BUILD_BENCH"
        cmake \
            -D CMAKE_BUILD_TYPE=Release \
            -D NCNN_VULKAN=ON \
            -D NCNN_BUILD_BENCHMARK=ON \
            -D NCNN_BUILD_TOOLS=ON \
            -D NCNN_BUILD_EXAMPLES=OFF \
            -D NCNN_BENCHMARK=ON \
            ..
        make -j"$MAKE_JOBS" benchncnn
    ) >>"$LOG_NCNN" 2>&1

    if [ -x "${NCNN_BUILD_BENCH}/benchmark/benchncnn" ]; then
        echo "[ncnn] OK: ${NCNN_BUILD_BENCH}/benchmark/benchncnn" | tee -a "$LOG_NCNN"
        return 0
    fi
    echo "[ncnn] FAIL: benchncnn not generated" | tee -a "$LOG_NCNN"
    return 1
}

build_tflite_benchmark_model() {
    echo "[tflite] start -> $LOG_TFLITE"
    if [ ! -d "$TF_SRC" ]; then
        echo "[tflite] missing source: $TF_SRC" | tee -a "$LOG_TFLITE"
        return 1
    fi
    (
        set -e
        cd "$TF_SRC"
        bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
    ) >>"$LOG_TFLITE" 2>&1

    if [ -x "$TFLITE_BENCH_BIN" ]; then
        echo "[tflite] OK: $TFLITE_BENCH_BIN" | tee -a "$LOG_TFLITE"
        return 0
    fi
    echo "[tflite] FAIL: benchmark_model not generated" | tee -a "$LOG_TFLITE"
    return 1
}

NCNN_RC=0
TFLITE_RC=0

build_ncnn_bench || NCNN_RC=$?
build_tflite_benchmark_model || TFLITE_RC=$?

echo ""
echo "========== TOOL BUILD SUMMARY =========="
echo "ncnn_rc=$NCNN_RC"
echo "tflite_rc=$TFLITE_RC"
echo "NCNN_BENCHNCNN_BIN=${NCNN_BUILD_BENCH}/benchmark/benchncnn"
echo "TFLITE_BENCHMARK_MODEL_BIN=$TFLITE_BENCH_BIN"
echo "log_ncnn=$LOG_NCNN"
echo "log_tflite=$LOG_TFLITE"
echo "======================================="

if [ "$NCNN_RC" -eq 0 ] && [ "$TFLITE_RC" -eq 0 ]; then
    exit 0
fi
exit 1
