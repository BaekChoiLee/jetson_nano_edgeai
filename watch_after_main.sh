#!/bin/bash
set -uo pipefail

# Wait until rebenchmark_clean_v2.sh fully exits, then run isolated layer pass.
# This avoids self-matching pgrep loops (infinite wait).

cd "$(dirname "$0")"

RUN="${1:-}"
LOG="${2:-results/layer_after_main_$(date +%Y%m%d_%H%M%S).log}"
MAKE_JOBS="${MAKE_JOBS:-2}"

if [ -z "$RUN" ]; then
    echo "Usage: bash watch_after_main.sh <RESULTS_ROOT> [LOG_PATH]"
    exit 1
fi

{
    echo "[watch-main] start $(date)"
    echo "[watch-main] run=$RUN"
    while true; do
        if pgrep -f "[r]ebenchmark_clean_v2.sh" >/dev/null 2>&1; then
            echo "[watch-main] benchmark running... $(date)"
            sleep 60
            continue
        fi
        break
    done

    echo "[watch-main] benchmark stopped, start layer tool build $(date)"
    MAKE_JOBS="$MAKE_JOBS" bash setup/build_layer_tools.sh || echo "[watch-main] tool build failed"

    export NCNN_BENCHNCNN_BIN="${NCNN_BENCHNCNN_BIN:-$HOME/ncnn/build_bench/benchmark/benchncnn}"
    export TFLITE_BENCHMARK_MODEL_BIN="${TFLITE_BENCHMARK_MODEL_BIN:-$HOME/tensorflow-2.13.0/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model}"

    echo "[watch-main] run isolated layer benchmark $(date)"
    bash run_layer_runtime_tflite_ncnn_isolated.sh "$RUN" || echo "[watch-main] isolated layer failed"

    echo "[watch-main] done $(date)"
} >> "$LOG" 2>&1

echo "[watch-main] log: $LOG"
