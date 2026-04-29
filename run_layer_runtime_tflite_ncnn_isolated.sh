#!/bin/bash
set -uo pipefail

# Isolated layer-runtime sweep for tflite/ncnn only.
# Usage:
#   bash run_layer_runtime_tflite_ncnn_isolated.sh <RESULTS_ROOT>
# Example:
#   bash run_layer_runtime_tflite_ncnn_isolated.sh results/clean_v2_20260415_081254

cd "$(dirname "$0")"

RESULTS_ROOT="${1:-}"
if [ -z "$RESULTS_ROOT" ]; then
    echo "Usage: bash run_layer_runtime_tflite_ncnn_isolated.sh <RESULTS_ROOT>"
    exit 1
fi

MODEL_DIR="${MODEL_DIR:-models}"
NUM_WARMUP="${NUM_WARMUP:-10}"
NUM_RUNS="${NUM_RUNS:-50}"
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
    tflite_cpu
    tflite_gpu
    ncnn_cpu
    ncnn_vulkan
)

OUT_DIR="${RESULTS_ROOT}/layer_runtime"
mkdir -p "$OUT_DIR"

echo "[isolated-layer] results_root=$RESULTS_ROOT"
echo "[isolated-layer] model_dir=$MODEL_DIR"
echo "[isolated-layer] tflite_tool=$TFLITE_BENCHMARK_MODEL_BIN"
echo "[isolated-layer] ncnn_tool=$NCNN_BENCHNCNN_BIN"

for model in "${MODELS[@]}"; do
    for rt in "${RUNTIMES[@]}"; do
        out_json="${OUT_DIR}/${model}_${rt}.json"
        out_csv="${OUT_DIR}/${model}_${rt}.csv"
        echo "  [run] $model / $rt"
        python3 benchmark/layer_runtime_analyzer.py \
            --model "$model" \
            --runtime "$rt" \
            --model-dir "$MODEL_DIR" \
            --num-warmup "$NUM_WARMUP" \
            --num-runs "$NUM_RUNS" \
            --tflite-benchmark-bin "$TFLITE_BENCHMARK_MODEL_BIN" \
            --ncnn-benchmark-bin "$NCNN_BENCHNCNN_BIN" \
            --output-json "$out_json" \
            --output-csv "$out_csv" \
            || echo "    [fail] $model / $rt"
    done
done

echo "[isolated-layer] consolidate"
python3 analysis/consolidate_v2.py --results-dir "$RESULTS_ROOT" --output-dir "$RESULTS_ROOT" \
    || echo "[warn] consolidate failed"

echo "[isolated-layer] done"
