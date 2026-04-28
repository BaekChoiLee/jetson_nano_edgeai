#!/bin/bash
# run_pipeline_jetson.sh - Jetson Nano에서 TensorRT 변환, 벤치마크, 시각화 수행
# 사용법: bash run_pipeline_jetson.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

first_executable() {
    for candidate in "$@"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

if [ -z "${BENCHNCNN_PATH:-}" ]; then
    BENCHNCNN_PATH="$(first_executable \
        "$SCRIPT_DIR/build_ncnn/benchmark/benchncnn" \
        "$SCRIPT_DIR/build_ncnn/benchncnn" \
        "$SCRIPT_DIR/benchncnn" \
        "/usr/local/bin/benchncnn" \
        "/usr/bin/benchncnn" || true)"
    if [ -n "$BENCHNCNN_PATH" ]; then
        export BENCHNCNN_PATH
    fi
fi

if [ -z "${TFLITE_BENCHMARK_MODEL_PATH:-}" ]; then
    TFLITE_BENCHMARK_MODEL_PATH="$(first_executable \
        "$SCRIPT_DIR/benchmark_model" \
        "$SCRIPT_DIR/build_tflite/tools/benchmark/benchmark_model" \
        "$SCRIPT_DIR/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model" \
        "/usr/local/bin/benchmark_model" \
        "/usr/bin/benchmark_model" || true)"
    if [ -n "$TFLITE_BENCHMARK_MODEL_PATH" ]; then
        export TFLITE_BENCHMARK_MODEL_PATH
    fi
fi

echo "====================================================="
echo "Jetson Nano Benchmark Pipeline"
echo "====================================================="

echo "[1/4] 데이터셋 준비 중..."
"$PYTHON_BIN" scripts/prepare_dataset.py --data-dir ./data/imagenet_sample --num-samples 1000
if [ $? -ne 0 ]; then
    echo "[오류] 데이터셋 준비 실패"
    exit 1
fi
echo "[완료] 데이터셋 준비 성공"
echo "-----------------------------------------------------"

echo "[2/4] TensorRT Engine 변환 중..."
"$PYTHON_BIN" scripts/convert_models.py --model-dir ./models --targets tensorrt
if [ $? -ne 0 ]; then
    echo "[오류] TensorRT 변환 단계 실패"
    exit 1
fi
echo "[완료] TensorRT 변환 단계 종료"
echo "-----------------------------------------------------"

echo "[3/4] 벤치마크 실행 중..."
has_tool() {
    env_name="$1"
    executable="$2"
    env_value="$(eval "printf '%s' \"\${${env_name}:-}\"")"
    if [ -n "$env_value" ] && [ -x "$env_value" ]; then
        return 0
    fi
    if command -v "$executable" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

DEFAULT_RUNTIMES="pytorch_cuda,tensorrt_fp32,tensorrt_fp16,tensorrt_int8,onnxrt_cpu"
if [ -z "${RUNTIMES:-}" ]; then
    RUNTIMES="${DEFAULT_RUNTIMES}"

    missing_tflite=0
    missing_ncnn=0
    for model in mobilenetv3s efficientnetb0 shufflenetv2 resnet50; do
        if [ ! -f "./models/${model}.tflite" ]; then
            missing_tflite=1
        fi
        if [ ! -f "./models/${model}.param" ] || [ ! -f "./models/${model}.bin" ]; then
            missing_ncnn=1
        fi
    done

    if [ "$missing_tflite" -eq 0 ] && has_tool "TFLITE_BENCHMARK_MODEL_PATH" "benchmark_model"; then
        RUNTIMES="${RUNTIMES},tflite_cpu"
    else
        if [ "$missing_tflite" -ne 0 ]; then
            echo "  - [Skip] tflite_cpu: models/*.tflite 파일이 부족합니다."
        else
            echo "  - [Skip] tflite_cpu: benchmark_model 실행 파일을 찾지 못했습니다."
            echo "    TFLITE_BENCHMARK_MODEL_PATH=/path/to/benchmark_model 로 지정할 수 있습니다."
        fi
    fi

    if [ "$missing_ncnn" -eq 0 ] && has_tool "BENCHNCNN_PATH" "benchncnn"; then
        RUNTIMES="${RUNTIMES},ncnn_vulkan"
    else
        if [ "$missing_ncnn" -ne 0 ]; then
            echo "  - [Skip] ncnn_vulkan: models/*.param 또는 models/*.bin 파일이 부족합니다."
        else
            echo "  - [Skip] ncnn_vulkan: benchncnn 실행 파일을 찾지 못했습니다."
            echo "    BENCHNCNN_PATH=/path/to/benchncnn 로 지정할 수 있습니다."
        fi
    fi
fi
RUNS="${RUNS:-100}"
WARMUP="${WARMUP:-10}"
COOLDOWN="${COOLDOWN:-10}"
echo "  - Runtimes: ${RUNTIMES}"
echo "  - Runs/Warmup/Cooldown: ${RUNS}/${WARMUP}/${COOLDOWN}"
"$PYTHON_BIN" run_all.py --model all --runtime "${RUNTIMES}" --task cls --power-mode 10w --runs "${RUNS}" --warmup "${WARMUP}" --cooldown "${COOLDOWN}"
if [ $? -ne 0 ]; then
    echo "[오류] 벤치마크 실행 실패"
    exit 1
fi
echo "[완료] 벤치마크 실행 성공"
echo "-----------------------------------------------------"

echo "[4/4] 결과 시각화 처리 중..."
"$PYTHON_BIN" scripts/visualize_results.py --results-dir ./results --output-dir ./results
if [ $? -ne 0 ]; then
    echo "[오류] 결과 시각화 실패"
    exit 1
fi
echo "[완료] 결과 시각화 성공"

echo "====================================================="
echo "Jetson 파이프라인 완료"
echo "결과는 ./results 폴더에 저장됩니다."
echo "====================================================="
