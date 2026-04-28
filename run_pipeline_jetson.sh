#!/bin/bash
# run_pipeline_jetson.sh - Jetson Nano에서 TensorRT 변환, 벤치마크, 시각화 수행
# 사용법: bash run_pipeline_jetson.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

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
    default_path="$3"
    env_value="$(eval "printf '%s' \"\${${env_name}:-}\"")"
    if [ -n "$env_value" ] && [ -x "$env_value" ]; then
        return 0
    fi
    if command -v "$executable" >/dev/null 2>&1; then
        return 0
    fi
    if [ -n "$default_path" ] && [ -x "$default_path" ]; then
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

    if [ "$missing_tflite" -eq 0 ] && has_tool "TFLITE_BENCHMARK_MODEL_PATH" "benchmark_model" "/usr/local/bin/benchmark_model"; then
        RUNTIMES="${RUNTIMES},tflite_cpu"
    else
        echo "  - [Skip] tflite_cpu: models/*.tflite 또는 benchmark_model이 준비되지 않았습니다."
    fi

    if [ "$missing_ncnn" -eq 0 ] && has_tool "BENCHNCNN_PATH" "benchncnn" "/usr/local/bin/benchncnn"; then
        RUNTIMES="${RUNTIMES},ncnn_vulkan"
    else
        echo "  - [Skip] ncnn_vulkan: models/*.param/*.bin 또는 benchncnn이 준비되지 않았습니다."
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
