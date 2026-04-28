#!/bin/bash
# run_pipeline_mac.sh - MacBook에서 모델 export 및 비-TensorRT 변환만 수행
# 사용법: bash run_pipeline_mac.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x "$SCRIPT_DIR/.venv-convert/bin/python" ]; then
    PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv-convert/bin/python}"
    if [ "$PYTHON_BIN" = "python3" ]; then
        PYTHON_BIN="$SCRIPT_DIR/.venv-convert/bin/python"
    fi
    export PATH="$SCRIPT_DIR/.venv-convert/bin:$PATH"
fi

if [ -z "${ONNX2NCNN_PATH:-}" ] && [ -x "$SCRIPT_DIR/.tools/onnx2ncnn" ]; then
    export ONNX2NCNN_PATH="$SCRIPT_DIR/.tools/onnx2ncnn"
fi

echo "====================================================="
echo "Mac Model Export/Conversion Pipeline"
echo "====================================================="

echo "[1/3] 분류 모델 ONNX Export 중..."
"$PYTHON_BIN" scripts/export_models.py --output-dir ./models
if [ $? -ne 0 ]; then
    echo "[오류] 모델 Export 실패"
    exit 1
fi
echo "[완료] 모델 Export 성공"
echo "-----------------------------------------------------"

echo "[2/3] ONNX -> TFLite/ncnn 변환 중..."
"$PYTHON_BIN" scripts/convert_models.py --model-dir ./models --targets tflite ncnn --strict
if [ $? -ne 0 ]; then
    echo "[오류] 모델 변환 단계 실패"
    echo "필요 도구를 설치한 뒤 다시 실행하세요:"
    echo "  brew install git-lfs ncnn"
    echo "  pip install tensorflow tf-keras onnx onnx-tf onnx2tf"
    echo "onnx2ncnn이 PATH에 없으면 ONNX2NCNN_PATH=/path/to/onnx2ncnn 를 지정하세요."
    exit 1
fi
echo "[완료] Mac 변환 단계 종료"
echo "-----------------------------------------------------"

echo "[3/3] Git LFS 추적 상태 확인"
if command -v git-lfs >/dev/null 2>&1; then
    git lfs track "models/*.onnx" "models/*.onnx.data" "models/*.tflite" "models/*.param" "models/*.bin" "models/*.engine" "models/*.trt"
    echo "[완료] Git LFS track 설정 확인"
else
    echo "[오류] git-lfs가 설치되어 있지 않습니다. 대용량 모델 파일 push 전에 설치하세요."
    exit 1
fi

echo "====================================================="
echo "Mac 단계 완료"
echo "다음 파일을 commit/push 하세요: .gitattributes, scripts/, run_pipeline_*.sh, models/*"
echo "Jetson에서는 git pull 후 bash run_pipeline_jetson.sh 를 실행하세요."
echo "====================================================="
