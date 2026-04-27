#!/bin/bash
# run_pipeline_mac.sh - MacBook에서 모델 export 및 비-TensorRT 변환만 수행
# 사용법: bash run_pipeline_mac.sh

set -u

echo "====================================================="
echo "Mac Model Export/Conversion Pipeline"
echo "====================================================="

echo "[1/3] 분류 모델 ONNX Export 중..."
python3 scripts/export_models.py --output-dir ./models
if [ $? -ne 0 ]; then
    echo "[오류] 모델 Export 실패"
    exit 1
fi
echo "[완료] 모델 Export 성공"
echo "-----------------------------------------------------"

echo "[2/3] ONNX -> TFLite/ncnn 변환 중..."
python3 scripts/convert_models.py --model-dir ./models --targets tflite ncnn
if [ $? -ne 0 ]; then
    echo "[오류] 모델 변환 단계 실패"
    exit 1
fi
echo "[완료] Mac 변환 단계 종료"
echo "-----------------------------------------------------"

echo "[3/3] Git LFS 추적 상태 확인"
if command -v git-lfs >/dev/null 2>&1; then
    git lfs track "models/*.onnx" "models/*.tflite" "models/*.param" "models/*.bin" "models/*.engine" "models/*.trt"
    echo "[완료] Git LFS track 설정 확인"
else
    echo "[경고] git-lfs가 설치되어 있지 않습니다. 대용량 모델 파일 push 전에 설치하세요."
fi

echo "====================================================="
echo "Mac 단계 완료"
echo "다음 파일을 commit/push 하세요: .gitattributes, scripts/, run_pipeline_*.sh, models/*"
echo "Jetson에서는 git pull 후 bash run_pipeline_jetson.sh 를 실행하세요."
echo "====================================================="
