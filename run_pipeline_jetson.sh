#!/bin/bash
# run_pipeline_jetson.sh - Jetson Nano에서 TensorRT 변환, 벤치마크, 시각화 수행
# 사용법: bash run_pipeline_jetson.sh

set -u

echo "====================================================="
echo "Jetson Nano Benchmark Pipeline"
echo "====================================================="

echo "[1/4] 데이터셋 준비 중..."
python3 scripts/prepare_dataset.py --data-dir ./data/imagenet_sample --num-samples 1000
if [ $? -ne 0 ]; then
    echo "[오류] 데이터셋 준비 실패"
    exit 1
fi
echo "[완료] 데이터셋 준비 성공"
echo "-----------------------------------------------------"

echo "[2/4] TensorRT Engine 변환 중..."
python3 scripts/convert_models.py --model-dir ./models --targets tensorrt
if [ $? -ne 0 ]; then
    echo "[오류] TensorRT 변환 단계 실패"
    exit 1
fi
echo "[완료] TensorRT 변환 단계 종료"
echo "-----------------------------------------------------"

echo "[3/4] 벤치마크 실행 중..."
python3 run_all.py --model all --runtime all --task cls --power-mode 10w
if [ $? -ne 0 ]; then
    echo "[오류] 벤치마크 실행 실패"
    exit 1
fi
echo "[완료] 벤치마크 실행 성공"
echo "-----------------------------------------------------"

echo "[4/4] 결과 시각화 처리 중..."
python3 scripts/visualize_results.py --results-dir ./results --output-dir ./results
if [ $? -ne 0 ]; then
    echo "[오류] 결과 시각화 실패"
    exit 1
fi
echo "[완료] 결과 시각화 성공"

echo "====================================================="
echo "Jetson 파이프라인 완료"
echo "결과는 ./results 폴더에 저장됩니다."
echo "====================================================="
