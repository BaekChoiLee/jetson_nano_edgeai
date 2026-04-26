#!/bin/bash
# run_pipeline.sh - Jetson Nano Edge AI 전체 벤치마크 파이프라인 자동 실행 스크립트
# 사용법: bash run_pipeline.sh

echo "====================================================="
echo "Jetson Nano Edge AI Benchmark Pipeline"
echo "====================================================="

# 1. 데이터셋 준비 (파이프라인용 FakeData 생성)
echo "[1/3] 데이터셋 준비 중..."
python3 scripts/prepare_dataset.py --data-dir ./data/imagenet_sample --num-samples 1000
if [ $? -ne 0 ]; then
    echo "[오류] 데이터셋 준비 실패"
    exit 1
fi
echo "[완료] 데이터셋 준비 성공"
echo "-----------------------------------------------------"

# 2. 파이토치 원본 모델 다운로드 및 ONNX 내보내기
echo "[2/3] 분류 모델들 ONNX Export 중..."
python3 scripts/export_models.py --output-dir ./models
if [ $? -ne 0 ]; then
    echo "[오류] 모델 Export 실패"
    exit 1
fi
echo "[완료] 모델 Export 성공"
echo "-----------------------------------------------------"

# 3. 전체 벤치마크 실행 (모든 모델, 모든 런타임 대상)
echo "[3/3] 통합 벤치마크 파이프라인(run_all.py) 실행 중..."
# 기본적으로 10w 모드, 모든 모델(분류/탐지 포함), 모든 런타임 대상
# 필요 시 파라미터를 수정하여 실행하세요.
python3 run_all.py --model all --runtime all --task all --power-mode 10w
if [ $? -ne 0 ]; then
    echo "[오류] 벤치마크 실행 실패"
    exit 1
fi

echo "====================================================="
echo "모든 파이프라인 실행이 완료되었습니다!"
echo "결과는 ./results 폴더에 저장되어 있습니다."
echo "====================================================="
