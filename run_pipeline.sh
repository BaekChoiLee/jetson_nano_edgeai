#!/bin/bash
# run_pipeline.sh - Jetson Nano Edge AI 전체 벤치마크 파이프라인 자동 실행 스크립트
# 사용법: bash run_pipeline.sh

echo "====================================================="
echo "Jetson Nano Edge AI Benchmark Pipeline"
echo "====================================================="

# 1. 데이터셋 준비 (파이프라인용 FakeData 생성)
echo "[1/5] 데이터셋 준비 중..."
python3 scripts/prepare_dataset.py --data-dir ./data/imagenet_sample --num-samples 1000
if [ $? -ne 0 ]; then
    echo "[오류] 데이터셋 준비 실패"
    exit 1
fi
echo "[완료] 데이터셋 준비 성공"
echo "-----------------------------------------------------"

# 2. 파이토치 원본 모델 다운로드 및 ONNX 내보내기
echo "[2/5] 분류 모델들 ONNX Export 중..."
python3 scripts/export_models.py --output-dir ./models
if [ $? -ne 0 ]; then
    echo "[오류] 모델 Export 실패"
    exit 1
fi
echo "[완료] 모델 Export 성공"
echo "-----------------------------------------------------"

# 3. ncnn 모델 변환 (onnx2ncnn이 설치된 경우)
echo "[3/5] ncnn 모델 변환 확인 중..."
if command -v onnx2ncnn >/dev/null 2>&1; then
    for model in mobilenetv3s efficientnetb0 shufflenetv2 resnet50; do
        onnx_path="./models/${model}.onnx"
        param_path="./models/${model}.param"
        bin_path="./models/${model}.bin"
        if [ -f "$onnx_path" ]; then
            echo "  - Converting ${model}.onnx -> ${model}.param/.bin"
            onnx2ncnn "$onnx_path" "$param_path" "$bin_path"
            if [ $? -ne 0 ]; then
                echo "[경고] ${model} ncnn 변환 실패. ncnn_vulkan 런타임은 해당 모델에서 스킵될 수 있습니다."
            fi
        fi
    done
else
    echo "[경고] onnx2ncnn을 찾을 수 없습니다. ncnn_vulkan 레이어 분석은 .param/.bin 파일이 있을 때만 실행됩니다."
fi
echo "-----------------------------------------------------"

# 4. 전체 벤치마크 실행 (모든 모델, 모든 런타임 대상)
echo "[4/5] 통합 벤치마크 파이프라인(run_all.py) 실행 중..."
# 기본적으로 10w 모드, 모든 모델(분류/탐지 포함), 모든 런타임 대상
# 필요 시 파라미터를 수정하여 실행하세요.
python3 run_all.py --model all --runtime all --task all --power-mode 10w
if [ $? -ne 0 ]; then
    echo "[오류] 벤치마크 실행 실패"
    exit 1
fi
echo "[완료] 벤치마크 실행 성공"
echo "-----------------------------------------------------"

# 5. 결과 시각화
echo "[5/5] 결과 시각화 처리 중..."
python3 scripts/visualize_results.py --results-dir ./results --output-dir ./results
if [ $? -ne 0 ]; then
    echo "[오류] 결과 시각화 실패"
    exit 1
fi
echo "[완료] 결과 시각화 성공"

echo "====================================================="
echo "모든 파이프라인 실행이 완료되었습니다!"
echo "결과는 ./results 폴더에, 그래프는 ./results/figures 폴더에 저장되어 있습니다."
echo "====================================================="
