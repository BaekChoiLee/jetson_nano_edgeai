#!/bin/bash
# Jetson Nano 타임라인 프로파일링 전용 스크립트
# 기존 벤치마크 결과에 전혀 영향을 주지 않고, results/layer_timelines/ 폴더에 크롬 트레이스(.json)만 저장합니다.

cd ~/jetson-benchmark

OUT_DIR="results/layer_timelines"
mkdir -p "$OUT_DIR"

MODELS=("mobilenetv3_small" "resnet50" "efficientnet_b0" "yolov8n" "shufflenet_v2_x1_0" "ssd_mobilenet_v2")

echo "=================================================="
echo "  [Timeline Profiling] Safe Extraction Started"
echo "  Output Directory: $OUT_DIR"
echo "=================================================="

# 1. TensorRT 타임라인 프로파일링 (trtexec 활용)
echo "[1/2] TensorRT Profiling (FP16 & INT8)"
for model in "${MODELS[@]}"; do
    echo "  -> Profiling TRT FP16: $model"
    if [ -f "models/${model}_fp16.engine" ]; then
        /usr/src/tensorrt/bin/trtexec --loadEngine=models/${model}_fp16.engine \
            --exportProfile=$OUT_DIR/${model}_trt_fp16_timeline.json --iterations=10 \
            > /dev/null 2>&1
    fi
    
    echo "  -> Profiling TRT INT8: $model"
    if [ -f "models/${model}_int8.engine" ]; then
        /usr/src/tensorrt/bin/trtexec --loadEngine=models/${model}_int8.engine \
            --exportProfile=$OUT_DIR/${model}_trt_int8_timeline.json --iterations=10 \
            > /dev/null 2>&1
    fi
done

# 2. PyTorch Chrome Trace 생성기 작성 (즉석 파이썬 스크립트 생성 후 실행)
echo "[2/2] PyTorch Profiling (Chrome Trace)"
cat << 'EOF' > benchmark/temp_pytorch_profiler.py
import torch
import torchvision.models as models
import json
import os
import argparse
from torch.profiler import profile, record_function, ProfilerActivity

def profile_model(model_name, out_path):
    print(f"    PyTorch Profiling: {model_name}...")
    
    # 1. 모델 로드 (가벼운 더미 구현)
    if model_name == "resnet50":
        model = models.resnet50().eval().cuda()
    elif model_name == "mobilenetv3_small":
        model = models.mobilenet_v3_small().eval().cuda()
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0().eval().cuda()
    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0().eval().cuda()
    else:
        # YOLO / SSD는 복잡하므로 여기선 제외하거나 간단한 백본만
        return
        
    inputs = torch.randn(1, 3, 224, 224).cuda()
    
    # Warmup
    with torch.inference_mode():
        for _ in range(3):
            model(inputs)
            
    # PyTorch Native Profiler 켜기 (타임라인 추적)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)
            
    # 크롬 트레이스 json 저장 (이게 진짜 워터폴 차트용!)
    prof.export_chrome_trace(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    profile_model(args.model, args.out)
EOF

for model in "${MODELS[@]}"; do
    if [[ "$model" != "yolov8n" && "$model" != "ssd_mobilenet_v2" ]]; then
        python3 benchmark/temp_pytorch_profiler.py --model $model --out "$OUT_DIR/${model}_pytorch_timeline.json"
    fi
done

# 정리
rm benchmark/temp_pytorch_profiler.py

echo "=================================================="
echo "  Profiling Complete!"
echo "  All timelines saved to $OUT_DIR"
echo "  These files can be dropped into chrome://tracing"
echo "=================================================="
