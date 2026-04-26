#!/bin/bash
# setup.sh - Jetson Nano Edge AI 1주차 환경 구축 스크립트
# 사용법: bash setup.sh

echo "====================================================="
echo "Jetson Nano Edge AI Environment Setup Script"
echo "====================================================="

# 1. 시스템 패키지 업데이트 및 기본 도구 설치
echo "[1/5] 시스템 패키지 업데이트 및 기본 도구 설치..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev libjpeg-dev zlib1g-dev \
    libpython3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev \
    gfortran cmake build-essential g++ jq htop

# 2. Python 라이브러리 (CPU & 툴) 설치
echo "[2/5] Python 필수 라이브러리 설치 (ONNX Runtime, codecarbon 등)..."
python3 -m pip install -U pip
python3 -m pip install onnx onnxruntime  # GPU 버전 링크 만료로 CPU 사용
python3 -m pip install codecarbon pandas matplotlib seaborn jupyterlab tqdm pycocotools

# 3. TFLite Runtime 설치
echo "[3/5] TFLite Runtime 설치..."
# Jetson Nano(AArch64)용 tflite_runtime
python3 -m pip install tflite_runtime

# 4. ncnn 빌드 및 설치
echo "[4/5] ncnn 소스 빌드 (Vulkan 지원)..."
mkdir -p build_ncnn && cd build_ncnn
if [ ! -d "ncnn" ]; then
    git clone https://github.com/Tencent/ncnn.git
    cd ncnn
    git submodule update --init
    mkdir -p build && cd build
    cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/jetson.toolchain.cmake \
          -DNCNN_VULKAN=ON \
          -DNCNN_PYTHON=ON \
          -DNCNN_BUILD_EXAMPLES=OFF ..
    make -j4
    make install
    cd ../../../
else
    echo "ncnn directory already exists. Skipping clone."
    cd ../
fi

# 5. tegrastats 실행 권한 확인
echo "[5/5] tegrastats 권한 설정..."
# tegrastats 실행은 sudo 권한이 필요할 수 있으므로 메시지로 알림
echo "tegrastats는 비밀번호 없이 sudo로 실행되거나 권한 부여가 필요할 수 있습니다."

echo "====================================================="
echo "Setup Complete!"
echo "PyTorch와 TensorRT는 JetPack 4.x 이미지에 기본 포함된 버전을 사용하거나,"
echo "NVIDIA 공식 포럼의 설치 가이드를 참조하세요."
echo "====================================================="
