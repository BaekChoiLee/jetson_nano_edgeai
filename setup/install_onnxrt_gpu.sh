#!/bin/bash
# =============================================================================
# install_onnxrt_gpu.sh — JetPack 4.6.1 용 onnxruntime-gpu 설치
# =============================================================================
# JetPack 4.6.1 = L4T 32.6.1, CUDA 10.2, Python 3.8, aarch64
# 현재 설치: onnxruntime 1.19.2 (CPU only, CUDAExecutionProvider 없음)
# 목표: CUDA/TensorRT EP 지원 버전으로 교체
# =============================================================================
set -e

JETPACK_VERSION="461"  # 4.6.1
PYTHON_VERSION="cp38"
ARCH="linux_aarch64"

echo "=== onnxruntime-gpu 설치 (JetPack 4.6.1) ==="

# NVIDIA Jetson Zoo 경로 (공식 지원)
NVIDIA_REDIST="https://developer.download.nvidia.com/compute/redist/jp/v${JETPACK_VERSION}/onnxruntime/"

# 시도할 버전 (최신 → 이전 순)
VERSIONS=("1.12.1" "1.11.1" "1.10.0" "1.9.0")

WHEEL=""
for VER in "${VERSIONS[@]}"; do
    WNAME="onnxruntime_gpu-${VER}-${PYTHON_VERSION}-${PYTHON_VERSION}-${ARCH}.whl"
    URL="${NVIDIA_REDIST}${WNAME}"
    echo "시도: $URL"
    if curl -f -s --head "$URL" > /dev/null 2>&1; then
        echo "발견: $URL"
        WHEEL="$URL"
        break
    fi
done

if [ -z "$WHEEL" ]; then
    # GitHub releases fallback
    echo "NVIDIA Redist에서 찾지 못함, GitHub releases 시도..."
    for VER in "1.12.1" "1.11.1"; do
        WNAME="onnxruntime_gpu-${VER}-${PYTHON_VERSION}-${PYTHON_VERSION}-${ARCH}.whl"
        URL="https://github.com/microsoft/onnxruntime/releases/download/v${VER}/${WNAME}"
        if curl -f -s --head "$URL" > /dev/null 2>&1; then
            WHEEL="$URL"
            break
        fi
    done
fi

if [ -n "$WHEEL" ]; then
    echo "설치: $WHEEL"
    pip3 uninstall -y onnxruntime 2>/dev/null || true
    pip3 install "$WHEEL"
else
    echo "[ERROR] 사전 빌드 wheel을 찾을 수 없음."
    echo "소스 빌드가 필요합니다 (수 시간 소요)."
    echo "대안: setup/build_onnxrt_from_source.sh 실행"
    exit 1
fi

echo "=== 설치 확인 ==="
python3 -c "
import onnxruntime as ort
print('Version:', ort.__version__)
print('Providers:', ort.get_available_providers())
assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'CUDA EP 없음!'
print('OK: CUDAExecutionProvider 사용 가능')
"
