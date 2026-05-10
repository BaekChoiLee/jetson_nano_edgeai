#!/bin/bash
# Install ONNX Runtime with CUDA/TensorRT EP on Jetson Nano
set -e

echo "=== Installing ONNX Runtime ==="

# Method 1: Pre-built wheel from NVIDIA (recommended)
echo "[1] Trying pre-built wheel for JetPack 4.x..."

# NVIDIA provides onnxruntime-gpu wheels for Jetson
# Check https://elinux.org/Jetson_Zoo for latest URLs
ONNXRT_VERSION="1.10.0"
PYTHON_VER="cp38"  # Q-engineering image uses Python 3.8

# Try NVIDIA's Jetson wheel
pip3 install onnxruntime-gpu==1.10.0 2>/dev/null && {
    echo "  Installed via pip (onnxruntime-gpu)."
} || {
    echo "  pip install failed, trying NVIDIA Jetson wheel..."

    # Download from NVIDIA Jetson Zoo
    WHEEL_URL="https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.whl"
    WHEEL_NAME="onnxruntime_gpu-${ONNXRT_VERSION}-${PYTHON_VER}-${PYTHON_VER}-linux_aarch64.whl"

    wget -q -O "$WHEEL_NAME" "$WHEEL_URL" 2>/dev/null && {
        pip3 install "$WHEEL_NAME"
        rm -f "$WHEEL_NAME"
        echo "  Installed from NVIDIA Jetson wheel."
    } || {
        echo ""
        echo "  ====================================="
        echo "  Pre-built wheel not available."
        echo "  Building from source is required."
        echo "  ====================================="
        echo ""
        echo "  Source build instructions:"
        echo "  git clone --recursive https://github.com/microsoft/onnxruntime"
        echo "  cd onnxruntime"
        echo "  ./build.sh --config Release \\"
        echo "    --use_cuda --cuda_home /usr/local/cuda \\"
        echo "    --cudnn_home /usr/lib/aarch64-linux-gnu \\"
        echo "    --use_tensorrt --tensorrt_home /usr/lib/aarch64-linux-gnu \\"
        echo "    --build_wheel --skip_tests \\"
        echo "    --parallel 2"
        echo ""
        echo "  NOTE: Source build takes 4-6 hours on Jetson Nano."
        echo "  Consider building on a more powerful machine and copying."
        exit 1
    }
}

# Verify installation
echo ""
echo "[2] Verifying ONNX Runtime..."
python3 -c "
import onnxruntime as ort
print(f'  ONNX Runtime version: {ort.__version__}')
providers = ort.get_available_providers()
print(f'  Available providers: {providers}')
if 'CUDAExecutionProvider' in providers:
    print('  [OK] CUDA EP available')
else:
    print('  [WARN] CUDA EP not available')
if 'TensorrtExecutionProvider' in providers:
    print('  [OK] TensorRT EP available')
else:
    print('  [INFO] TensorRT EP not available (optional)')
"
echo "=== ONNX Runtime installation complete ==="
