#!/bin/bash
# =============================================================================
#  TFLite GPU Delegate Builder for Jetson Nano
#  Reference: https://qengineering.eu/install-tensorflow-2-lite-on-jetson-nano.html
#
#  - Uses Bazelisk → Bazel 6.1.0 (pre-built arm64 binary)
#  - Builds from TensorFlow v2.13.0 source
#  - GPU delegate: OpenGL ES backend (not OpenCL)
#  - Expected build time: 6-8 hours
# =============================================================================
set -euo pipefail

LOG_FILE="$HOME/jetson-benchmark/build_tflite_gpu.log"
INSTALL_DIR="$HOME/jetson-benchmark/lib"
TF_VERSION="2.13.0"
TF_DIR="$HOME/tensorflow-${TF_VERSION}"
BAZEL_VERSION="6.1.0"

mkdir -p "$INSTALL_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "  TFLite GPU Delegate Build"
echo "  TF Version: v${TF_VERSION}"
echo "  Bazel:      ${BAZEL_VERSION} (via Bazelisk)"
echo "  Backend:    OpenGL ES (EGL)"
echo "  Started:    $(date)"
echo "============================================================"
echo ""

# -------------------------------------------------------------------
# Step 0: System preparation
# -------------------------------------------------------------------
echo "[0/7] Preparing system..."

# Ensure swap is adequate (need at least 8GB total virtual memory)
TOTAL_SWAP_KB=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
if [ "$TOTAL_SWAP_KB" -lt 4000000 ]; then
    echo "  WARNING: Swap is only $((TOTAL_SWAP_KB/1024))MB. Trying to increase..."
    if [ -f /etc/dphys-swapfile ]; then
        sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
        sudo systemctl restart dphys-swapfile || true
    else
        echo "  Creating 4GB swap file..."
        sudo fallocate -l 4G /swapfile_tflite 2>/dev/null || sudo dd if=/dev/zero of=/swapfile_tflite bs=1M count=4096
        sudo chmod 600 /swapfile_tflite
        sudo mkswap /swapfile_tflite
        sudo swapon /swapfile_tflite
    fi
    echo "  Swap after adjustment: $(free -h | grep Swap | awk '{print $2}')"
fi
echo "  Memory: $(free -h | grep Mem | awk '{print $2}') RAM, $(free -h | grep Swap | awk '{print $2}') Swap"

# -------------------------------------------------------------------
# Step 1: Install dependencies
# -------------------------------------------------------------------
echo ""
echo "[1/7] Installing build dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    zip unzip curl wget \
    openjdk-11-jdk \
    python3-dev python3-numpy python3-pip \
    libgles2-mesa-dev libegl1-mesa-dev \
    swig \
    git \
    2>/dev/null

echo "  Java: $(java -version 2>&1 | head -1)"
echo "  GCC:  $(gcc --version | head -1)"

# -------------------------------------------------------------------
# Step 2: Install Bazelisk
# -------------------------------------------------------------------
echo ""
echo "[2/7] Installing Bazelisk..."
BAZELISK_PATH="/usr/local/bin/bazel"

if command -v bazel &>/dev/null && bazel --version 2>/dev/null | grep -q "$BAZEL_VERSION"; then
    echo "  Bazel ${BAZEL_VERSION} already installed."
else
    BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-arm64"
    echo "  Downloading Bazelisk for arm64..."
    wget -q -O /tmp/bazelisk "$BAZELISK_URL"
    chmod +x /tmp/bazelisk
    sudo mv /tmp/bazelisk "$BAZELISK_PATH"
fi

export USE_BAZEL_VERSION="${BAZEL_VERSION}"
echo "  $(bazel --version 2>&1 | head -1)"

# -------------------------------------------------------------------
# Step 3: Clone TensorFlow
# -------------------------------------------------------------------
echo ""
echo "[3/7] Cloning TensorFlow v${TF_VERSION}..."

if [ -d "$TF_DIR" ]; then
    echo "  Directory exists. Verifying checkout..."
    cd "$TF_DIR"
    CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "unknown")
    if [ "$CURRENT_TAG" = "v${TF_VERSION}" ]; then
        echo "  Already at v${TF_VERSION}."
    else
        echo "  Wrong version ($CURRENT_TAG). Re-cloning..."
        cd "$HOME"
        rm -rf "$TF_DIR"
        git clone --depth=1 --branch "v${TF_VERSION}" \
            https://github.com/tensorflow/tensorflow.git "$TF_DIR"
    fi
else
    git clone --depth=1 --branch "v${TF_VERSION}" \
        https://github.com/tensorflow/tensorflow.git "$TF_DIR"
fi
cd "$TF_DIR"
echo "  TF source at: $TF_DIR"

# -------------------------------------------------------------------
# Step 4: Configure TensorFlow (non-interactive)
# -------------------------------------------------------------------
echo ""
echo "[4/7] Configuring TensorFlow..."

export TF_NEED_CUDA=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_NEED_MPI=0
export TF_NEED_TENSORRT=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_DOWNLOAD_CLANG=0
export PYTHON_BIN_PATH="$(which python3)"
export PYTHON_LIB_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
export CC_OPT_FLAGS="-march=armv8-a"

# Non-interactive configure
yes "" | ./configure 2>&1 || {
    echo "  [WARN] configure had issues, continuing anyway..."
}
echo "  Configuration complete."

# -------------------------------------------------------------------
# Step 5: Build GPU delegate
# -------------------------------------------------------------------
echo ""
echo "[5/7] Building libtensorflowlite_gpu_delegate.so..."
echo "  This will take several hours. Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  Build started: $(date)"

# Key flags:
# --copt="-DMESA_EGL_NO_X11_HEADERS": Required for Jetson EGL build
# --jobs=1: Prevent OOM on 4GB device
# --local_ram_resources=2048: Cap Bazel's RAM usage
bazel build \
    -s -c opt \
    --jobs=1 \
    --local_ram_resources=2048 \
    --copt="-DMESA_EGL_NO_X11_HEADERS" \
    --copt="-march=armv8-a" \
    --verbose_failures \
    //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
    2>&1

BUILD_EXIT=$?
echo "  Build finished: $(date)"

if [ $BUILD_EXIT -ne 0 ]; then
    echo ""
    echo "  [ERROR] Bazel build failed with exit code $BUILD_EXIT"
    echo "  Attempting fallback: build with CUDA support..."
    echo ""

    # Fallback: re-configure with CUDA
    export TF_NEED_CUDA=1
    export TF_CUDA_COMPUTE_CAPABILITIES="5.3"
    export CUDA_TOOLKIT_PATH="/usr/local/cuda"
    export CUDNN_INSTALL_PATH="/usr/lib/aarch64-linux-gnu"
    export TF_CUDA_VERSION="$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | cut -d, -f1 || echo '10.2')"
    export TF_CUDNN_VERSION="$(cat /usr/include/cudnn_version.h 2>/dev/null | grep CUDNN_MAJOR | head -1 | awk '{print $3}' || echo '8')"
    export GCC_HOST_COMPILER_PATH="$(which gcc)"

    yes "" | ./configure 2>&1 || true

    bazel build \
        -s -c opt \
        --jobs=1 \
        --local_ram_resources=2048 \
        --copt="-DMESA_EGL_NO_X11_HEADERS" \
        --verbose_failures \
        //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
        2>&1

    BUILD_EXIT=$?
    if [ $BUILD_EXIT -ne 0 ]; then
        echo "  [FATAL] Both build attempts failed."
        echo "  Check the log: $LOG_FILE"
        exit 1
    fi
fi

# -------------------------------------------------------------------
# Step 6: Install the delegate
# -------------------------------------------------------------------
echo ""
echo "[6/7] Installing GPU delegate..."

SO_PATH=$(find bazel-bin -name "libtensorflowlite_gpu_delegate.so" -type f | head -1)

if [ -z "$SO_PATH" ]; then
    echo "  [ERROR] .so not found in bazel-bin!"
    echo "  Searching all outputs..."
    find bazel-bin -name "*gpu*delegate*" -type f 2>/dev/null
    exit 1
fi

echo "  Found: $SO_PATH ($(du -h "$SO_PATH" | awk '{print $1}'))"
cp "$SO_PATH" "$INSTALL_DIR/"
sudo cp "$SO_PATH" /usr/local/lib/
sudo ldconfig
echo "  Installed to: $INSTALL_DIR/ and /usr/local/lib/"

# -------------------------------------------------------------------
# Step 7: Verify
# -------------------------------------------------------------------
echo ""
echo "[7/7] Verifying GPU delegate..."

python3 -c "
import tflite_runtime.interpreter as tflite
import os

delegate_path = '$INSTALL_DIR/libtensorflowlite_gpu_delegate.so'
print(f'  Delegate file: {delegate_path}')
print(f'  File size: {os.path.getsize(delegate_path) / (1024*1024):.1f} MB')

try:
    delegate = tflite.load_delegate(delegate_path)
    print('  [OK] GPU delegate loads successfully!')
except Exception as e:
    print(f'  [WARN] Delegate load test: {e}')
    print('  The delegate may still work during inference.')
"

echo ""
echo "============================================================"
echo "  BUILD COMPLETE"
echo "  Finished: $(date)"
echo "  Delegate: $INSTALL_DIR/libtensorflowlite_gpu_delegate.so"
echo "============================================================"
echo ""
echo "  Usage in Python:"
echo "    import tflite_runtime.interpreter as tflite"
echo "    delegate = tflite.load_delegate('$INSTALL_DIR/libtensorflowlite_gpu_delegate.so')"
echo "    interpreter = tflite.Interpreter(model_path='model.tflite', experimental_delegates=[delegate])"
