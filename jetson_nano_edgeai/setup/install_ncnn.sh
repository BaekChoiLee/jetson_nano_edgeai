#!/bin/bash
# Build and install ncnn with Vulkan support on Jetson Nano
set -e

echo "=== Building ncnn from source ==="

NCNN_DIR="${HOME}/ncnn"
INSTALL_PREFIX="/usr/local"

# Step 1: Dependencies
echo "[1/5] Installing dependencies..."
sudo apt-get install -y \
    libprotobuf-dev \
    protobuf-compiler \
    libvulkan-dev \
    vulkan-utils

# Step 2: Clone ncnn
echo "[2/5] Cloning ncnn repository..."
if [ -d "$NCNN_DIR" ]; then
    if [ -d "$NCNN_DIR/.git" ]; then
        echo "  ncnn directory exists, pulling latest..."
        cd "$NCNN_DIR" && git pull
    else
        echo "  ncnn directory exists but is not a git repo. Removing and re-cloning..."
        rm -rf "$NCNN_DIR"
        git clone --depth=1 https://github.com/Tencent/ncnn.git "$NCNN_DIR"
    fi
else
    git clone --depth=1 https://github.com/Tencent/ncnn.git "$NCNN_DIR"
fi
cd "$NCNN_DIR"
git submodule update --depth=1 --init

# Step 3: Build
echo "[3/5] Building ncnn (this takes ~30 minutes)..."
mkdir -p build && cd build
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -D NCNN_VULKAN=ON \
    -D NCNN_DISABLE_RTTI=OFF \
    -D NCNN_BUILD_TOOLS=ON \
    -D NCNN_BUILD_BENCHMARK=ON \
    -D NCNN_BUILD_EXAMPLES=OFF \
    -D NCNN_PYTHON=ON \
    ..

make -j$(nproc)

# Step 4: Install
echo "[4/5] Installing ncnn..."
sudo make install

# Install Python bindings
cd "${NCNN_DIR}/python"
pip3 install . 2>/dev/null || echo "  [INFO] Python bindings install skipped (optional)."

# Step 5: Verify
echo "[5/5] Verifying ncnn..."
BENCHNCNN="${NCNN_DIR}/build/benchmark/benchncnn"
if [ -f "$BENCHNCNN" ]; then
    echo "  benchncnn binary: $BENCHNCNN"
    echo "  Testing with squeezenet..."
    cd "${NCNN_DIR}/build/benchmark"
    ./benchncnn 1 4 0 -1 0 2>/dev/null && echo "  [OK] benchncnn works." || echo "  [WARN] benchncnn test failed (may need model files)."
else
    echo "  [ERROR] benchncnn not found at $BENCHNCNN"
    exit 1
fi

# onnx2ncnn tool
ONNX2NCNN="${NCNN_DIR}/build/tools/onnx/onnx2ncnn"
if [ -f "$ONNX2NCNN" ]; then
    echo "  onnx2ncnn binary: $ONNX2NCNN"
    echo "  [OK] onnx2ncnn available."
else
    echo "  [WARN] onnx2ncnn not found. ONNX conversion may not work."
fi

echo ""
echo "=== ncnn build complete ==="
echo "  Install prefix: $INSTALL_PREFIX"
echo "  benchncnn:      $BENCHNCNN"
echo "  onnx2ncnn:      $ONNX2NCNN"
echo ""
echo "  Add to your scripts:"
echo "  export NCNN_DIR=$NCNN_DIR"
