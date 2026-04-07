#!/bin/bash
# Jetson Nano initial environment setup
# Run this FIRST after flashing Q-engineering Ubuntu 20.04 image
set -e

echo "============================================"
echo "  Jetson Nano Benchmark Environment Setup"
echo "============================================"

# --- Step 1: Swap memory (4GB) ---
echo "[1/6] Setting up swap memory..."
if [ -f /swapfile ]; then
    echo "  Swap file already exists, skipping."
else
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
    echo "  4GB swap file created and enabled."
fi
free -h | grep -i swap

# --- Step 2: Fan control (max speed) ---
echo "[2/6] Setting fan to max speed..."
if [ -w /sys/devices/pwm-fan/target_pwm ]; then
    echo 255 | sudo tee /sys/devices/pwm-fan/target_pwm > /dev/null
    echo "  Fan set to max (255)."
else
    echo "  PWM fan control not found, skipping."
fi

# --- Step 3: System packages ---
echo "[3/6] Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libvulkan-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libatlas-base-dev \
    gfortran \
    build-essential \
    pkg-config \
    wget \
    curl \
    unzip

# CMake 3.21+ (Jetson default is too old for ncnn)
CMAKE_VER=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}' || echo "0.0.0")
CMAKE_MAJOR=$(echo "$CMAKE_VER" | cut -d. -f1)
CMAKE_MINOR=$(echo "$CMAKE_VER" | cut -d. -f2)
if [ "$CMAKE_MAJOR" -lt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 21 ]); then
    echo "  Upgrading CMake to 3.22.1..."
    wget -q https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-aarch64.sh
    sudo bash cmake-3.22.1-linux-aarch64.sh --prefix=/usr/local --skip-license
    rm cmake-3.22.1-linux-aarch64.sh
    echo "  CMake upgraded to $(cmake --version | head -1)"
else
    echo "  CMake $CMAKE_VER is sufficient."
fi

# --- Step 4: Python packages ---
echo "[4/6] Installing Python packages..."
pip3 install --upgrade pip
pip3 install \
    numpy \
    pillow \
    pandas \
    matplotlib \
    scipy \
    tqdm \
    onnx \
    pyyaml

# --- Step 5: CUDA environment ---
echo "[5/6] Setting CUDA environment variables..."
CUDA_EXPORT='export PATH=/usr/local/cuda/bin:$PATH'
CUDA_LD='export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
if ! grep -q "cuda/bin" ~/.bashrc 2>/dev/null; then
    echo "$CUDA_EXPORT" >> ~/.bashrc
    echo "$CUDA_LD" >> ~/.bashrc
    echo "  CUDA paths added to ~/.bashrc"
else
    echo "  CUDA paths already in ~/.bashrc"
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# --- Step 6: System info ---
echo "[6/6] System Information"
echo "============================================"
echo "  OS:       $(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY | cut -d= -f2)"
echo "  Kernel:   $(uname -r)"
echo "  Arch:     $(uname -m)"
echo "  Python:   $(python3 --version)"
echo "  pip:      $(pip3 --version | awk '{print $2}')"
echo "  CMake:    $(cmake --version | head -1 | awk '{print $3}')"
echo "  CUDA:     $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',' || echo 'not found')"
echo "  Memory:   $(free -h | grep Mem | awk '{print $2}')"
echo "  Swap:     $(free -h | grep Swap | awk '{print $2}')"
echo "  Disk:     $(df -h / | tail -1 | awk '{print $4}') free"
echo "============================================"
echo "Setup complete! Run verify_env.py next."
