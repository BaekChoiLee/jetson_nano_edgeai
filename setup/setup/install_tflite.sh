#!/bin/bash
# Install TFLite Runtime on Jetson Nano
set -e

echo "=== Installing TFLite Runtime ==="

# Step 1: Install tflite-runtime
echo "[1/2] Installing tflite-runtime..."
pip3 install tflite-runtime 2>/dev/null && {
    echo "  Installed tflite-runtime via pip."
} || {
    echo "  pip install failed, trying alternative..."
    # For aarch64, may need to specify URL
    pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime 2>/dev/null && {
        echo "  Installed from Google Coral repo."
    } || {
        echo "  [WARN] Could not install tflite-runtime."
        echo "  TensorFlow 2.4.1 (pre-installed in Q-eng image) includes TFLite."
        echo "  Trying to use tf.lite instead..."
    }
}

# Step 2: Verify
echo "[2/2] Verifying TFLite..."
python3 -c "
try:
    import tflite_runtime.interpreter as tflite
    print(f'  tflite_runtime version: {tflite.__version__ if hasattr(tflite, \"__version__\") else \"unknown\"}')
    print('  [OK] tflite_runtime available')
except ImportError:
    print('  tflite_runtime not found, trying tf.lite...')
    import tensorflow as tf
    print(f'  TensorFlow version: {tf.__version__}')
    interpreter = tf.lite.Interpreter
    print('  [OK] tf.lite available (using full TensorFlow)')
"

echo ""
echo "=== TFLite installation complete ==="
echo ""
echo "NOTE on GPU delegate:"
echo "  The Jetson Nano GPU delegate requires custom Bazel build."
echo "  In Q-engineering tests, GPU delegate is often SLOWER than CPU"
echo "  on Jetson Nano due to delegation overhead."
echo "  We benchmark CPU-only TFLite by default."
echo "  To build GPU delegate: https://qengineering.eu/install-tensorflow-2-lite-on-jetson-nano.html"
