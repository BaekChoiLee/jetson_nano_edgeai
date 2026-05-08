#!/usr/bin/env python3
"""Try converting ShuffleNet using onnx2tf SavedModel."""
import os, sys
os.chdir(os.path.expanduser("~/jetson-benchmark"))

import tensorflow as tf

tflite_path = "models/shufflenet_v2_x1_0.tflite"

# Try multiple SavedModel sources
sources = [
    "models/shufflenet_v2_x1_0_onnx2tf",
    "models/shufflenet_v2_x1_0_saved_model",
    "models/shufflenet_v2_x1_0_saved_model_retry",
]

for sm in sources:
    if not os.path.exists(sm):
        print(f"SKIP {sm}: not found")
        continue
    print(f"\nTrying: {sm}")

    # Attempt 1: builtins only
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(sm)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite)
        print(f"  [OK] builtins only: {len(tflite)/1e6:.1f} MB")
        break
    except Exception as e:
        print(f"  builtins failed: {str(e)[:150]}")

    # Attempt 2: builtins + SELECT_TF_OPS
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(sm)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite)
        print(f"  [OK] with TF_OPS: {len(tflite)/1e6:.1f} MB")
        break
    except Exception as e:
        print(f"  TF_OPS failed: {str(e)[:150]}")

    # Attempt 3: quantized
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(sm)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite)
        print(f"  [OK] quantized: {len(tflite)/1e6:.1f} MB")
        break
    except Exception as e:
        print(f"  quantized failed: {str(e)[:150]}")
else:
    print("\n[FAIL] All conversion attempts failed")
    sys.exit(1)

# Verify
print("\nVerification:")
interp = tf.lite.Interpreter(model_path=tflite_path)
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()
print(f"  Input: shape={inp[0]['shape']} dtype={inp[0]['dtype']}")
print(f"  Output: shape={out[0]['shape']} dtype={out[0]['dtype']}")
print("[OK] Done")
