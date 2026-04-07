#!/usr/bin/env python3
"""Convert ONNX model to TFLite format."""

import argparse
import os
import sys
import subprocess


def convert_onnx_to_tflite(onnx_path, output_dir=None):
    """Convert ONNX → TF SavedModel → TFLite."""
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    output_dir = output_dir or os.path.dirname(onnx_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    saved_model_dir = os.path.join(output_dir, f"{model_name}_saved_model")
    tflite_path = os.path.join(output_dir, f"{model_name}.tflite")
    tflite_fp16_path = os.path.join(output_dir, f"{model_name}_fp16.tflite")

    # Step 1: ONNX → TF SavedModel
    print(f"[1/3] Converting ONNX to TF SavedModel...")
    print(f"  Input:  {onnx_path}")
    print(f"  Output: {saved_model_dir}")

    try:
        # Method 1: onnx-tf
        from onnx_tf.backend import prepare
        import onnx

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(saved_model_dir)
        print("  [OK] Converted via onnx-tf")
    except ImportError:
        print("  onnx-tf not available, trying onnx2tf...")
        try:
            # Method 2: onnx2tf (pip install onnx2tf)
            subprocess.run(
                [
                    "onnx2tf",
                    "-i", onnx_path,
                    "-o", saved_model_dir,
                    "-osd",
                ],
                check=True,
            )
            print("  [OK] Converted via onnx2tf")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("  [ERROR] Neither onnx-tf nor onnx2tf available.")
            print("  Install one of:")
            print("    pip3 install onnx-tf")
            print("    pip3 install onnx2tf")
            sys.exit(1)

    # Step 2: TF SavedModel → TFLite (default quantization)
    print(f"\n[2/3] Converting to TFLite (float32)...")
    try:
        import tensorflow as tf

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"  Saved: {tflite_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  [ERROR] TFLite conversion failed: {e}")
        sys.exit(1)

    # Step 3: TFLite FP16 quantized version
    print(f"\n[3/3] Converting to TFLite (float16 quantized)...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_fp16_model = converter.convert()

        with open(tflite_fp16_path, "wb") as f:
            f.write(tflite_fp16_model)

        size_mb = os.path.getsize(tflite_fp16_path) / (1024 * 1024)
        print(f"  Saved: {tflite_fp16_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  [WARN] FP16 quantization failed: {e}")
        print(f"  Continuing with float32 version only.")

    print(f"\n[OK] TFLite conversion complete for {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TFLite")
    parser.add_argument("--onnx-path", required=True, help="Path to ONNX model")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.onnx_path):
        print(f"[ERROR] ONNX file not found: {args.onnx_path}")
        sys.exit(1)

    convert_onnx_to_tflite(args.onnx_path, args.output_dir)


if __name__ == "__main__":
    main()
