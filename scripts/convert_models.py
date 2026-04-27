import argparse
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_MODELS = ["mobilenetv3s", "efficientnetb0", "shufflenetv2", "resnet50"]


def resolve_tool(env_name, executable):
    override = os.environ.get(env_name)
    if override:
        resolved = shutil.which(override) or override
        return resolved if os.path.exists(resolved) or shutil.which(resolved) else None
    return shutil.which(executable)


def run_cmd(cmd, description):
    print(f"  - {description}")
    print(f"    $ {' '.join(str(part) for part in cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"    [Warning] failed with exit code {result.returncode}")
        tail = result.stdout[-2000:].strip()
        if tail:
            print(tail)
        return False
    return True


def convert_tensorrt(model_name, onnx_path, model_dir):
    trtexec = resolve_tool("TRTEXEC_PATH", "trtexec")
    if not trtexec:
        print("[Warning] trtexec not found. Skipping TensorRT engine conversion.")
        print("          Set TRTEXEC_PATH if trtexec is not in PATH.")
        return

    precision_flags = {
        "fp32": [],
        "fp16": ["--fp16"],
        # INT8 usually needs calibration or Q/DQ ranges. We still try it so
        # Jetson setups with supported INT8 paths can generate an engine.
        "int8": ["--int8"],
    }

    for precision, flags in precision_flags.items():
        engine_path = model_dir / f"{model_name}_{precision}.engine"
        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--explicitBatch",
        ] + flags
        ok = run_cmd(cmd, f"TensorRT {precision.upper()} engine: {engine_path.name}")
        if ok:
            print(f"    [Done] saved {engine_path}")


def convert_ncnn(model_name, onnx_path, model_dir):
    onnx2ncnn = resolve_tool("ONNX2NCNN_PATH", "onnx2ncnn")
    if not onnx2ncnn:
        print("[Warning] onnx2ncnn not found. Skipping ncnn conversion.")
        print("          Set ONNX2NCNN_PATH if onnx2ncnn is not in PATH.")
        return

    param_path = model_dir / f"{model_name}.param"
    bin_path = model_dir / f"{model_name}.bin"
    ok = run_cmd(
        [onnx2ncnn, str(onnx_path), str(param_path), str(bin_path)],
        f"ncnn model: {param_path.name} / {bin_path.name}",
    )
    if ok:
        print(f"    [Done] saved {param_path} and {bin_path}")


def convert_tflite(model_name, onnx_path, model_dir):
    tflite_path = model_dir / f"{model_name}.tflite"

    onnx2tf = resolve_tool("ONNX2TF_PATH", "onnx2tf")
    tflite_convert = resolve_tool("TFLITE_CONVERT_PATH", "tflite_convert")
    if onnx2tf and tflite_convert:
        saved_model_dir = model_dir / f"{model_name}_saved_model"
        ok = run_cmd(
            [onnx2tf, "-i", str(onnx_path), "-o", str(saved_model_dir)],
            f"TensorFlow SavedModel for TFLite: {saved_model_dir.name}",
        )
        if ok:
            ok = run_cmd(
                [
                    tflite_convert,
                    f"--saved_model_dir={saved_model_dir}",
                    f"--output_file={tflite_path}",
                ],
                f"TFLite flatbuffer: {tflite_path.name}",
            )
            if ok:
                print(f"    [Done] saved {tflite_path}")
        return

    onnx_tf_available = shutil.which("python3") is not None
    if onnx_tf_available and tflite_convert:
        saved_model_dir = model_dir / f"{model_name}_saved_model"
        ok = run_cmd(
            [
                "python3",
                "-m",
                "onnx_tf.backend",
                "convert",
                "-i",
                str(onnx_path),
                "-o",
                str(saved_model_dir),
            ],
            f"TensorFlow SavedModel via onnx-tf: {saved_model_dir.name}",
        )
        if ok:
            ok = run_cmd(
                [
                    tflite_convert,
                    f"--saved_model_dir={saved_model_dir}",
                    f"--output_file={tflite_path}",
                ],
                f"TFLite flatbuffer: {tflite_path.name}",
            )
            if ok:
                print(f"    [Done] saved {tflite_path}")
        return

    print("[Warning] TFLite conversion tools not found. Skipping TFLite conversion.")
    print("          Supported paths:")
    print("          - onnx2tf + tflite_convert")
    print("          - python3 -m onnx_tf.backend + tflite_convert")
    print("          Set ONNX2TF_PATH or TFLITE_CONVERT_PATH if needed.")


def main():
    parser = argparse.ArgumentParser(description="Convert exported ONNX models to TensorRT, TFLite, and ncnn artifacts.")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--targets", nargs="+", default=["tensorrt", "tflite", "ncnn"], choices=["tensorrt", "tflite", "ncnn"])
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    print("=====================================")
    print("Converting ONNX Models")
    print(f"Model Directory: {model_dir}")
    print(f"Targets: {args.targets}")
    print("=====================================")

    for model_name in args.models:
        onnx_path = model_dir / f"{model_name}.onnx"
        if not onnx_path.exists():
            print(f"[Skip] ONNX model not found: {onnx_path}")
            continue

        print(f"\n[Converting Model: {model_name}]")
        if "tensorrt" in args.targets:
            convert_tensorrt(model_name, onnx_path, model_dir)
        if "tflite" in args.targets:
            convert_tflite(model_name, onnx_path, model_dir)
        if "ncnn" in args.targets:
            convert_ncnn(model_name, onnx_path, model_dir)

    print("\n[DONE] Model conversion step finished.")


if __name__ == "__main__":
    main()
