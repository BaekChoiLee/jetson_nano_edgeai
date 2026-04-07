#!/usr/bin/env python3
"""Verify all benchmark frameworks are installed on Jetson Nano."""

import sys
import os
import subprocess
import shutil


def check_module(name, import_name=None):
    """Try importing a module, return (success, version_info)."""
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "installed")
        return True, version
    except ImportError as e:
        return False, str(e)


def check_binary(name, version_cmd=None):
    """Check if a binary exists in PATH."""
    path = shutil.which(name)
    if not path:
        return False, "not found in PATH"
    if version_cmd:
        try:
            result = subprocess.run(
                version_cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            return True, result.stdout.strip().split("\n")[0]
        except Exception as e:
            return True, f"found at {path}"
    return True, f"found at {path}"


def main():
    results = []
    critical_failures = []

    print("=" * 60)
    print("  Jetson Nano Benchmark Environment Verification")
    print("=" * 60)
    print()

    # --- Python & System ---
    print("[System]")
    print(f"  Python: {sys.version}")
    print(f"  Platform: {sys.platform}")

    # --- PyTorch ---
    print("\n[PyTorch]")
    ok, info = check_module("torch")
    if ok:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_info = ""
        if cuda_available:
            cuda_info = f", CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}"
        print(f"  torch: {torch.__version__}{cuda_info}")
        results.append(("PyTorch", "OK", torch.__version__))
        if not cuda_available:
            print("  [WARN] CUDA not available for PyTorch!")
            results.append(("PyTorch CUDA", "WARN", "not available"))
        else:
            results.append(("PyTorch CUDA", "OK", torch.version.cuda))
    else:
        print(f"  [FAIL] {info}")
        results.append(("PyTorch", "FAIL", info))
        critical_failures.append("PyTorch")

    # TorchVision
    ok, info = check_module("torchvision")
    if ok:
        import torchvision
        print(f"  torchvision: {torchvision.__version__}")
        results.append(("TorchVision", "OK", torchvision.__version__))
    else:
        print(f"  [FAIL] {info}")
        results.append(("TorchVision", "FAIL", info))
        critical_failures.append("TorchVision")

    # --- TensorRT ---
    print("\n[TensorRT]")
    ok, info = check_module("tensorrt")
    if ok:
        import tensorrt as trt
        print(f"  TensorRT: {trt.__version__}")
        results.append(("TensorRT", "OK", trt.__version__))
    else:
        print(f"  [FAIL] {info}")
        results.append(("TensorRT", "FAIL", info))
        critical_failures.append("TensorRT")

    # trtexec binary
    ok, info = check_binary("trtexec", "trtexec --help 2>&1 | head -1")
    trtexec_alt = "/usr/src/tensorrt/bin/trtexec"
    if not ok and os.path.exists(trtexec_alt):
        ok = True
        info = f"found at {trtexec_alt}"
    print(f"  trtexec: {info}")
    results.append(("trtexec", "OK" if ok else "WARN", info))

    # --- ONNX Runtime ---
    print("\n[ONNX Runtime]")
    ok, info = check_module("onnxruntime", "onnxruntime")
    if ok:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  ONNX Runtime: {ort.__version__}")
        print(f"  Providers: {providers}")
        results.append(("ONNX Runtime", "OK", ort.__version__))
        has_cuda = "CUDAExecutionProvider" in providers
        has_trt = "TensorrtExecutionProvider" in providers
        results.append(("ONNX CUDA EP", "OK" if has_cuda else "WARN", str(has_cuda)))
        results.append(("ONNX TRT EP", "OK" if has_trt else "INFO", str(has_trt)))
    else:
        print(f"  [FAIL] {info}")
        results.append(("ONNX Runtime", "FAIL", info))
        critical_failures.append("ONNX Runtime")

    # --- TFLite ---
    print("\n[TFLite]")
    tflite_ok = False
    try:
        import tflite_runtime.interpreter as tflite
        ver = getattr(tflite, "__version__", "unknown")
        print(f"  tflite_runtime: {ver}")
        results.append(("TFLite", "OK", f"tflite_runtime {ver}"))
        tflite_ok = True
    except ImportError:
        try:
            import tensorflow as tf
            print(f"  Using tf.lite from TensorFlow {tf.__version__}")
            results.append(("TFLite", "OK", f"tf.lite ({tf.__version__})"))
            tflite_ok = True
        except ImportError as e:
            print(f"  [FAIL] {e}")
            results.append(("TFLite", "FAIL", str(e)))
            critical_failures.append("TFLite")

    # --- ncnn ---
    print("\n[ncnn]")
    ncnn_ok = False
    # Check Python binding
    ok, info = check_module("ncnn")
    if ok:
        print(f"  ncnn Python: {info}")
        results.append(("ncnn Python", "OK", info))
        ncnn_ok = True

    # Check benchncnn binary
    ncnn_dir = os.environ.get("NCNN_DIR", os.path.expanduser("~/ncnn"))
    benchncnn = os.path.join(ncnn_dir, "build", "benchmark", "benchncnn")
    onnx2ncnn = os.path.join(ncnn_dir, "build", "tools", "onnx", "onnx2ncnn")

    if os.path.exists(benchncnn):
        print(f"  benchncnn: {benchncnn}")
        results.append(("ncnn benchncnn", "OK", benchncnn))
        ncnn_ok = True
    else:
        bin_path = shutil.which("benchncnn")
        if bin_path:
            print(f"  benchncnn: {bin_path}")
            results.append(("ncnn benchncnn", "OK", bin_path))
            ncnn_ok = True
        else:
            print("  [WARN] benchncnn not found")
            results.append(("ncnn benchncnn", "WARN", "not found"))

    if os.path.exists(onnx2ncnn):
        print(f"  onnx2ncnn: {onnx2ncnn}")
        results.append(("ncnn onnx2ncnn", "OK", onnx2ncnn))
    else:
        print("  [WARN] onnx2ncnn not found")
        results.append(("ncnn onnx2ncnn", "WARN", "not found"))

    if not ncnn_ok:
        critical_failures.append("ncnn")

    # --- Supporting Libraries ---
    print("\n[Supporting Libraries]")
    for lib in ["numpy", "PIL", "cv2", "pandas", "matplotlib", "onnx", "scipy"]:
        import_name = "PIL" if lib == "PIL" else ("cv2" if lib == "cv2" else lib)
        ok, info = check_module(lib, import_name)
        status = "OK" if ok else "WARN"
        symbol = "OK" if ok else "WARN"
        print(f"  {lib}: [{symbol}] {info}")
        results.append((lib, status, info))

    # --- CUDA / cuDNN ---
    print("\n[CUDA Environment]")
    nvcc_ok, nvcc_info = check_binary("nvcc", "nvcc --version 2>&1 | grep release")
    print(f"  nvcc: {nvcc_info}")
    results.append(("nvcc", "OK" if nvcc_ok else "WARN", nvcc_info))

    # tegrastats
    tegra_ok, tegra_info = check_binary("tegrastats")
    print(f"  tegrastats: {tegra_info}")
    results.append(("tegrastats", "OK" if tegra_ok else "WARN", tegra_info))

    # nvpmodel
    nv_ok, nv_info = check_binary("nvpmodel")
    print(f"  nvpmodel: {nv_info}")
    results.append(("nvpmodel", "OK" if nv_ok else "WARN", nv_info))

    # --- GPU Memory ---
    print("\n[GPU Memory]")
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem / (1024**2)
            print(f"  GPU memory: {total:.0f} MB")
    except Exception:
        print("  Could not query GPU memory.")

    # --- Summary Table ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Component':<22} {'Status':<8} {'Info'}")
    print(f"  {'-'*22} {'-'*8} {'-'*28}")
    for name, status, info in results:
        symbol = {"OK": "[OK]", "WARN": "[!!]", "FAIL": "[XX]", "INFO": "[--]"}.get(status, "[??]")
        short_info = str(info)[:35]
        print(f"  {name:<22} {symbol:<8} {short_info}")

    print()
    if critical_failures:
        print(f"  CRITICAL FAILURES: {', '.join(critical_failures)}")
        print("  Fix these before running benchmarks!")
        sys.exit(1)
    else:
        print("  All critical frameworks available!")
        print("  Ready to run benchmarks.")
        sys.exit(0)


if __name__ == "__main__":
    main()
