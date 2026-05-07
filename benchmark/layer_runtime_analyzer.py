#!/usr/bin/env python3
"""Runtime-aware layer latency profiler.

Supported runtimes:
  - pytorch_cpu, pytorch_cuda  -> PyTorch profiler (operator-level)
  - tensorrt_fp32/fp16/int8    -> TensorRT IProfiler (layer-level)
  - onnxrt_cuda, onnxrt_trt    -> ONNX Runtime profiling trace (node-level)
  - tflite_cpu, tflite_gpu     -> benchmark_model op profiling (if available)
  - ncnn_cpu, ncnn_vulkan      -> benchncnn layer timing (requires NCNN_BENCHMARK=ON)
"""

import argparse
import csv
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
from collections import defaultdict

import numpy as np


CLASSIFICATION_FACTORIES = {
    "mobilenetv3_small": "mobilenet_v3_small",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "shufflenet_v2_x1_0": "shufflenet_v2_x1_0",
}

DETECTION_INPUT_SIZES = {
    "yolov8n": (1, 3, 640, 640),
    "ssd_mobilenet_v2": (1, 3, 320, 320),
}

ALL_SUPPORTED_MODELS = list(CLASSIFICATION_FACTORIES.keys()) + list(DETECTION_INPUT_SIZES.keys())


def _model_hw(model_name):
    if model_name in DETECTION_INPUT_SIZES:
        _, _, h, w = DETECTION_INPUT_SIZES[model_name]
        return h, w
    return 224, 224


def _resolve_tflite_path(model_name, model_dir):
    candidates = []
    if model_name == "ssd_mobilenet_v2":
        candidates.extend(
            [
                os.path.join(model_dir, "ssd_mobilenet_v2.tflite"),
                os.path.join(model_dir, "ssd_mobilenet_v2_raw.tflite"),
                os.path.join(model_dir, "ssd_mobilenet_v2_float32.tflite"),
            ]
        )
    elif model_name == "yolov8n":
        candidates.extend(
            [
                os.path.join(model_dir, "yolov8n.tflite"),
                os.path.join(model_dir, "yolov8n_fp16.tflite"),
                os.path.join(model_dir, "yolov8n_float32.tflite"),
            ]
        )
    else:
        candidates.append(os.path.join(model_dir, f"{model_name}.tflite"))

    # dynamic fallback globs
    for pat in (
        os.path.join(model_dir, f"{model_name}*.tflite"),
        os.path.join(model_dir, f"{model_name}_saved_model", "*.tflite"),
    ):
        for p in sorted(glob.glob(pat)):
            if p not in candidates:
                candidates.append(p)

    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    raise FileNotFoundError(f"TFLite model not found for {model_name}: {candidates}")


def _resolve_ncnn_paths(model_name, model_dir):
    candidates = []
    # ultralytics export dir
    candidates.extend(
        [
            (
                os.path.join(model_dir, f"{model_name}_ncnn_model", "model.param"),
                os.path.join(model_dir, f"{model_name}_ncnn_model", "model.bin"),
            ),
            # prefer pnnx-exported .ncnn.param (benchncnn per-layer timing works)
            (
                os.path.join(model_dir, f"{model_name}.ncnn.param"),
                os.path.join(model_dir, f"{model_name}.ncnn.bin"),
            ),
            (
                os.path.join(model_dir, f"{model_name}.param"),
                os.path.join(model_dir, f"{model_name}.bin"),
            ),
        ]
    )
    if model_name == "ssd_mobilenet_v2":
        candidates.extend(
            [
                (
                    os.path.join(model_dir, "ssd_mobilenet_v2.ncnn.param"),
                    os.path.join(model_dir, "ssd_mobilenet_v2.ncnn.bin"),
                ),
                (
                    os.path.join(model_dir, "ssd_mobilenet_v2_raw.param"),
                    os.path.join(model_dir, "ssd_mobilenet_v2_raw.bin"),
                ),
                (
                    os.path.join(model_dir, "ssd_mobilenet_v2_raw.ncnn.param"),
                    os.path.join(model_dir, "ssd_mobilenet_v2_raw.ncnn.bin"),
                ),
            ]
        )

    for param_path, bin_path in candidates:
        if (
            os.path.exists(param_path)
            and os.path.exists(bin_path)
            and os.path.getsize(param_path) > 0
            and os.path.getsize(bin_path) > 0
        ):
            return param_path, bin_path
    raise FileNotFoundError(f"ncnn model files not found for {model_name}: {candidates}")


def _get_tflite_input_name(model_path):
    """Return the first input tensor name from a TFLite flatbuffer.

    benchmark_model requires --input_layer to match --input_layer_shape in
    newer builds.  We detect the name dynamically so the command stays valid
    across model variants (yolov8n, ssd, classification models).
    """
    try:
        for backend in ("tflite_runtime", "tensorflow"):
            try:
                if backend == "tflite_runtime":
                    from tflite_runtime.interpreter import Interpreter as _Interp
                else:
                    import tensorflow as tf
                    _Interp = tf.lite.Interpreter
                itp = _Interp(model_path=model_path)
                itp.allocate_tensors()
                name = itp.get_input_details()[0]["name"]
                # Sanitize: benchmark_model accepts names without the :0 suffix
                return name.split(":")[0]
            except Exception:
                continue
    except Exception:
        pass
    # Fallback: use generic name; benchmark_model may accept empty string for
    # single-input models but this at least prevents the "0 items" error.
    return "input"


def _find_tflite_benchmark_model():
    env = os.environ.get("TFLITE_BENCHMARK_MODEL_BIN")
    candidates = []
    if env:
        candidates.append(env)
    candidates.extend(
        [
            shutil.which("benchmark_model") or "",
            os.path.expanduser("~/jetson-benchmark/tools/benchmark_model"),
            os.path.expanduser("~/tensorflow-2.13.0/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model"),
            os.path.expanduser("~/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model"),
            "/usr/local/bin/benchmark_model",
            "/usr/bin/benchmark_model",
        ]
    )
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _find_benchncnn():
    env = os.environ.get("NCNN_BENCHNCNN_BIN")
    candidates = []
    if env:
        candidates.append(env)
    candidates.extend(
        [
            shutil.which("benchncnn") or "",
            os.path.expanduser("~/ncnn/build-bench/benchmark/benchncnn"),
            os.path.expanduser("~/ncnn/build_bench/benchmark/benchncnn"),
            os.path.expanduser("~/ncnn/build/benchmark/benchncnn"),
        ]
    )
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _run_cmd(cmd, timeout_sec=600):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
    )
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, out


def _parse_tflite_run_order(output_text, default_samples):
    """Parse benchmark_model op profiling text.

    We parse the "Run Order" section which includes per-node avg ms.
    """
    in_regular = False
    in_run_order = False
    records = []
    for raw in output_text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            continue

        # Two profiling sections exist: Init (AllocateTensors only) and Regular (ops).
        # Skip the Init section entirely; start accepting rows only after the
        # "Operator-wise Profiling Info for Regular Benchmark Runs" header.
        if "Operator-wise Profiling Info for Regular" in stripped:
            in_regular = True
            in_run_order = False
            continue
        if not in_regular:
            continue

        if "============================== Run Order" in stripped:
            in_run_order = True
            continue
        if in_run_order and "============================== Top by Computation Time" in stripped:
            break
        if not in_run_order:
            continue
        if stripped.startswith("[node type]") or stripped.startswith("[Node type]"):
            continue
        if stripped.startswith("="):
            continue

        # Canonical row format (README):
        # [node type] [start] [first] [avg ms] [%] [cdf%] [mem KB] [times called] [Name]
        # Example:
        # CONV_2D 0.000 4.269 4.269 0.107% 0.107% 0.000 0 [Mobilenet/...]
        name_match = re.search(r"\[(?P<name>[^\]]+)\](?::\d+)?\s*$", stripped)
        if not name_match:
            continue

        left = stripped[: name_match.start()].strip()
        parts = re.split(r"\s+", left)
        # Detect format: new (7 cols) has no "start" column
        #   new: node_type first avg %  cdf%  mem  times
        #   old: node_type start first avg %  cdf%  mem  times
        # avg_ms is the numeric column immediately before first "%" entry.
        pct_idx = next((i for i, v in enumerate(parts) if v.endswith("%")), -1)
        if pct_idx < 2:
            continue
        node_type = parts[0]
        try:
            avg_ms = float(parts[pct_idx - 1])
        except Exception:
            continue
        times_called = 0
        try:
            times_called = int(float(parts[-1]))
        except Exception:
            times_called = 0
        samples = times_called if times_called > 0 else int(default_samples)

        records.append(
            {
                "layer_name": f"{node_type}:{name_match.group('name')}",
                "backend": "tflite",
                "granularity": "tflite_node",
                "mean_ms": round(float(avg_ms), 6),
                "std_ms": 0.0,
                "min_ms": None,
                "max_ms": None,
                "samples": samples,
            }
        )

    # De-duplicate while preserving first occurrence
    dedup = []
    seen = set()
    for r in records:
        key = r["layer_name"]
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    dedup.sort(key=lambda r: r["mean_ms"], reverse=True)
    return dedup


def _profile_tflite(
    model_name,
    runtime,
    model_dir,
    num_warmup,
    num_runs,
    tflite_benchmark_bin="",
):
    model_path = _resolve_tflite_path(model_name, model_dir)
    benchmark_bin = ""
    if tflite_benchmark_bin and os.path.exists(tflite_benchmark_bin):
        benchmark_bin = tflite_benchmark_bin
    else:
        benchmark_bin = _find_tflite_benchmark_model()
    if not benchmark_bin:
        raise FileNotFoundError(
            "benchmark_model not found. "
            "Set --tflite-benchmark-bin or TFLITE_BENCHMARK_MODEL_BIN."
        )

    h, w = _model_hw(model_name)
    use_gpu = runtime == "tflite_gpu"

    # Detect input tensor name — benchmark_model requires --input_layer when
    # --input_layer_shape is specified (required by newer TFLite benchmark builds).
    input_layer_name = _get_tflite_input_name(model_path)

    cmd = [
        benchmark_bin,
        f"--graph={model_path}",
        f"--warmup_runs={int(num_warmup)}",
        f"--num_runs={int(num_runs)}",
        "--enable_op_profiling=true",
        "--max_profiling_buffer_entries=4096",
        "--num_threads=4",
        f"--use_gpu={'true' if use_gpu else 'false'}",
        "--use_xnnpack=false",
    ]
    if use_gpu:
        cmd.extend(
            [
                "--gpu_precision_loss_allowed=true",
                "--gpu_experimental_enable_quant=true",
            ]
        )

    # benchmark_model's op-profiling table only reports mean avg_ms — no per-op
    # min/max. Run the full benchmark N times and aggregate per-op means to
    # recover dispersion statistics. Each invocation honors num_warmup/num_runs.
    repeats = int(os.environ.get("TFLITE_LAYER_REPEATS", "3"))
    repeats = max(1, min(repeats, 10))

    per_op_means = defaultdict(list)   # key -> list[mean_ms per repeat]
    per_op_samples = {}                # key -> int (from last repeat)

    for r_idx in range(repeats):
        rc, out = _run_cmd(cmd, timeout_sec=max(600, int(num_runs) * 4))
        if rc != 0:
            raise RuntimeError(
                f"benchmark_model failed rc={rc} (repeat {r_idx}/{repeats}). "
                f"cmd={shlex.join(cmd)}\n{out[-4000:]}"
            )
        rep_records = _parse_tflite_run_order(out, default_samples=num_runs)
        if not rep_records:
            raise RuntimeError(
                "No per-op rows parsed from benchmark_model output. "
                "Check if --enable_op_profiling is supported in this binary."
            )
        for rec in rep_records:
            key = rec["layer_name"]
            per_op_means[key].append(float(rec["mean_ms"]))
            per_op_samples[key] = int(rec.get("samples") or num_runs)

    records = []
    for key, means in per_op_means.items():
        arr = np.array(means, dtype=np.float64)
        records.append(
            {
                "layer_name": key,
                "backend": "tflite",
                "granularity": "tflite_node",
                "mean_ms": round(float(arr.mean()), 6),
                "std_ms": round(float(arr.std()), 6) if arr.size >= 2 else 0.0,
                "min_ms": round(float(arr.min()), 6),
                "max_ms": round(float(arr.max()), 6),
                "samples": per_op_samples.get(key, num_runs),
            }
        )
    # records.sort(key=lambda r: r["mean_ms"], reverse=True)

    return {
        "status": "ok",
        "backend": "tflite",
        "granularity": "tflite_node",
        "records": records,
        "tool_path": benchmark_bin,
        "model_path": model_path,
        "repeats": repeats,
    }


def _parse_ncnn_layer_lines(output_text):
    """Parse NCNN_BENCHMARK per-layer timing lines from benchncnn output."""
    # Example (from ncnn benchmark.cpp):
    # ConvolutionDepthWise      conv_dw_1                     1.23ms    | ...
    layer_times = defaultdict(list)
    for raw in output_text.splitlines():
        line = raw.strip()
        if not line or "|" not in line or ("ms" not in line and "us" not in line):
            continue
        m = re.match(
            r"^(?P<layer_type>[A-Za-z0-9_]+)\s+(?P<layer_name>\S+)\s+(?P<val>[-+]?\d+(?:\.\d+)?)(?P<unit>ms|us)\b",
            line,
        )
        if not m:
            continue
        key = f"{m.group('layer_type')}:{m.group('layer_name')}"
        _v = float(m.group("val"))
        if m.group("unit") == "us":
            _v /= 1000.0
        layer_times[key].append(_v)

    records = []
    for name, values in layer_times.items():
        arr = np.array(values, dtype=np.float64)
        if arr.size == 0:
            continue
        records.append(
            {
                "layer_name": name,
                "backend": "ncnn",
                "granularity": "ncnn_layer",
                "mean_ms": round(float(arr.mean()), 6),
                "std_ms": round(float(arr.std()), 6),
                "min_ms": round(float(arr.min()), 6),
                "max_ms": round(float(arr.max()), 6),
                "samples": int(arr.size),
            }
        )
    # records.sort(key=lambda r: r["mean_ms"], reverse=True)
    return records


def _detect_ncnn_benchmark_macro(benchncnn_path):
    # For a standard ncnn tree layout:
    #   ~/ncnn/build/benchmark/benchncnn -> ~/ncnn/build/src/platform.h
    build_dir = os.path.dirname(os.path.dirname(os.path.abspath(benchncnn_path)))
    platform_h = os.path.join(build_dir, "src", "platform.h")
    if not os.path.exists(platform_h):
        return None
    try:
        with open(platform_h, "r") as f:
            text = f.read()
        m = re.search(r"#define\s+NCNN_BENCHMARK\s+(\d+)", text)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None


def _profile_ncnn(
    model_name,
    runtime,
    model_dir,
    num_warmup,
    num_runs,
    ncnn_benchmark_bin="",
):
    param_path, bin_path = _resolve_ncnn_paths(model_name, model_dir)
    benchncnn_bin = ""
    if ncnn_benchmark_bin and os.path.exists(ncnn_benchmark_bin):
        benchncnn_bin = ncnn_benchmark_bin
    else:
        benchncnn_bin = _find_benchncnn()
    if not benchncnn_bin:
        raise FileNotFoundError(
            "benchncnn not found. Set --ncnn-benchmark-bin or NCNN_BENCHNCNN_BIN."
        )

    h, w = _model_hw(model_name)
    loop_count = int(num_warmup) + int(num_runs)
    gpu_device = "0" if runtime == "ncnn_vulkan" else "-1"
    cmd = [
        benchncnn_bin,
        str(loop_count),
        "2",   # num threads
        "0",   # powersave
        gpu_device,
        "0",   # cooling down
        f"param={param_path}",
        f"shape=[{h},{w},3]",
    ]

    rc, out = _run_cmd(cmd, timeout_sec=max(600, int(loop_count) * 3))
    if rc != 0:
        raise RuntimeError(
            f"benchncnn failed rc={rc}. cmd={shlex.join(cmd)}\n{out[-4000:]}"
        )

    records = _parse_ncnn_layer_lines(out)
    if not records:
        macro = _detect_ncnn_benchmark_macro(benchncnn_bin)
        detail = (
            f"NCNN_BENCHMARK={macro}" if macro is not None else "NCNN_BENCHMARK unknown"
        )
        raise RuntimeError(
            "No per-layer timings in benchncnn output. "
            f"{detail}. Rebuild ncnn with -DNCNN_BENCHMARK=ON "
            "in a separate build dir and set NCNN_BENCHNCNN_BIN."
        )

    return {
        "status": "ok",
        "backend": "ncnn",
        "granularity": "ncnn_layer",
        "records": records,
        "tool_path": benchncnn_bin,
        "param_path": param_path,
        "bin_path": bin_path,
    }


def _resolve_trt_engine_path(model_name, runtime, model_dir):
    precision = runtime.split("_")[-1]
    candidates = [
        os.path.join(model_dir, f"{model_name}_{precision}.engine"),
    ]
    if model_name == "ssd_mobilenet_v2":
        candidates.extend(
            [
                os.path.join(model_dir, f"{model_name}_effnms_fpinput_{precision}.engine"),
                os.path.join(model_dir, f"{model_name}_effnms_{precision}.engine"),
            ]
        )
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"TRT engine not found for {model_name}/{runtime}: {candidates}")


def _load_pytorch_model(model_name, runtime, model_dir):
    import torch

    use_cuda = runtime == "pytorch_cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if model_name in CLASSIFICATION_FACTORIES:
        import torchvision.models as tvm

        factory = getattr(tvm, CLASSIFICATION_FACTORIES[model_name])
        # torchvision version compatibility
        try:
            model = factory(weights="DEFAULT")
        except Exception:
            try:
                model = factory(pretrained=True)
            except Exception:
                model = factory()
        input_shape = (1, 3, 224, 224)
    elif model_name in DETECTION_INPUT_SIZES:
        ts_path = os.path.join(model_dir, f"{model_name}.torchscript")
        if not os.path.exists(ts_path):
            raise FileNotFoundError(f"TorchScript not found: {ts_path}")
        model = torch.jit.load(ts_path, map_location=device)
        input_shape = DETECTION_INPUT_SIZES[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval().to(device)
    x = torch.randn(*input_shape, device=device)
    return model, x, device


def _profile_pytorch(model_name, runtime, model_dir, num_warmup, num_runs):
    import torch

    model, x, device = _load_pytorch_model(model_name, runtime, model_dir)

    # warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    records = []
    backend = "pytorch"
    granularity = "op"

    # Prefer torch.profiler. Fallback to autograd profiler for older torch builds.
    try:
        from torch.profiler import profile, ProfilerActivity
        import gc

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        # Split num_runs into smaller sessions to cap peak RAM/trace size.
        # Single 50-run session on 4GB Jetson can OOM during key_averages()
        # aggregation for large models (e.g., efficientnet, ssd). 10 runs × 5
        # sessions keeps each trace manageable; between sessions we dump the
        # trace and force GC so memory returns to baseline.
        session_size = int(os.environ.get("LAYER_PROFILE_SESSION_SIZE", "10"))
        session_size = max(1, min(session_size, max(1, num_runs)))
        num_sessions = max(1, (num_runs + session_size - 1) // session_size)

        trace_dir = os.environ.get("LAYER_TRACE_DIR")
        per_key_tot = {}            # key -> {"count", "cpu_us", "cuda_us"}
        per_key_sess_means = {}     # key -> [ms per session]

        remaining = num_runs
        for s_idx in range(num_sessions):
            runs_this = min(session_size, remaining)
            if runs_this <= 0:
                break
            remaining -= runs_this

            with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
                with torch.inference_mode():
                    for _ in range(runs_this):
                        _ = model(x)
                    if device.type == "cuda":
                        torch.cuda.synchronize()

            for ev in prof.key_averages():
                count = int(getattr(ev, "count", 0) or 0)
                if count <= 0:
                    continue
                key = str(getattr(ev, "key", "unknown"))
                cpu_us = float(getattr(ev, "self_cpu_time_total", 0.0) or 0.0)
                cuda_us = float(getattr(ev, "self_cuda_time_total", 0.0) or 0.0)
                total_us = cuda_us if (device.type == "cuda" and cuda_us > 0.0) else cpu_us
                sess_mean_ms = total_us / count / 1000.0
                if sess_mean_ms <= 0.0:
                    continue
                tot = per_key_tot.setdefault(key, {"count": 0, "cpu_us": 0.0, "cuda_us": 0.0})
                tot["count"] += count
                tot["cpu_us"] += cpu_us
                tot["cuda_us"] += cuda_us
                per_key_sess_means.setdefault(key, []).append(sess_mean_ms)

            # Dump trace to disk (optional) and release the profile.
            if trace_dir:
                try:
                    os.makedirs(trace_dir, exist_ok=True)
                    prof.export_chrome_trace(
                        os.path.join(trace_dir, f"{model_name}_{runtime}_s{s_idx}.json")
                    )
                except Exception:
                    pass
            del prof
            gc.collect()

        for key, tot in per_key_tot.items():
            count = tot["count"]
            total_us = tot["cuda_us"] if (device.type == "cuda" and tot["cuda_us"] > 0.0) else tot["cpu_us"]
            mean_ms = total_us / count / 1000.0
            sess_means = per_key_sess_means.get(key, [])
            min_ms = float(min(sess_means)) if sess_means else None
            max_ms = float(max(sess_means)) if sess_means else None
            std_ms = float(np.std(sess_means)) if len(sess_means) >= 2 else 0.0
            records.append(
                {
                    "layer_name": key,
                    "backend": backend,
                    "granularity": granularity,
                    "mean_ms": round(float(mean_ms), 6),
                    "std_ms": round(std_ms, 6),
                    "min_ms": round(min_ms, 6) if min_ms is not None else None,
                    "max_ms": round(max_ms, 6) if max_ms is not None else None,
                    "samples": count,
                }
            )
    except Exception:
        # Old API fallback
        with torch.autograd.profiler.profile(use_cuda=(device.type == "cuda")) as prof:
            with torch.inference_mode():
                for _ in range(num_runs):
                    _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        events = prof.key_averages()
        for ev in events:
            count = int(getattr(ev, "count", 0) or 0)
            if count <= 0:
                continue
            cpu_total_us = float(getattr(ev, "self_cpu_time_total", 0.0) or 0.0)
            cuda_total_us = float(getattr(ev, "self_cuda_time_total", 0.0) or 0.0)
            total_us = cuda_total_us if (device.type == "cuda" and cuda_total_us > 0.0) else cpu_total_us
            mean_ms = total_us / count / 1000.0
            if mean_ms <= 0.0:
                continue
            records.append(
                {
                    "layer_name": str(getattr(ev, "key", "unknown")),
                    "backend": backend,
                    "granularity": granularity,
                    "mean_ms": round(float(mean_ms), 6),
                    "std_ms": 0.0,
                    "min_ms": None,
                    "max_ms": None,
                    "samples": count,
                }
            )

    # records.sort(key=lambda r: r["mean_ms"], reverse=True)
    return {
        "status": "ok",
        "backend": backend,
        "granularity": granularity,
        "records": records,
        "device_used": str(device),
    }


def _profile_tensorrt(model_name, runtime, model_dir, num_warmup, num_runs):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    engine_path = _resolve_trt_engine_path(model_name, runtime, model_dir)
    logger = trt.Logger(trt.Logger.WARNING)

    with open(engine_path, "rb") as f:
        runtime_trt = trt.Runtime(logger)
        engine = runtime_trt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TRT execution context")

    class LayerProfiler(trt.IProfiler):
        def __init__(self):
            super().__init__()
            self.times = defaultdict(list)

        def report_layer_time(self, layer_name, ms):
            self.times[str(layer_name)].append(float(ms))

    profiler = LayerProfiler()
    context.profiler = profiler

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for i in range(engine.num_bindings):
        shape = tuple(engine.get_binding_shape(i))
        if any(int(d) < 0 for d in shape):
            raise RuntimeError(f"Dynamic binding shape not supported in this profiler: {shape}")
        dtype = trt.nptype(engine.get_binding_dtype(i))
        size = int(trt.volume(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        dev_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(dev_mem))

        if engine.binding_is_input(i):
            host_mem[:] = np.random.randn(size).astype(dtype)
            inputs.append((host_mem, dev_mem))
        else:
            outputs.append((host_mem, dev_mem))

    def infer_once():
        for host_mem, dev_mem in inputs:
            cuda.memcpy_htod_async(dev_mem, host_mem, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for host_mem, dev_mem in outputs:
            cuda.memcpy_dtoh_async(host_mem, dev_mem, stream)
        stream.synchronize()

    for _ in range(num_warmup):
        infer_once()
    for _ in range(num_runs):
        infer_once()

    records = []
    for layer_name, values in profiler.times.items():
        arr = np.array(values, dtype=np.float64)
        if arr.size == 0:
            continue
        records.append(
            {
                "layer_name": layer_name,
                "backend": "tensorrt",
                "granularity": "trt_layer",
                "mean_ms": round(float(arr.mean()), 6),
                "std_ms": round(float(arr.std()), 6),
                "min_ms": round(float(arr.min()), 6),
                "max_ms": round(float(arr.max()), 6),
                "samples": int(arr.size),
            }
        )

    # records.sort(key=lambda r: r["mean_ms"], reverse=True)
    return {
        "status": "ok",
        "backend": "tensorrt",
        "granularity": "trt_layer",
        "records": records,
        "engine_path": engine_path,
    }


def _resolve_onnx_path(model_name, model_dir):
    candidates = [os.path.join(model_dir, f"{model_name}.onnx")]
    if model_name == "ssd_mobilenet_v2":
        candidates = [
            os.path.join(model_dir, "ssd_mobilenet_v2_raw.onnx"),
            os.path.join(model_dir, "ssd_mobilenet_v2_effnms.onnx"),
            os.path.join(model_dir, "ssd_mobilenet_v2.onnx"),
        ] + candidates
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"ONNX not found for {model_name}: {candidates}")


def _profile_onnxrt(model_name, runtime, model_dir, num_warmup, num_runs):
    import onnxruntime as ort

    onnx_path = _resolve_onnx_path(model_name, model_dir)
    provider = "CUDAExecutionProvider" if runtime == "onnxrt_cuda" else "TensorrtExecutionProvider"
    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(f"{provider} not available. Available={available}")

    so = ort.SessionOptions()
    so.enable_profiling = True
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(onnx_path, sess_options=so, providers=[provider])

    # Feed EVERY declared input. SSD with built-in NMS exposes auxiliary inputs
    # (score_threshold, iou_threshold, etc.) that ORT rejects with
    # "Unexpected input" if only inputs[0] is provided.
    _ORT_TYPE_TO_NP = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(bool)": np.bool_,
    }

    def _resolve_shape(dims):
        out = []
        for d in dims:
            if isinstance(d, str) or d is None:
                out.append(1)
            else:
                out.append(max(1, int(d)))
        return out

    def _random_tensor(dtype, shape):
        if np.issubdtype(dtype, np.floating):
            return np.random.randn(*shape).astype(dtype)
        if np.issubdtype(dtype, np.integer):
            return np.zeros(shape, dtype=dtype)
        if dtype is np.bool_:
            return np.zeros(shape, dtype=np.bool_)
        return np.zeros(shape, dtype=np.float32)

    feed = {}
    for inp in session.get_inputs():
        dtype = _ORT_TYPE_TO_NP.get(str(inp.type), np.float32)
        shape = _resolve_shape(inp.shape)
        feed[inp.name] = _random_tensor(dtype, shape)

    for _ in range(num_warmup):
        _ = session.run(None, feed)
    for _ in range(num_runs):
        _ = session.run(None, feed)

    profile_path = session.end_profiling()

    op_times = defaultdict(list)
    try:
        with open(profile_path, "r") as f:
            events = json.load(f)
        for ev in events:
            if not isinstance(ev, dict):
                continue
            dur = ev.get("dur")
            if dur is None:
                continue
            # ORT profiling duration is typically in microseconds.
            dur_ms = float(dur) / 1000.0
            if dur_ms <= 0.0:
                continue
            name = str(ev.get("name", "unknown"))
            args = ev.get("args", {}) if isinstance(ev.get("args"), dict) else {}
            op_name = args.get("op_name")
            if op_name:
                key = f"{op_name}:{name}"
            else:
                key = name
            op_times[key].append(dur_ms)
    finally:
        try:
            if os.path.exists(profile_path):
                os.remove(profile_path)
        except Exception:
            pass

    records = []
    for name, values in op_times.items():
        arr = np.array(values, dtype=np.float64)
        if arr.size == 0:
            continue
        records.append(
            {
                "layer_name": name,
                "backend": "onnxruntime",
                "granularity": "onnx_node",
                "mean_ms": round(float(arr.mean()), 6),
                "std_ms": round(float(arr.std()), 6),
                "min_ms": round(float(arr.min()), 6),
                "max_ms": round(float(arr.max()), 6),
                "samples": int(arr.size),
            }
        )
    # records.sort(key=lambda r: r["mean_ms"], reverse=True)
    return {
        "status": "ok",
        "backend": "onnxruntime",
        "granularity": "onnx_node",
        "records": records,
        "onnx_path": onnx_path,
        "provider": provider,
    }


def profile_layer_runtime(
    model_name,
    runtime,
    model_dir,
    num_warmup,
    num_runs,
    tflite_benchmark_bin="",
    ncnn_benchmark_bin="",
):
    if runtime in ("pytorch_cpu", "pytorch_cuda"):
        return _profile_pytorch(model_name, runtime, model_dir, num_warmup, num_runs)
    if runtime in ("tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8"):
        return _profile_tensorrt(model_name, runtime, model_dir, num_warmup, num_runs)
    if runtime in ("onnxrt_cuda", "onnxrt_trt"):
        return _profile_onnxrt(model_name, runtime, model_dir, num_warmup, num_runs)
    if runtime in ("tflite_cpu", "tflite_gpu"):
        return _profile_tflite(
            model_name,
            runtime,
            model_dir,
            num_warmup,
            num_runs,
            tflite_benchmark_bin=tflite_benchmark_bin,
        )
    if runtime in ("ncnn_cpu", "ncnn_vulkan"):
        return _profile_ncnn(
            model_name,
            runtime,
            model_dir,
            num_warmup,
            num_runs,
            ncnn_benchmark_bin=ncnn_benchmark_bin,
        )

    return {
        "status": "na",
        "backend": "unsupported",
        "granularity": "na",
        "records": [],
        "error": f"N/A: runtime {runtime} has no stable per-layer profiler in this pipeline",
    }


def _write_csv(path, model_name, runtime, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "model",
        "runtime",
        "rank",
        "layer_name",
        "backend",
        "granularity",
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "samples",
        "percent_total_ms",
        "cumulative_percent_ms",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["model"] = model_name
            row["runtime"] = runtime
            writer.writerow(row)


def _enrich_records(records):
    """Add rank and contribution columns to per-layer timing records."""
    if not records:
        return []

    enriched = [dict(r) for r in records]
    enriched.sort(key=lambda r: float(r.get("mean_ms") or 0.0), reverse=True)
    total_ms = sum(float(r.get("mean_ms") or 0.0) for r in enriched)
    cumulative = 0.0

    for idx, row in enumerate(enriched, 1):
        mean_ms = float(row.get("mean_ms") or 0.0)
        cumulative += mean_ms
        row["rank"] = idx
        row["percent_total_ms"] = (
            round((mean_ms / total_ms) * 100.0, 6) if total_ms > 0 else 0.0
        )
        row["cumulative_percent_ms"] = (
            round((cumulative / total_ms) * 100.0, 6) if total_ms > 0 else 0.0
        )
    return enriched


def main():
    parser = argparse.ArgumentParser(description="Runtime-aware layer latency profiler")
    parser.add_argument("--model", required=True, choices=ALL_SUPPORTED_MODELS)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=50)
    parser.add_argument(
        "--tflite-benchmark-bin",
        default=os.environ.get("TFLITE_BENCHMARK_MODEL_BIN", ""),
    )
    parser.add_argument(
        "--ncnn-benchmark-bin",
        default=os.environ.get("NCNN_BENCHNCNN_BIN", ""),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    meta = {
        "model": args.model,
        "runtime": args.runtime,
        "num_warmup": args.num_warmup,
        "num_runs": args.num_runs,
        "status": "error",
        "backend": "",
        "granularity": "",
        "records_count": 0,
        "error": "",
    }

    try:
        result = profile_layer_runtime(
            model_name=args.model,
            runtime=args.runtime,
            model_dir=args.model_dir,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            tflite_benchmark_bin=args.tflite_benchmark_bin,
            ncnn_benchmark_bin=args.ncnn_benchmark_bin,
        )
        records = _enrich_records(result.get("records", []))
        meta["status"] = result.get("status", "error")
        meta["backend"] = result.get("backend", "")
        meta["granularity"] = result.get("granularity", "")
        meta["records_count"] = len(records)
        meta["error"] = result.get("error", "")
        meta["schema_version"] = "layer_runtime_v2"
        meta["records"] = records
        meta["total_mean_ms"] = round(
            sum(float(r.get("mean_ms") or 0.0) for r in records), 6
        )
        for k in (
            "device_used",
            "engine_path",
            "onnx_path",
            "provider",
            "tool_path",
            "model_path",
            "param_path",
            "bin_path",
        ):
            if k in result:
                meta[k] = result[k]

        if meta["status"] == "ok":
            _write_csv(args.output_csv, args.model, args.runtime, records)
        else:
            # Keep output contract deterministic even for N/A/error
            _write_csv(args.output_csv, args.model, args.runtime, [])
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = str(e)
        meta["schema_version"] = "layer_runtime_v2"
        meta["records"] = []
        meta["total_mean_ms"] = 0.0
        _write_csv(args.output_csv, args.model, args.runtime, [])

    with open(args.output_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[layer-runtime] {args.model}/{args.runtime} "
        f"status={meta['status']} records={meta['records_count']}"
    )
    if meta["error"]:
        print(f"  error: {meta['error']}")


if __name__ == "__main__":
    main()
