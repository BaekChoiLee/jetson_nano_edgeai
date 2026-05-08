#!/usr/bin/env python3
"""
quick_verify.py — 6모델 × 11런타임 경량 검증 (1 warmup + 3 iters)
- 목적: 실제 수치가 나오는지 빠르게 확인 (전체 벤치마크 전 smoke test)
- 방식: 세션/인터프리터를 한 번만 만들고 재사용 (TRT 엔진 재빌드 방지)
- 소요: ~20-40분 (전체 벤치마크의 1/10 수준)
"""

import os, sys, time, json, traceback
import numpy as np

BENCH_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR   = os.path.dirname(BENCH_DIR)
MODELS_DIR = os.path.join(PROJ_DIR, "models")
DATA_DIR   = os.path.join(PROJ_DIR, "data")

MODELS = [
    "mobilenetv3_small",
    "efficientnet_b0",
    "shufflenet_v2_x1_0",
    "resnet50",
    "yolov8n",
    "ssd_mobilenet_v2",
]
RUNTIMES = [
    "pytorch_cpu", "pytorch_cuda",
    "tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8",
    "onnxrt_cuda", "onnxrt_trt",
    "tflite_cpu", "tflite_gpu",
    "ncnn_cpu", "ncnn_vulkan",
]
WARMUP = 1
ITERS  = 3

OUT_DIR = os.path.join(PROJ_DIR, "results", "quick_verify")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, BENCH_DIR)


# ── 모델 파일 존재 여부 확인 ──────────────────────────────
def check_model_files(model):
    files = {}
    exts = {
        "onnx": f"{model}.onnx",
        "tflite": f"{model}.tflite",
        "ncnn_param": f"{model}.ncnn.param",
        "ncnn_bin": f"{model}.ncnn.bin",
        "engine_fp32": f"{model}_fp32.engine",
        "engine_fp16": f"{model}_fp16.engine",
        "engine_int8": f"{model}_int8.engine",
    }
    for k, fn in exts.items():
        p = os.path.join(MODELS_DIR, fn)
        files[k] = p if os.path.exists(p) else None
    return files


# ── 세션/인터프리터 생성 (1회) ────────────────────────────
def make_session(model, runtime):
    """
    Returns (session_obj, input_name_or_detail, runtime_type)
    runtime_type: 'ort', 'tflite', 'ncnn'
    Raises: FileNotFoundError, Exception
    """
    files = check_model_files(model)

    # ── PyTorch / ONNX Runtime (CPU/CUDA/TRT EP) ─────────
    if runtime in ("pytorch_cpu", "pytorch_cuda",
                   "tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8",
                   "onnxrt_cuda", "onnxrt_trt"):

        import onnxruntime as ort

        onnx_p = files["onnx"]
        if not onnx_p:
            raise FileNotFoundError(f"Missing: {model}.onnx")

        if runtime == "pytorch_cpu":
            import torch; torch.set_num_threads(1)
            providers = ["CPUExecutionProvider"]
        elif runtime == "pytorch_cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif runtime == "tensorrt_fp32":
            # TRT EP: fp32 (no fp16/int8), 256MB workspace
            trt_opts = {
                "trt_max_workspace_size": 1 << 28,  # 256MB
                "trt_fp16_enable": False,
                "trt_int8_enable": False,
            }
            providers = [("TensorrtExecutionProvider", trt_opts),
                         "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif runtime == "tensorrt_fp16":
            trt_opts = {
                "trt_max_workspace_size": 1 << 28,
                "trt_fp16_enable": True,
                "trt_int8_enable": False,
            }
            providers = [("TensorrtExecutionProvider", trt_opts),
                         "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif runtime == "tensorrt_int8":
            trt_opts = {
                "trt_max_workspace_size": 1 << 28,
                "trt_fp16_enable": True,
                "trt_int8_enable": True,
            }
            providers = [("TensorrtExecutionProvider", trt_opts),
                         "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif runtime == "onnxrt_trt":
            providers = ["TensorrtExecutionProvider",
                         "CUDAExecutionProvider", "CPUExecutionProvider"]
        else:  # onnxrt_cuda
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        sess = ort.InferenceSession(onnx_p, providers=providers)
        input_name = sess.get_inputs()[0].name
        return ("ort", sess, input_name)

    # ── TFLite ───────────────────────────────────────────
    elif runtime.startswith("tflite_"):
        tflite_p = files["tflite"]
        if not tflite_p:
            raise FileNotFoundError(f"Missing: {model}.tflite")
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            load_delegate = tf.lite.experimental.load_delegate

        if runtime == "tflite_gpu":
            try:
                gpu_del = load_delegate("libdelegate.so")
                interp = Interpreter(tflite_p, experimental_delegates=[gpu_del])
            except Exception:
                interp = Interpreter(tflite_p)
        else:
            interp = Interpreter(tflite_p, num_threads=1)

        interp.allocate_tensors()
        return ("tflite", interp, None)

    # ── ncnn ─────────────────────────────────────────────
    elif runtime.startswith("ncnn_"):
        param_p = files["ncnn_param"]
        bin_p   = files["ncnn_bin"]
        if not param_p or not bin_p:
            raise FileNotFoundError(f"Missing ncnn files for {model}")
        import ncnn
        net = ncnn.Net()
        if runtime == "ncnn_vulkan":
            net.opt.use_vulkan_compute = True
        net.load_param(param_p)
        net.load_model(bin_p)
        input_name = _ncnn_input_name(param_p)
        return ("ncnn", net, input_name)

    else:
        raise ValueError(f"Unknown runtime: {runtime}")


# ── 단일 추론 (세션 재사용) ───────────────────────────────
def run_infer(session_tuple, dummy_input):
    """
    session_tuple: (type, session_obj, meta)
    dummy_input: np.float32 (1, C, H, W) NCHW
    Returns: list of np arrays
    """
    stype, sess, meta = session_tuple

    if stype == "ort":
        return sess.run(None, {meta: dummy_input})

    elif stype == "tflite":
        interp = sess
        in_d = interp.get_input_details()[0]
        inp = dummy_input
        if len(in_d["shape"]) == 4 and in_d["shape"][3] == dummy_input.shape[1]:
            inp = dummy_input.transpose(0, 2, 3, 1)
        interp.set_tensor(in_d["index"], inp.astype(in_d["dtype"]))
        interp.invoke()
        return [interp.get_tensor(od["index"]) for od in interp.get_output_details()]

    elif stype == "ncnn":
        net = sess
        input_name = meta
        ex = net.create_extractor()
        chw = dummy_input[0]
        c, h, w = chw.shape
        import ncnn
        mat = ncnn.Mat(w, h, c)
        arr = np.array(mat, copy=False)
        if arr.shape == chw.shape:
            arr[...] = chw
        else:
            for ci in range(c):
                np.array(mat.channel(ci), copy=False)[...] = chw[ci]
        ex.input(input_name, mat)
        for out_name in ["output", "prob", "1000", "softmax", "cls_score",
                         "detection_out", "boxes", "scores"]:
            ret, out_mat = ex.extract(out_name)
            if ret == 0:
                return [np.array(out_mat).flatten()]
        raise RuntimeError(f"ncnn: no extractable output for {net}")

    else:
        raise ValueError(f"Unknown session type: {stype}")


def _ncnn_input_name(param_path):
    try:
        with open(param_path) as f:
            for line in f:
                if line.strip().startswith("Input"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
    except Exception:
        pass
    return "input"


def make_dummy(model):
    size = 640 if "yolov8" in model else (320 if "ssd" in model else 224)
    return np.random.randn(1, 3, size, size).astype(np.float32), size


def validate_output(outputs, model):
    if not outputs:
        return False, "empty output"
    out = outputs[0]
    arr = np.asarray(out).flatten()
    if len(arr) == 0:
        return False, "zero-length output"
    if not np.isfinite(arr).all():
        nan_pct = np.isnan(arr).mean() * 100
        return False, f"{nan_pct:.0f}% NaN/Inf in output"
    if np.allclose(arr, 0.0):
        return False, "all-zero output"
    return True, f"ok shape={np.asarray(out).shape}"


# ── 메인 ─────────────────────────────────────────────────
results = {}
summary = {"pass": 0, "fail": 0, "skip": 0}

print(f"\n{'='*70}", flush=True)
print(f"QUICK VERIFY — {len(MODELS)} models × {len(RUNTIMES)} runtimes", flush=True)
print(f"Warmup={WARMUP}  Iters={ITERS}  (session created ONCE per cell)", flush=True)
print(f"{'='*70}\n", flush=True)

for model in MODELS:
    results[model] = {}
    dummy, size = make_dummy(model)

    for runtime in RUNTIMES:
        tag = f"{model:30s} × {runtime}"
        cell = {"status": "fail", "lat_ms": None, "error": None}

        try:
            # 세션 1회 생성 (TRT 엔진 빌드 포함)
            session_tuple = make_session(model, runtime)

            # warmup
            for _ in range(WARMUP):
                out = run_infer(session_tuple, dummy)

            # measure
            lats = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                out = run_infer(session_tuple, dummy)
                lats.append((time.perf_counter() - t0) * 1000)

            ok, msg = validate_output(out, model)
            lat_mean = float(np.mean(lats))

            if ok:
                cell = {"status": "pass", "lat_ms": round(lat_mean, 2), "msg": msg}
                summary["pass"] += 1
                print(f"  ✓ {tag}  {lat_mean:7.1f} ms   {msg}", flush=True)
            else:
                cell = {"status": "fail", "lat_ms": round(lat_mean, 2), "error": msg}
                summary["fail"] += 1
                print(f"  ✗ {tag}  ran but bad output: {msg}", flush=True)

        except FileNotFoundError as e:
            cell = {"status": "skip", "error": str(e)}
            summary["skip"] += 1
            print(f"  - {tag}  SKIP (no file): {e}", flush=True)
        except Exception as e:
            cell = {"status": "fail", "error": str(e)[:120]}
            summary["fail"] += 1
            print(f"  ✗ {tag}  ERROR: {str(e)[:100]}", flush=True)

        results[model][runtime] = cell

# ── 결과 매트릭스 ─────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print("MATRIX  (✓=pass  ✗=fail  -=skip)", flush=True)
print(f"{'='*70}", flush=True)
rt_short = [r.replace("pytorch_","pt_").replace("tensorrt_","trt_")
             .replace("onnxrt_","ort_").replace("tflite_","tf_")
             .replace("ncnn_","nc_") for r in RUNTIMES]
print(f"{'Model':<26}" + "".join(f"{r:<10}" for r in rt_short), flush=True)
print("-" * (26 + 10 * len(RUNTIMES)), flush=True)

for model in MODELS:
    row = f"{model:<26}"
    for rt in RUNTIMES:
        st = results[model][rt]["status"]
        sym = "✓" if st == "pass" else ("-" if st == "skip" else "✗")
        lat = results[model][rt].get("lat_ms")
        cell_str = f"{sym}{'('+str(lat)+'ms)' if lat else ''}"[:9]
        row += f"{cell_str:<10}"
    print(row, flush=True)

total = summary["pass"] + summary["fail"] + summary["skip"]
print(f"\n  PASS={summary['pass']}  FAIL={summary['fail']}  SKIP={summary['skip']}  / {total} cells", flush=True)

# ── 저장 ─────────────────────────────────────────────────
out_json = os.path.join(OUT_DIR, "quick_verify.json")
with open(out_json, "w") as f:
    json.dump({"results": results, "summary": summary}, f, indent=2)
print(f"\n  Saved: {out_json}", flush=True)

# fail/skip 요약
print(f"\n{'─'*70}", flush=True)
print("FAILURES / SKIPS:", flush=True)
for model in MODELS:
    for rt in RUNTIMES:
        c = results[model][rt]
        if c["status"] != "pass":
            print(f"  [{c['status'].upper():4s}] {model} × {rt}: {c.get('error','')[:80]}", flush=True)
