#!/usr/bin/env python3
"""
quick_verify.py — 6모델 × 11런타임 경량 검증 (1 warmup + 3 iters)
- 목적: 실제 수치가 나오는지 빠르게 확인 (전체 벤치마크 전 smoke test)
- 방식: 세션/인터프리터를 한 번만 만들고 재사용 (TRT 엔진 재빌드 방지)
- 소요: ~20-40분 (전체 벤치마크의 1/10 수준)
"""

import os, sys, time, json, traceback
import multiprocessing as mp
import gc
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
    # SSD has non-standard filenames — try several alternatives
    onnx_candidates = [f"{model}.onnx"]
    if model == "ssd_mobilenet_v2":
        onnx_candidates = [
            "ssd_mobilenet_v2_raw_fpinput.onnx",  # NHWC, works
            "ssd_mobilenet_v2.onnx",
        ]

    files["onnx"] = None
    for fn in onnx_candidates:
        p = os.path.join(MODELS_DIR, fn)
        if os.path.exists(p):
            files["onnx"] = p
            break

    for k, fn in {
        "tflite": f"{model}.tflite",
        "ncnn_param": f"{model}.ncnn.param",
        "ncnn_bin": f"{model}.ncnn.bin",
        "engine_fp32": f"{model}_fp32.engine",
        "engine_fp16": f"{model}_fp16.engine",
        "engine_int8": f"{model}_int8.engine",
    }.items():
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
            trt_opts = {
                "trt_max_workspace_size": 1 << 28,
                "trt_fp16_enable": False,
                "trt_int8_enable": False,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": os.path.join(PROJ_DIR, "models", "trt_cache"),
            }
            providers = [("TensorrtExecutionProvider", trt_opts),
                         "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif runtime == "tensorrt_fp16":
            trt_opts = {
                "trt_max_workspace_size": 1 << 28,
                "trt_fp16_enable": True,
                "trt_int8_enable": False,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": os.path.join(PROJ_DIR, "models", "trt_cache"),
            }
            providers = [("TensorrtExecutionProvider", trt_opts),
                         "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif runtime == "tensorrt_int8":
            trt_opts = {
                "trt_max_workspace_size": 1 << 28,
                "trt_fp16_enable": True,
                "trt_int8_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": os.path.join(PROJ_DIR, "models", "trt_cache"),
            }
            providers = [("TensorrtExecutionProvider", trt_opts),
                         "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif runtime == "onnxrt_trt":
            trt_opts = {
                "trt_max_workspace_size": 1 << 28,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": os.path.join(PROJ_DIR, "models", "trt_cache"),
            }
            providers = [("TensorrtExecutionProvider", trt_opts),
                         "CUDAExecutionProvider", "CPUExecutionProvider"]
        else:  # onnxrt_cuda
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        sess = ort.InferenceSession(onnx_p, providers=providers)
        inp_info = sess.get_inputs()[0]
        input_name = inp_info.name
        # Detect NHWC: shape like [1, H, W, 3] — last dim is channels
        inp_shape = inp_info.shape
        is_nhwc = (len(inp_shape) == 4 and
                   isinstance(inp_shape[3], int) and inp_shape[3] == 3)
        return ("ort", sess, (input_name, is_nhwc))

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
        # Don't load net here — ncnn objects can't be pickled.
        # Pass file paths; subprocess loads net fresh each call.
        input_name = _ncnn_input_name(param_p)
        use_vulkan = (runtime == "ncnn_vulkan")
        return ("ncnn", None, (param_p, bin_p, input_name, use_vulkan))

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
        input_name, is_nhwc = meta
        inp = dummy_input
        if is_nhwc:
            # NCHW (1,C,H,W) → NHWC (1,H,W,C)
            inp = dummy_input.transpose(0, 2, 3, 1)
        return sess.run(None, {input_name: inp})

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
        # Run in subprocess to isolate segfaults
        param_p, bin_p, input_name, use_vulkan = meta
        result_q = mp.Queue()
        p = mp.Process(target=_ncnn_worker,
                       args=(param_p, bin_p, input_name, use_vulkan, dummy_input, result_q))
        p.start()
        p.join(timeout=180)
        if p.is_alive():
            p.kill(); p.join()
            raise RuntimeError("ncnn: timeout (>60s)")
        if p.exitcode != 0:
            raise RuntimeError(f"ncnn: process died (segfault? exitcode={p.exitcode})")
        status, payload = result_q.get()
        if status == "ok":
            return [payload]
        raise RuntimeError(f"ncnn: {payload}")

    else:
        raise ValueError(f"Unknown session type: {stype}")


def _ncnn_worker(param_p, bin_p, input_name, use_vulkan, dummy_input, result_q):
    """
    Runs inside a subprocess so segfaults don't kill the parent.
    Loads ncnn net fresh, runs one inference, puts result in queue.
    """
    try:
        import ncnn
        import numpy as np
        net = ncnn.Net()
        if use_vulkan:
            net.opt.use_vulkan_compute = True
        net.load_param(param_p)
        net.load_model(bin_p)

        ex = net.create_extractor()
        # ncnn expects NCHW without batch dim: (C, H, W)
        arr = dummy_input[0]  # shape: (C, H, W)
        C, H, W = arr.shape
        mat_in = ncnn.Mat(W, H, C, arr.astype(np.float32).tobytes())
        ex.input(input_name, mat_in)

        # Try common output names
        output_names = ["out0", "output", "prob", "1000", "softmax",
                        "cls_score", "detection_out", "boxes", "scores", "output0"]
        out_arr = None
        for oname in output_names:
            try:
                ret, mat_out = ex.extract(oname)
                if ret == 0 and mat_out is not None:
                    out_arr = np.array(mat_out).flatten()
                    break
            except Exception:
                continue

        if out_arr is None:
            result_q.put(("error", "no output extracted from any known name"))
        else:
            result_q.put(("ok", out_arr))
    except Exception as e:
        result_q.put(("error", str(e)))


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
            # ncnn은 메모리 격리를 위해 subprocess로 분리 실행
            if runtime.startswith("ncnn_"):
                import subprocess
                files = check_model_files(model)
                if not files["ncnn_param"] or not files["ncnn_bin"]:
                    raise FileNotFoundError(f"Missing ncnn files for {model}")
                input_name = _ncnn_input_name(files["ncnn_param"])
                size = 640 if "yolov8" in model else (320 if "ssd" in model else 224)
                payload = {
                    "param": files["ncnn_param"],
                    "bin": files["ncnn_bin"],
                    "input_name": input_name,
                    "use_vulkan": (runtime == "ncnn_vulkan"),
                    "shape": [3, size, size],
                }
                proc = subprocess.run(
                    [sys.executable, os.path.join(BENCH_DIR, "ncnn_run.py")],
                    input=json.dumps(payload), capture_output=True, text=True, timeout=240
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"ncnn subprocess exit={proc.returncode}: {proc.stderr[:200]}")
                try:
                    # Find last JSON-looking line (ncnn Vulkan prints init logs to stdout)
                    json_line = None
                    for line in reversed(proc.stdout.strip().split(chr(10))):
                        line = line.strip()
                        if line.startswith("{") and line.endswith("}"):
                            json_line = line
                            break
                    if not json_line:
                        raise ValueError("no JSON line found")
                    res = json.loads(json_line)
                except Exception:
                    raise RuntimeError(f"ncnn bad output: {proc.stdout[:200]}")
                if "error" in res:
                    raise RuntimeError(f"ncnn: {res['error']}")
                lat_mean = res["lat_ms"]
                ok, msg = True, f"ok shape={tuple(res['shape'])}"
                session_tuple = None
            else:
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
        try:
            del session_tuple
        except NameError:
            pass
        gc.collect()

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
