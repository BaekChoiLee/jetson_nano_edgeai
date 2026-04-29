#!/usr/bin/env python3
# =============================================================================
# detection_infer.py — 12 런타임 × 2 검출 모델 infer 함수 팩토리
# =============================================================================
# 역할: 런타임(pytorch_cpu, tensorrt_fp16, ncnn_vulkan 등) 과 모델(yolov8n,
#       ssd_mobilenet_v2)을 받아 raw output 을 반환하는 infer 함수를 생성한다.
#       raw output 은 detection_postprocess.py 가 NMS/디코드를 수행한다.
#
# 12 런타임 커버리지:
#   pytorch_cpu, pytorch_cuda
#   tensorrt_fp32, tensorrt_fp16, tensorrt_int8
#   onnxrt_cuda, onnxrt_trt
#   tflite_cpu, tflite_gpu
#   ncnn_cpu, ncnn_vulkan, ncnn_vulkan_fixed
#
# 출력 스펙 (raw, 공정 비교를 위해 NMS 미적용):
#   yolov8n:          ndarray [1, 84, 8400]
#   ssd_mobilenet_v2: (box_encodings [1, N, 4], class_scores [1, N, 91])
#
# 설계 원칙:
#   - 각 런타임 로더는 lazy import (해당 패키지가 없는 환경에서도 다른 런타임 사용 가능)
#   - 모델 경로 해석은 _resolve_*_path() 헬퍼로 집중
#   - 세션/엔진은 팩토리 내부에서 1회 생성, 반환된 callable 이 재사용
# =============================================================================

"""Per-runtime raw infer function factories for YOLOv8n and SSD-MobileNet V2."""

import glob
import os

import numpy as np

# numpy 1.24+ 호환성: 구버전 TensorRT/pycuda가 np.bool 참조 가능
if not hasattr(np, "bool"):
    np.bool = np.bool_

# =============================================================================
# 모델 입력 크기 (preprocessing 에서 사용)
# =============================================================================
_MODEL_INPUT_HW = {
    "yolov8n":          (640, 640),
    "ssd_mobilenet_v2": (320, 320),
}


def get_input_hw(model_name):
    """모델별 (H, W) 반환. 미등록 모델은 (224, 224)."""
    return _MODEL_INPUT_HW.get(model_name, (224, 224))


def preprocess_image(image_bgr_uint8, model_name):
    """OpenCV BGR uint8 → 모델 입력용 NCHW float32.

    - YOLOv8n: RGB, [0, 1], letterbox 생략 (간단 resize)
    - SSD-MV2: RGB, [0, 1] (TF preprocess 는 letterbox 없이 resize)
    """
    import cv2  # lazy import

    h, w = get_input_hw(model_name)
    img = cv2.resize(image_bgr_uint8, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    # SSD MobileNet v2 (TF lineage) has baked-in /127.5 normalize; expects raw [0,255].
    # YOLOv8 expects [0,1].
    if model_name == 'yolov8n':
        img = img / 255.0
    # HWC → NCHW
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(img, dtype=np.float32)


# =============================================================================
# 런타임 디스패처
# =============================================================================
def get_detection_infer_fn(model_name, runtime, model_dir="models"):
    """런타임 × 모델 조합에 대한 raw infer 함수를 생성한다.

    Returns: callable(input_nchw_float32) -> raw_output
    """
    if runtime in ("pytorch_cpu", "pytorch_cuda"):
        device = "cuda" if runtime == "pytorch_cuda" else "cpu"
        return _make_torchscript_infer_fn(model_name, model_dir, device)

    if runtime in ("tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8"):
        precision = runtime.split("_")[-1]
        return _make_trt_infer_fn(model_name, model_dir, precision)

    if runtime in ("onnxrt_cuda", "onnxrt_trt"):
        provider = (
            "CUDAExecutionProvider" if runtime == "onnxrt_cuda"
            else "TensorrtExecutionProvider"
        )
        return _make_onnxrt_infer_fn(model_name, model_dir, provider)

    if runtime in ("tflite_cpu", "tflite_gpu"):
        use_gpu = runtime == "tflite_gpu"
        return _make_tflite_infer_fn(model_name, model_dir, use_gpu)

    if runtime in ("ncnn_cpu", "ncnn_vulkan", "ncnn_vulkan_fixed"):
        use_vulkan = runtime in ("ncnn_vulkan", "ncnn_vulkan_fixed")
        fixed = runtime == "ncnn_vulkan_fixed"
        return _make_ncnn_infer_fn(model_name, model_dir, use_vulkan, fixed)

    raise ValueError(f"Unknown runtime: {runtime}")


# =============================================================================
# TorchScript (PyTorch CPU/CUDA)
# =============================================================================
def _make_torchscript_infer_fn(model_name, model_dir, device):
    import torch

    ts_path = os.path.join(model_dir, f"{model_name}.torchscript")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"TorchScript not found: {ts_path}")

    model = torch.jit.load(ts_path, map_location=device)
    model.eval()

    def infer(x_np):
        with torch.inference_mode():
            x = torch.from_numpy(x_np).to(device)
            out = model(x)
        return _torch_out_to_raw(out, model_name)

    return infer


def _torch_out_to_raw(out, model_name):
    """TorchScript 출력을 후처리 모듈이 기대하는 형태로 정규화."""
    if model_name == "yolov8n":
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.detach().cpu().numpy()
    # ssd_mv2: (box_encodings, class_scores) 튜플 기대
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return (
            out[0].detach().cpu().numpy(),
            out[1].detach().cpu().numpy(),
        )
    if isinstance(out, dict):
        be = out.get("box_encodings")
        cs = out.get("class_scores")
        if be is not None and cs is not None:
            return (be.detach().cpu().numpy(), cs.detach().cpu().numpy())
    raise RuntimeError(
        f"Unexpected TorchScript output for {model_name}: {type(out)}"
    )


# =============================================================================
# TensorRT (fp32/fp16/int8)
# =============================================================================
def _make_trt_infer_fn(model_name, model_dir, precision):
    import ctypes
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401 - 컨텍스트 초기화 부작용 필요
    import pycuda.driver as cuda

    engine_path = _resolve_trt_engine_path(model_name, model_dir, precision)
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"TRT engine not found: {engine_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    # EfficientNMS 등 plugin layer 포함 엔진 역직렬화를 위해
    # plugin registry를 선초기화한다.
    try:
        ctypes.CDLL("libnvinfer_plugin.so")
    except Exception:
        pass
    try:
        trt.init_libnvinfer_plugins(logger, "")
    except Exception:
        pass

    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"TensorRT engine deserialize failed: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"TensorRT context creation failed: {engine_path}")

    # I/O 바인딩 준비
    bindings = []
    host_buffers = {}
    device_buffers = {}
    input_name = None
    output_names = []

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        shape = tuple(engine.get_binding_shape(i))
        # 동적 shape 가 있으면 batch=1 가정
        shape = tuple(max(1, d) for d in shape)
        size = int(np.prod(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        dev_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(dev_mem))
        host_buffers[name] = (host_mem, shape, dtype)
        device_buffers[name] = dev_mem
        if engine.binding_is_input(i):
            input_name = name
        else:
            output_names.append(name)

    stream = cuda.Stream()

    def infer(x_np):
        host_in, shape_in, dtype_in = host_buffers[input_name]
        # NHWC engine + NCHW input → transpose (matches ORT _prepare_onnx_input)
        if (len(shape_in) == 4 and shape_in[-1] == 3
                and x_np.ndim == 4 and x_np.shape[1] == 3):
            x_np = np.transpose(x_np, (0, 2, 3, 1))
        np.copyto(host_in, x_np.astype(dtype_in).ravel())
        cuda.memcpy_htod_async(device_buffers[input_name], host_in, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        outputs = {}
        for name in output_names:
            host_out, shape_out, dtype_out = host_buffers[name]
            cuda.memcpy_dtoh_async(host_out, device_buffers[name], stream)
            stream.synchronize()
            outputs[name] = np.array(host_out, dtype=dtype_out).reshape(shape_out)
        return _trt_out_to_raw(outputs, model_name)

    return infer


def _trt_out_to_raw(outputs, model_name):
    """TRT binding 이름 기반 출력 정규화."""
    if model_name == "yolov8n":
        # ultralytics ONNX export 의 기본 출력 이름
        for k in ("output0", "output", "out0"):
            if k in outputs:
                return outputs[k]
        return next(iter(outputs.values()))
    # ssd_mv2
    def _pick(*keys):
        for key in keys:
            if key in outputs and outputs[key] is not None:
                return outputs[key]
        return None

    det_boxes = _pick("nmsed_boxes", "detection_boxes")
    det_scores = _pick("nmsed_scores", "detection_scores")
    det_classes = _pick("nmsed_classes", "detection_classes")
    num_det = outputs.get("num_detections")
    if det_boxes is not None and det_scores is not None:
        return {
            "detection_boxes": det_boxes,
            "detection_scores": det_scores,
            "detection_classes": det_classes,
            "num_detections": num_det,
            "detection_multiclass_scores": None,
        }

    be = outputs.get("box_encodings")
    cs = outputs.get("class_scores")
    if be is None or cs is None:
        # fallback: 첫 두 개
        vals = list(outputs.values())
        be, cs = vals[0], vals[1]
    return (be, cs)


# =============================================================================
# ONNX Runtime (CUDA EP, TensorRT EP)
# =============================================================================
def _make_onnxrt_infer_fn(model_name, model_dir, provider):
    import onnxruntime as ort

    if "ssd_mobilenet_v2" in model_name:
        if provider == "CUDAExecutionProvider":
            # FIX: use raw ONNX (no built-in NMS) for fair comparison with other runtimes
            onnx_candidates = [
                os.path.join(model_dir, "ssd_mobilenet_v2_raw3_fpinput_ir8.onnx"),
                os.path.join(model_dir, "ssd_mobilenet_v2_raw_ir8.onnx"),
                os.path.join(model_dir, "ssd_mobilenet_v2_raw.onnx"),
            ]
        else:
            # onnxrt_trt: raw3_fpinput_ir8 preferred (box_encodings (1,1917,4) + class_scores (1,1917,91) verified)
            # raw_ir8 fallback caused class_scores shape collapse to (1,1,1917) under TRT EP
            onnx_candidates = [
                os.path.join(model_dir, "ssd_mobilenet_v2_raw3_fpinput_ir8.onnx"),
                os.path.join(model_dir, "ssd_mobilenet_v2_raw_ir8.onnx"),
                os.path.join(model_dir, "ssd_mobilenet_v2_raw.onnx"),
            ]
        onnx_path = onnx_candidates[0]
        for c in onnx_candidates:
            if os.path.exists(c) and os.path.getsize(c) > 0:
                onnx_path = c
                break
    else:
        onnx_path = _resolve_onnx_path(model_name, model_dir)

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    sess_options = ort.SessionOptions()
    available = ort.get_available_providers()
    if provider == "TensorrtExecutionProvider" and provider in available:
        # TRT 미지원 subgraph가 있을 수 있어 CUDA/CPU fallback 체인 포함
        providers = [
            p for p in
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            if p in available
        ]
    else:
        providers = [provider] if provider in available else ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    except Exception as e:
        msg = str(e)
        if "Unsupported model IR version" in msg:
            try:
                import onnx
                m = onnx.load(onnx_path)
                if getattr(m, "ir_version", 0) > 8:
                    compat_path = os.path.splitext(onnx_path)[0] + "_ir8.onnx"
                    m.ir_version = 8
                    onnx.save(m, compat_path)
                    print(f"[INFO] ORT compat fallback: downgraded IR -> {compat_path}")
                    session = ort.InferenceSession(
                        compat_path, sess_options, providers=providers
                    )
                else:
                    raise
            except Exception:
                raise e
        else:
            raise

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = list(input_meta.shape) if input_meta.shape is not None else []
    input_type = str(getattr(input_meta, "type", ""))
    output_names = [o.name for o in session.get_outputs()]

    def _prepare_onnx_input(x_np):
        x = x_np
        # SSD TF 계열 ONNX는 NHWC 입력인 경우가 있다.
        if len(input_shape) == 4 and input_shape[-1] == 3 and x.ndim == 4 and x.shape[1] == 3:
            x = np.transpose(x, (0, 2, 3, 1))
        # 모델 입력 dtype에 맞춰 캐스팅
        if "uint8" in input_type:
            if x.dtype != np.uint8:
                if np.issubdtype(x.dtype, np.floating):
                    # preprocess_image now returns [0,255] for SSD; clip+cast.
                    x = np.clip(x, 0, 255).astype(np.uint8)
                else:
                    x = x.astype(np.uint8)
        else:
            if x.dtype != np.float32:
                x = x.astype(np.float32)
        return np.ascontiguousarray(x)

    def infer(x_np):
        x_in = _prepare_onnx_input(x_np)
        outs = session.run(output_names, {input_name: x_in})
        return _ort_out_to_raw(outs, output_names, model_name)

    return infer


def _ort_out_to_raw(outs, output_names, model_name):
    if model_name == "yolov8n":
        return outs[0]
    # ssd_mv2: 이름으로 매핑
    name_to_out = dict(zip(output_names, outs))
    # TF2 OD API 직변환 출력 (이미 detection_* 텐서 포함)
    if "detection_boxes" in name_to_out and "detection_scores" in name_to_out:
        return {
            "detection_boxes": name_to_out.get("detection_boxes"),
            "detection_scores": name_to_out.get("detection_scores"),
            "detection_classes": name_to_out.get("detection_classes"),
            "num_detections": name_to_out.get("num_detections"),
            "detection_multiclass_scores": name_to_out.get("detection_multiclass_scores"),
        }
    be = name_to_out.get("box_encodings")
    cs = name_to_out.get("class_scores")
    if be is None or cs is None:
        be, cs = outs[0], outs[1]
    return (be, cs)


# =============================================================================
# TFLite (CPU, GPU delegate)
# =============================================================================
def _make_tflite_infer_fn(model_name, model_dir, use_gpu):
    tflite_path = _resolve_tflite_path(model_name, model_dir)
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite not found: {tflite_path}")

    backend_order = ["tflite_runtime", "tensorflow"]
    if "ssd_mobilenet_v2" in model_name:
        # SSD TFLite는 Flex/Select TF Ops가 필요한 경우가 잦아
        # TensorFlow 패키지의 인터프리터를 우선 사용한다.
        backend_order = ["tensorflow", "tflite_runtime"]

    interpreter = None
    selected_backend = ""
    last_error = None
    for backend in backend_order:
        try:
            if backend == "tflite_runtime":
                from tflite_runtime.interpreter import Interpreter as _Interpreter
                try:
                    from tflite_runtime.interpreter import load_delegate as _load_delegate
                except Exception:
                    _load_delegate = None
            else:
                import tensorflow as tf
                _Interpreter = tf.lite.Interpreter
                _load_delegate = getattr(tf.lite.experimental, "load_delegate", None)

            experimental_delegates = []
            if use_gpu and _load_delegate is not None:
                try:
                    # Jetson Linux GPU delegate
                    experimental_delegates.append(_load_delegate("libdelegate_gpu.so"))
                except Exception as e:
                    print(f"[WARN] tflite gpu delegate load failed ({backend}): {e}")

            _itp = _Interpreter(
                model_path=tflite_path,
                experimental_delegates=experimental_delegates or None,
            )
            _itp.allocate_tensors()
            interpreter = _itp
            selected_backend = backend
            break
        except Exception as e:
            last_error = e
            msg = str(e)
            if backend == "tflite_runtime" and ("Flex" in msg or "Select TF" in msg):
                print("[INFO] tflite_runtime lacks Flex op support for this model; trying tensorflow interpreter.")
            else:
                print(f"[WARN] tflite interpreter init failed ({backend}): {e}")

    if interpreter is None:
        raise RuntimeError(f"failed to initialize tflite interpreter: {last_error}")
    print(f"[INFO] tflite backend selected: {selected_backend}")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]

    def infer(x_np):
        nonlocal input_details, output_details, input_index
        # TFLite 는 보통 NHWC → transpose 필요
        inp_shape = input_details[0]["shape"]
        if len(inp_shape) == 4 and inp_shape[-1] in (1, 3):
            nhwc = np.transpose(x_np, (0, 2, 3, 1))
        else:
            nhwc = x_np
        inp_dtype = input_details[0]["dtype"]
        if inp_dtype == np.uint8 and nhwc.dtype != np.uint8:
            # preprocess_image returns [0,255] float for SSD; just clip+cast.
            nhwc = np.clip(nhwc, 0, 255).astype(np.uint8)
        else:
            nhwc = nhwc.astype(inp_dtype)
        # 동적 입력 shape 모델(예: [1, -1, -1, 3]) 지원
        if tuple(input_details[0]["shape"]) != tuple(nhwc.shape):
            interpreter.resize_tensor_input(input_index, nhwc.shape, strict=False)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_index = input_details[0]["index"]
        interpreter.set_tensor(
            input_index, nhwc
        )
        interpreter.invoke()
        outs = [interpreter.get_tensor(d["index"]) for d in output_details]
        return _tflite_out_to_raw(outs, output_details, model_name)

    return infer


def _tflite_out_to_raw(outs, output_details, model_name):
    if model_name == "yolov8n":
        out = outs[0]
        # TFLite 는 종종 [1, 8400, 84] 로 나옴 → [1, 84, 8400] 으로 재정렬
        if out.ndim == 3 and out.shape[1] > out.shape[2]:
            out = np.transpose(out, (0, 2, 1))
        return out
    # ssd_mv2
    name_to_out = {d["name"]: o for d, o in zip(output_details, outs)}
    if "detection_boxes" in name_to_out and "detection_scores" in name_to_out:
        return {
            "detection_boxes": name_to_out.get("detection_boxes"),
            "detection_scores": name_to_out.get("detection_scores"),
            "detection_classes": name_to_out.get("detection_classes"),
            "num_detections": name_to_out.get("num_detections"),
            "detection_multiclass_scores": name_to_out.get("detection_multiclass_scores"),
        }
    # 이름이 generic(StatefulPartitionedCall:*)인 경우 shape로 추론
    det_boxes = None
    det_scores = None
    det_classes = None
    num_det = None
    raw_boxes = None
    raw_scores = None
    for o in outs:
        arr = np.asarray(o)
        if arr.ndim == 1 and arr.size == 1:
            num_det = arr
        elif arr.ndim == 2:
            v = arr.reshape(-1)
            # 정수값 위주면 class, 아니면 score로 간주
            if np.all(np.isfinite(v)) and np.allclose(v, np.round(v), atol=1e-3):
                # detection_anchor_indices(0~1916)를 classes로 오인하지 않도록 범위 필터
                vmin = float(np.min(v)) if v.size else 0.0
                vmax = float(np.max(v)) if v.size else 0.0
                if 0.0 <= vmin and vmax <= 100.0:
                    det_classes = arr
            else:
                det_scores = arr
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            # [1, N, 4]: N이 큰 경우 raw boxes 가능성이 높음
            if arr.shape[1] > 200:
                raw_boxes = arr
            else:
                det_boxes = arr
        elif arr.ndim == 3 and arr.shape[-1] > 10:
            raw_scores = arr
    if det_boxes is not None and det_scores is not None:
        return {
            "detection_boxes": det_boxes,
            "detection_scores": det_scores,
            "detection_classes": det_classes,
            "num_detections": num_det,
            "detection_multiclass_scores": None,
        }
    be = name_to_out.get("box_encodings")
    cs = name_to_out.get("class_scores")
    if be is None or cs is None:
        if raw_boxes is not None and raw_scores is not None:
            be, cs = raw_boxes, raw_scores
        else:
            be, cs = outs[0], outs[1]
    return (be, cs)


# =============================================================================
# ncnn (CPU, Vulkan, Vulkan fixed)
# =============================================================================
def _make_ncnn_infer_fn(model_name, model_dir, use_vulkan, fixed):
    import json
    import ncnn

    # 변환 스크립트가 남긴 상태 파일이 명시적으로 실패면 런타임 진입 전에 차단
    status_candidates = [
        os.path.join(model_dir, f"{model_name}.ncnn.status.json"),
    ]
    if "ssd_mobilenet_v2" in model_name:
        status_candidates.extend([
            os.path.join(model_dir, "ssd_mobilenet_v2_raw.ncnn.status.json"),
            os.path.join(model_dir, "ssd_mobilenet_v2.ncnn.status.json"),
        ])
    for st in status_candidates:
        if not os.path.exists(st):
            continue
        try:
            with open(st, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            # 상태 파일 파싱 실패는 하드 에러로 보지 않고 런타임 시도
            continue
        if isinstance(meta, dict) and meta.get("ok") is False:
            reason = meta.get("reason", "unknown")
            phase = meta.get("phase", "convert")
            raise RuntimeError(
                f"NCNN status not ok ({phase}): {reason} [{os.path.basename(st)}]"
            )

    param_path, bin_path = _resolve_ncnn_path(model_name, model_dir)
    if not (os.path.exists(param_path) and os.path.exists(bin_path)):
        raise FileNotFoundError(
            f"ncnn model files not found: {param_path}, {bin_path}"
        )

    if fixed:
        # vulkan_fixed: create_gpu_instance() 누락 버그 우회
        ncnn.create_gpu_instance()

    net = ncnn.Net()
    net.opt.use_vulkan_compute = use_vulkan
    if use_vulkan:
        net.opt.use_fp16_storage = True
        net.opt.use_fp16_arithmetic = True
    net.load_param(param_path)
    net.load_model(bin_path)

    # 입출력 이름 자동 감지
    try:
        in_names = list(net.input_names())
        out_names = list(net.output_names())
    except Exception:
        in_names, out_names = ["in0"], ["out0"]
    input_name = in_names[0] if in_names else "in0"
    # YOLOv8 은 out0 하나, SSD-MV2 는 두 개 (box_encodings, class_scores)

    h, w = get_input_hw(model_name)

    def infer(x_np):
        # NCHW float32 → ncnn.Mat (W, H, C)
        chw = x_np[0]  # [C, H, W]
        # SSD trimmed ncnn graph는 전처리(Mul/Sub)를 stripping 했으므로
        # Python에서 [-1,1] 정규화를 대신 수행한다.
        if model_name == "ssd_mobilenet_v2":
            chw = chw.astype(np.float32) / 127.5 - 1.0  # [-1,1] normalize for SSD ncnn (graph has no preprocess ops)
        mat_in = _numpy_to_ncnn_mat(chw)

        ex = net.create_extractor()
        ex.input(input_name, mat_in)

        if model_name == "yolov8n":
            _, out = ex.extract(out_names[0] if out_names else "out0")
            return _ncnn_mat_to_numpy(out, expected_ndim=3)
        # ssd_mv2: ncnn Postprocessor/Decode broken (20x20 vs 19x19 anchor mismatch).
        # Fix: extract raw box from pre-decode concat layer, use Python anchor decode.
        RAW_BOX_LAYER = "StatefulPartitionedCall/concat:0"
        import numpy as _np
        try:
            _, raw_box_mat = ex.extract(RAW_BOX_LAYER)
            raw_box = _ncnn_mat_to_numpy(raw_box_mat, expected_ndim=3)
            if raw_box.ndim == 3 and raw_box.shape[1] == 1:
                be_out = raw_box[:, 0, :][_np.newaxis, ...]
            else:
                be_out = raw_box.reshape(1, -1, 4)
        except Exception:
            be_out = None
            for name in out_names:
                if "box" in name.lower():
                    _, arr = ex.extract(name)
                    be_out = _ncnn_mat_to_numpy(arr, expected_ndim=3)
                    break
        cs_out = None
        for name in out_names:
            if "class" in name.lower() or "score" in name.lower():
                _, arr = ex.extract(name)
                cs_out = _ncnn_mat_to_numpy(arr, expected_ndim=3)
                break
        if be_out is not None and cs_out is not None:
            n_box = be_out.shape[1]
            if cs_out.shape[1] != n_box:
                cs_out = cs_out[:, :n_box, :]
        return (be_out, cs_out)

    return infer


def _numpy_to_ncnn_mat(chw):
    """numpy [C, H, W] float32 → ncnn.Mat (W, H, C)."""
    import ncnn
    c, h, w = chw.shape
    mat = ncnn.Mat(w, h, c)
    # ncnn.Mat 은 channel-major 저장 → 채널별 plane 복사
    mat_np = np.array(mat, copy=False)  # [C, H, W]
    if mat_np.shape == chw.shape:
        mat_np[...] = chw
    else:
        # fallback: pickle-free 복사
        for ci in range(c):
            ch_mat = mat.channel(ci)
            ch_np = np.array(ch_mat, copy=False)
            ch_np[...] = chw[ci]
    return mat


def _ncnn_mat_to_numpy(mat, expected_ndim=3):
    """ncnn.Mat → np.ndarray. batch 축 추가 포함."""
    arr = np.array(mat)
    if arr.ndim == expected_ndim - 1:
        arr = arr[None, ...]
    return arr.astype(np.float32)


# =============================================================================
# 경로 해석 헬퍼
# =============================================================================
def _resolve_onnx_path(model_name, model_dir):
    """YOLOv8n 은 <name>.onnx, SSD-MV2 는 <name>_raw.onnx."""
    if "ssd_mobilenet_v2" in model_name:
        return os.path.join(model_dir, "ssd_mobilenet_v2_raw.onnx")
    return os.path.join(model_dir, f"{model_name}.onnx")


def _resolve_trt_engine_path(model_name, model_dir, precision):
    """모델별 TRT 엔진 경로를 우선순위로 탐색한다."""
    candidates = []
    if "ssd_mobilenet_v2" in model_name:
        candidates.extend([
            os.path.join(model_dir, f"ssd_mobilenet_v2_{precision}.engine"),
            os.path.join(model_dir, f"ssd_mobilenet_v2_raw_{precision}.engine"),
            os.path.join(model_dir, f"ssd_mobilenet_v2_raw_fpinput_{precision}.engine"),
            os.path.join(model_dir, f"ssd_mobilenet_v2_raw3_fpinput_{precision}.engine"),
            os.path.join(model_dir, f"ssd_mobilenet_v2_effnms_fpinput_{precision}.engine"),
        ])
    else:
        candidates.append(os.path.join(model_dir, f"{model_name}_{precision}.engine"))

    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return candidates[0]


def _resolve_tflite_path(model_name, model_dir):
    """모델별 TFLite 경로를 우선순위로 탐색한다."""
    candidates = []
    if "ssd_mobilenet_v2" in model_name:
        candidates.extend([
            os.path.join(model_dir, "ssd_mobilenet_v2.tflite"),
            os.path.join(model_dir, "ssd_mobilenet_v2_raw.tflite"),
            os.path.join(model_dir, "ssd_mobilenet_v2_float32.tflite"),
            os.path.join(model_dir, "ssd_mobilenet_v2_raw_float32.tflite"),
        ])
    elif "yolov8n" in model_name:
        # ultralytics 버전별 출력명 호환
        candidates.extend([
            os.path.join(model_dir, "yolov8n.tflite"),
            os.path.join(model_dir, "yolov8n_float32.tflite"),
            os.path.join(model_dir, "yolov8n_saved_model", "yolov8n_float32.tflite"),
            os.path.join(model_dir, "yolov8n_saved_model", "yolov8n.tflite"),
        ])
    else:
        candidates.append(os.path.join(model_dir, f"{model_name}.tflite"))

    # fallback: model_name prefix 로 하위 경로까지 탐색
    dynamic_globs = [
        os.path.join(model_dir, f"{model_name}*.tflite"),
        os.path.join(model_dir, f"{model_name}_saved_model", "*.tflite"),
    ]
    for pat in dynamic_globs:
        for path in sorted(glob.glob(pat)):
            if path not in candidates:
                candidates.append(path)

    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return candidates[0]


def _resolve_ncnn_path(model_name, model_dir):
    """ultralytics export 는 <name>_ncnn_model/model.{param,bin}.

    convert_ncnn.sh 로 생성한 경우 <name>.param / <name>.bin.
    """
    def _find_pair_in_dir(dir_path):
        if not os.path.isdir(dir_path):
            return None
        # ultralytics 기본 출력
        model_param = os.path.join(dir_path, "model.param")
        model_bin = os.path.join(dir_path, "model.bin")
        if (
            os.path.exists(model_param)
            and os.path.exists(model_bin)
            and os.path.getsize(model_param) > 0
            and os.path.getsize(model_bin) > 0
        ):
            return model_param, model_bin
        # pnnx 출력: <name>.ncnn.param/.bin
        for param_path in sorted(glob.glob(os.path.join(dir_path, "*.param"))):
            bin_path = param_path[:-6] + ".bin"
            if (
                os.path.exists(bin_path)
                and os.path.getsize(param_path) > 0
                and os.path.getsize(bin_path) > 0
            ):
                return param_path, bin_path
        return None

    subdir_pair = _find_pair_in_dir(os.path.join(model_dir, f"{model_name}_ncnn_model"))
    if subdir_pair:
        return subdir_pair

    flat_candidates = []
    if "ssd_mobilenet_v2" in model_name:
        flat_candidates.extend([
            (
                os.path.join(model_dir, "ssd_mobilenet_v2.param"),
                os.path.join(model_dir, "ssd_mobilenet_v2.bin"),
            ),
            (
                os.path.join(model_dir, "ssd_mobilenet_v2_raw.param"),
                os.path.join(model_dir, "ssd_mobilenet_v2_raw.bin"),
            ),
            (
                os.path.join(model_dir, "ssd_mobilenet_v2.ncnn.param"),
                os.path.join(model_dir, "ssd_mobilenet_v2.ncnn.bin"),
            ),
            (
                os.path.join(model_dir, "ssd_mobilenet_v2_raw.ncnn.param"),
                os.path.join(model_dir, "ssd_mobilenet_v2_raw.ncnn.bin"),
            ),
        ])
    else:
        flat_candidates.extend([
            (
                os.path.join(model_dir, f"{model_name}.param"),
                os.path.join(model_dir, f"{model_name}.bin"),
            ),
            (
                os.path.join(model_dir, f"{model_name}.ncnn.param"),
                os.path.join(model_dir, f"{model_name}.ncnn.bin"),
            ),
        ])

    for param_path, bin_path in flat_candidates:
        if (
            os.path.exists(param_path)
            and os.path.exists(bin_path)
            and os.path.getsize(param_path) > 0
            and os.path.getsize(bin_path) > 0
        ):
            return param_path, bin_path

    return flat_candidates[0]
