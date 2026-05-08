#!/usr/bin/env python3
"""COCO-based TRT INT8 calibrator for detection models (YOLOv8n, SSD MobileNetV2).

Key difference vs build_trt_int8.py:
  - Uses detection_infer.preprocess_image (COCO flow, /255.0, RGB),
    not preprocess_imagenet (mean/std normalize, ImageNet-only).
  - Matches inference-time preprocessing exactly so calibration
    distribution == inference distribution.
"""
import os
import sys
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.detection_infer import preprocess_image, get_input_hw

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import cv2


class CocoBatchStream:
    def __init__(self, model_name, data_dir, max_images=200, batch_size=4):
        self.model_name = model_name
        self.batch_size = batch_size
        self.h, self.w = get_input_hw(model_name)
        self.idx = 0

        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPEG")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
            paths.extend(glob.glob(os.path.join(data_dir, ext)))
        paths = sorted(set(paths))
        np.random.seed(42)
        np.random.shuffle(paths)
        self.img_paths = paths[:max_images]
        print(f"[Calib] {model_name}: found {len(self.img_paths)} COCO images "
              f"in {data_dir}, input {self.h}x{self.w}")

    def peek_shape(self):
        # preprocess_image returns NCHW regardless of model; engine raveled bytes
        # match as long as calibration uses same preprocess as inference.
        return (self.batch_size, 3, self.h, self.w)

    def next_batch(self):
        if self.idx >= len(self.img_paths):
            return None
        batch = []
        for _ in range(self.batch_size):
            if self.idx >= len(self.img_paths):
                break
            p = self.img_paths[self.idx]
            self.idx += 1
            try:
                img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                x = preprocess_image(img_bgr, self.model_name)  # (1, 3, H, W)
                batch.append(x[0])
            except Exception as e:
                print(f"[Calib] skip {p}: {e}")
        if not batch:
            return None
        out = np.zeros((self.batch_size, 3, self.h, self.w), dtype=np.float32)
        stack = np.stack(batch, axis=0)
        out[: stack.shape[0]] = stack
        return out


class CocoEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_stream, cache_file):
        super().__init__()
        self.stream = batch_stream
        self.cache_file = cache_file
        shape = batch_stream.peek_shape()
        self.batch_size = shape[0]
        self.device_input = cuda.mem_alloc(int(np.prod(shape)) * np.float32().nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        batch = self.stream.next_batch()
        if batch is None:
            return None
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        # TRT 8.0 pybind11 crashes on None return; use empty bytes when no cache
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return b""

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(bytes(cache))


def build_int8_engine(onnx_path, engine_path, model_name, data_dir,
                       max_images=200, batch_size=4, shape_override=None):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[ERROR] ONNX parse failed.")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GB

    if not builder.platform_has_fast_int8:
        print("[WARN] platform lacks fast INT8")
    config.set_flag(trt.BuilderFlag.INT8)

    # Dynamic-shape SSD: fix input profile; calibrator batch must match dims[0]
    if shape_override is not None:
        input_name, dims = shape_override
        # Align calibrator batch with profile's batch dim to avoid mismatch
        if dims[0] != batch_size:
            print(f"[INFO] shape batch {dims[0]} != --batch-size {batch_size}; "
                  f"overriding calibrator batch to {dims[0]}")
            batch_size = dims[0]
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, dims, dims, dims)
        config.add_optimization_profile(profile)
        config.set_calibration_profile(profile)
        print(f"[INFO] set shape override: {input_name} = {dims}")

    cache_file = engine_path.replace(".engine", "_calib.cache")
    # For shape-overridden builds (e.g. SSD batch=1), CocoBatchStream must use
    # the adjusted batch_size set above.
    stream = CocoBatchStream(model_name, data_dir, max_images, batch_size)
    calibrator = CocoEntropyCalibrator(stream, cache_file)
    config.int8_calibrator = calibrator

    print(f"[INFO] building INT8 engine ({onnx_path} -> {engine_path})")
    engine = builder.build_engine(network, config)
    if engine is None:
        print("[ERROR] build_engine returned None")
        return False

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"[OK] saved {engine_path} ({size_mb:.1f} MB)")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model-name", required=True,
                    choices=["yolov8n", "ssd_mobilenet_v2"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--shape-name", default=None,
                    help="input tensor name for NHWC shape override")
    ap.add_argument("--shape-dims", default=None,
                    help="comma-separated dims e.g. 1,320,320,3")
    args = ap.parse_args()

    shape_override = None
    if args.shape_name and args.shape_dims:
        dims = tuple(int(x) for x in args.shape_dims.split(","))
        shape_override = (args.shape_name, dims)

    ok = build_int8_engine(
        args.onnx, args.engine, args.model_name, args.data_dir,
        args.max_images, args.batch_size, shape_override,
    )
    sys.exit(0 if ok else 1)
