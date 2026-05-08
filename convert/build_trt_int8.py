#!/usr/bin/env python3
# ============================================================================
# build_trt_int8.py - TRT INT8 정밀 캘리브레이션 빌더 (파이썬 기반)
# ============================================================================
# 역할: 단순 혼합 정밀도(trtexec --int8)가 아닌, 실제 ImageNet 검증 데이터셋을
#       통과하며 레이어별 활성화 값 분포를 정밀하게 추정(EntropyCalibrator)하여
#       정확도 손실을 최소화한 완벽한 INT8 엔진을 생성.
# ============================================================================

import os
import argparse
import sys
import numpy as np

# 상위 디렉터리 경로 추가 (accuracy_eval 모듈 참조 위함)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from benchmark.accuracy_eval import preprocess_imagenet
except ImportError:
    print("[WARN] Could not import preprocess_imagenet. Make sure path is correct.")
    sys.exit(1)

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class JpegBatchStream:
    def __init__(self, data_dir, max_images=100, batch_size=4, input_size=224):
        self.batch_size = batch_size
        self.input_size = input_size
        self.idx = 0
        
        self.img_paths = set()
        # val 디렉토리 전부 탐색하여 무작위 max_images 개 추출
        import glob
        all_imgs = glob.glob(os.path.join(data_dir, "**", "*.JPEG"), recursive=True)
        if not all_imgs:
            all_imgs = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
        
        np.random.seed(42)
        np.random.shuffle(all_imgs)
        self.img_paths = all_imgs[:max_images]
        print(f"[Calib] Found {len(self.img_paths)} calibration images.")

    def peek_shape(self):
        return (self.batch_size, 3, self.input_size, self.input_size)

    def next_batch(self):
        if self.idx >= len(self.img_paths):
            return None
        batch = []
        for _ in range(self.batch_size):
            if self.idx >= len(self.img_paths):
                break
            try:
                x = preprocess_imagenet(self.img_paths[self.idx], self.input_size)
                batch.append(x[0])
            except Exception:
                pass
            self.idx += 1
            
        if not batch:
            return None
        # 부족한 배치는 0으로 패딩 (TRT 요구사항에 따라 고정 배치 크기 전송 필요할 수 있음)
        out = np.zeros((self.batch_size, 3, self.input_size, self.input_size), dtype=np.float32)
        stack = np.stack(batch, axis=0)
        out[:stack.shape[0]] = stack
        return out


class ImageBatchEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_stream, cache_file="int8_calib.cache"):
        super().__init__()
        self.batch_stream = batch_stream
        self.cache_file = cache_file
        
        sample_shape = self.batch_stream.peek_shape()
        self.batch_size = sample_shape[0]
        self.device_input = cuda.mem_alloc(trt.volume(sample_shape) * np.float32().nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        batch = self.batch_stream.next_batch()
        if batch is None:
            return None
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_int8_engine(onnx_path, engine_path, data_dir, max_images, batch_size):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("[ERROR] Failed to parse ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30 # 1GB
    
    if not builder.platform_has_fast_int8:
        print("[WARN] Platform does not have fast INT8. Generating anyway, but performance may be suboptimal.")

    config.set_flag(trt.BuilderFlag.INT8)
    
    cache_file = engine_path.replace(".engine", "_calib.cache")
    stream = JpegBatchStream(data_dir, max_images, batch_size)
    calibrator = ImageBatchEntropyCalibrator(stream, cache_file)
    config.int8_calibrator = calibrator

    print(f"[INFO] Building INT8 Engine (this may take a while)...")
    engine = builder.build_engine(network, config)
    if engine is None:
        print("[ERROR] Failed to build INT8 engine.")
        return False

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
        
    print(f"[INFO] Successfully saved INT8 engine to {engine_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to input ONNX")
    parser.add_argument("--engine", required=True, help="Path to output INT8 Engine")
    parser.add_argument("--data-dir", required=True, help="Path to ImageNet validation dir for calibration")
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    build_int8_engine(args.onnx, args.engine, args.data_dir, args.max_images, batch_size=8)
