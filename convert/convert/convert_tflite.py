#!/usr/bin/env python3
# ============================================================================
# convert_tflite.py - ONNX 모델을 TFLite 형식으로 변환하는 스크립트
# ============================================================================
# 역할: ONNX 모델을 TensorFlow Lite로 2단계 변환
#   1단계: ONNX → TF SavedModel (onnx-tf 또는 onnx2tf 사용)
#   2단계: TF SavedModel → TFLite (float32 + float16 두 가지 버전)
#
# 왜 직접 변환이 안 되는가?
#   ONNX → TFLite 직접 변환 도구가 없으므로,
#   TF SavedModel을 중간 형식으로 거쳐야 함
#
# 사용법:
#   python convert_tflite.py --onnx-path ./models/resnet50.onnx --output-dir ./models
#
# 요구사항:
#   - tensorflow 패키지
#   - onnx-tf 또는 onnx2tf 중 하나 (ONNX→TF 변환용)
#
# 파이프라인 흐름:
#   export_onnx.py → [이 스크립트] → benchmark/run_tflite.py
# ============================================================================
"""Convert ONNX model to TFLite format."""

import argparse
import glob
import os
import subprocess
import sys
import shutil


def _infer_onnx_input_shape(onnx_path):
    """ONNX 첫 입력 텐서의 이름/shape를 읽어 onnx2tf 정적 shape 힌트로 사용."""
    try:
        import onnx
        m = onnx.load(onnx_path)
        if not m.graph.input:
            return None, None
        inp = m.graph.input[0]
        name = inp.name
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            # dynamic/symbolic 차원은 최소 1로 고정
            if getattr(d, "dim_value", 0):
                dims.append(int(d.dim_value))
            else:
                dims.append(1)
        if not dims:
            return name, None
        return name, dims
    except Exception:
        return None, None


def _ensure_user_bin_in_path():
    """~/.local/bin 이 PATH에 없으면 추가한다 (onnx2tf/onnxsim CLI 탐지용)."""
    user_bin = os.path.expanduser("~/.local/bin")
    cur = os.environ.get("PATH", "")
    parts = cur.split(os.pathsep) if cur else []
    if user_bin not in parts:
        os.environ["PATH"] = user_bin + os.pathsep + cur


def _saved_model_exists(saved_model_dir):
    """SavedModel 디렉토리 유효성 확인."""
    return (
        os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb"))
        or os.path.isfile(os.path.join(saved_model_dir, "saved_model.pbtxt"))
    )


def _find_direct_tflite_artifacts(saved_model_dir):
    """onnx2tf direct output(.tflite) 탐색."""
    if not os.path.isdir(saved_model_dir):
        return None, None
    files = sorted(glob.glob(os.path.join(saved_model_dir, "*.tflite")))
    if not files:
        return None, None

    fp32 = None
    fp16 = None
    for p in files:
        n = os.path.basename(p).lower()
        if "float16" in n or "fp16" in n:
            fp16 = p
        elif "float32" in n or "fp32" in n:
            fp32 = p
    if fp32 is None:
        # 이름 규칙이 없는 경우 첫 파일을 float32 대체로 사용
        fp32 = files[0]
    return fp32, fp16


def _maybe_patch_yolo_mul_broadcast(onnx_path):
    """YOLOv8 ONNX의 Mul broadcast 이슈를 우회하기 위한 최소 패치.

    /model.22/Mul_2 에 연결된 [1, 8400] 상수를 [1, 8400, 1]로 보정한다.
    """
    stem = os.path.splitext(os.path.basename(onnx_path))[0].lower()
    if "yolov8" not in stem:
        return onnx_path

    try:
        import numpy as np
        import onnx
        from onnx import numpy_helper
    except Exception:
        return onnx_path

    try:
        model = onnx.load(onnx_path)
    except Exception:
        return onnx_path

    init_map = {t.name: i for i, t in enumerate(model.graph.initializer)}
    patched = False
    for node in model.graph.node:
        if node.op_type != "Mul":
            continue
        if node.name != "/model.22/Mul_2":
            continue
        if len(node.input) < 2:
            continue
        const_name = node.input[1]
        idx = init_map.get(const_name)
        if idx is None:
            continue
        arr = numpy_helper.to_array(model.graph.initializer[idx])
        if arr.shape == (1, 8400):
            arr = arr.reshape(1, 8400, 1).astype(np.float32)
            model.graph.initializer[idx].CopyFrom(
                numpy_helper.from_array(arr, const_name)
            )
            patched = True
            break

    if not patched:
        return onnx_path

    patched_path = os.path.join(
        os.path.dirname(onnx_path) or ".",
        f"{os.path.splitext(os.path.basename(onnx_path))[0]}_mulfix.onnx",
    )
    try:
        onnx.save(model, patched_path)
        print(f"  [INFO] YOLO ONNX broadcast patch applied: {patched_path}")
        return patched_path
    except Exception as e:
        print(f"  [WARN] YOLO ONNX patch save failed: {e}")
        return onnx_path


def convert_onnx_to_tflite(onnx_path, output_dir=None):
    """ONNX 모델을 TFLite로 변환하는 메인 함수.

    변환 과정: ONNX → TF SavedModel → TFLite (float32 + float16)

    Args:
        onnx_path: 입력 ONNX 모델 파일 경로
        output_dir: 출력 디렉토리 (None이면 ONNX 파일과 같은 디렉토리)
    """
    _ensure_user_bin_in_path()

    # ONNX 파일명에서 모델 이름 추출 (예: resnet50.onnx → resnet50)
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    onnx_input_path = _maybe_patch_yolo_mul_broadcast(onnx_path)
    # 출력 디렉토리 결정: 인자 > ONNX 파일 디렉토리 > 현재 디렉토리
    output_dir = output_dir or os.path.dirname(onnx_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # 각 출력 파일 경로 정의
    saved_model_dir = os.path.join(output_dir, f"{model_name}_saved_model")  # TF SavedModel 디렉토리
    tflite_path = os.path.join(output_dir, f"{model_name}.tflite")           # float32 TFLite 파일
    tflite_fp16_path = os.path.join(output_dir, f"{model_name}_fp16.tflite") # float16 TFLite 파일

    # ---- 1단계: ONNX → TF SavedModel ----
    # TFLite 변환기는 TF SavedModel만 입력으로 받으므로 중간 변환 필수
    print(f"[1/3] Converting ONNX to TF SavedModel...")
    print(f"  Input:  {onnx_input_path}")
    print(f"  Output: {saved_model_dir}")

    try:
        # 방법 1: onnx-tf 라이브러리 사용 (우선 시도)
        # onnx-tf는 ONNX 그래프를 TensorFlow 연산으로 매핑
        from onnx_tf.backend import prepare
        import onnx

        onnx_model = onnx.load(onnx_input_path)
        # prepare(): ONNX 모델을 TF 백엔드로 변환 준비
        tf_rep = prepare(onnx_model)
        # export_graph(): TF SavedModel 형식으로 디스크에 저장
        tf_rep.export_graph(saved_model_dir)
        print("  [OK] Converted via onnx-tf")
    except Exception as e:
        # 방법 2: onnx2tf CLI 도구 사용 (onnx-tf가 없을 때 폴백)
        # onnx2tf는 더 최신 도구로 일부 모델에서 호환성이 더 좋음
        print(f"  onnx-tf path failed ({e}), trying onnx2tf...")
        converted = False
        last_error = None

        # 2-1) Python API 직접 호출 (PATH 문제/구버전 TF 호환 보정)
        try:
            import tensorflow as tf

            # TF 2.4 계열 호환: onnx2tf가 기대하는 API를 최소 보정
            if not hasattr(tf.keras.utils, "set_random_seed"):
                def _compat_set_random_seed(seed):
                    import random
                    import numpy as np
                    random.seed(seed)
                    np.random.seed(seed)
                    tf.random.set_seed(seed)

                tf.keras.utils.set_random_seed = _compat_set_random_seed

            if not hasattr(tf.config.experimental, "enable_op_determinism"):
                tf.config.experimental.enable_op_determinism = lambda: None

            # TF 2.4 계열은 unknown TensorShape 비교 시 __ne__ 가 ValueError 를 던져
            # onnx2tf 입력 노드 처리에서 중단될 수 있다. 비교를 안전하게 보정한다.
            try:
                from tensorflow.python.framework.tensor_shape import TensorShape
                if not hasattr(TensorShape, "_onnx2tf_safe_ne"):
                    TensorShape._onnx2tf_orig_ne = TensorShape.__ne__

                    def _safe_ne(self, other):
                        try:
                            return TensorShape._onnx2tf_orig_ne(self, other)
                        except ValueError:
                            return True

                    TensorShape.__ne__ = _safe_ne
                    TensorShape._onnx2tf_safe_ne = True
            except Exception:
                pass

            from onnx2tf import convert as onnx2tf_convert

            input_name, input_shape = _infer_onnx_input_shape(onnx_input_path)
            kwargs = {
                "input_onnx_file_path": onnx_input_path,
                "output_folder_path": saved_model_dir,
                # Jetson 환경에서 onnxsim 바이너리 의존 실패를 회피
                "not_use_onnxsim": True,
            }
            if input_shape:
                kwargs["batch_size"] = int(input_shape[0]) if len(input_shape) > 0 else 1
                if input_name:
                    shape_text = ",".join(str(x) for x in input_shape)
                    kwargs["overwrite_input_shape"] = [f"{input_name}:{shape_text}"]
            onnx2tf_convert(**kwargs)
            converted = True
            print("  [OK] Converted via onnx2tf (python API)")
        except Exception as e:
            last_error = e
            print(f"  [WARN] onnx2tf python API failed: {e}")

        # 2-2) CLI 호출 폴백 (onnx2tf, python -m onnx2tf)
        if not converted:
            env = os.environ.copy()
            user_bin = os.path.expanduser("~/.local/bin")
            env["PATH"] = user_bin + os.pathsep + env.get("PATH", "")
            cli_candidates = [
                [
                    "onnx2tf",
                    "-i", onnx_input_path,
                    "-o", saved_model_dir,
                    "-osd",
                ],
                [
                    sys.executable,
                    "-m",
                    "onnx2tf",
                    "-i", onnx_input_path,
                    "-o", saved_model_dir,
                    "-osd",
                ],
            ]
            for cmd in cli_candidates:
                try:
                    subprocess.run(cmd, check=True, env=env)
                    converted = True
                    print(f"  [OK] Converted via {' '.join(cmd[:2])}")
                    break
                except (FileNotFoundError, subprocess.CalledProcessError) as e:
                    last_error = e
                    print(f"  [WARN] onnx2tf command failed: {e}")

        if not converted:
            # 두 도구 모두 사용 불가능한 경우 에러 안내
            print("  [ERROR] Neither onnx-tf nor onnx2tf available.")
            if last_error is not None:
                print(f"  Last error: {last_error}")
            print("  Install one of:")
            print("    pip3 install onnx-tf")
            print("    pip3 install onnx2tf")
            sys.exit(1)

    # onnx2tf 가 SavedModel 없이 *.tflite 를 직접 생성하는 경우를 우선 수용한다.
    direct_fp32, direct_fp16 = _find_direct_tflite_artifacts(saved_model_dir)
    if not _saved_model_exists(saved_model_dir):
        if direct_fp32:
            shutil.copyfile(direct_fp32, tflite_path)
            size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
            print(f"  [OK] direct onnx2tf fp32 detected: {tflite_path} ({size_mb:.1f} MB)")
            if direct_fp16:
                shutil.copyfile(direct_fp16, tflite_fp16_path)
                size_mb16 = os.path.getsize(tflite_fp16_path) / (1024 * 1024)
                print(f"  [OK] direct onnx2tf fp16 detected: {tflite_fp16_path} ({size_mb16:.1f} MB)")
            else:
                print("  [WARN] direct fp16 artifact not found; float32 only.")
            print(f"\n[OK] TFLite conversion complete for {model_name}")
            return
        print(
            "  [ERROR] SavedModel not found and no direct .tflite artifacts were generated."
        )
        sys.exit(1)

    # ---- 2단계: TF SavedModel → TFLite (float32, 기본 양자화) ----
    # DEFAULT 최적화: 모델 크기를 줄이면서 float32 정밀도 유지
    print(f"\n[2/3] Converting to TFLite (float32)...")
    try:
        import tensorflow as tf

        # SavedModel에서 TFLite 변환기 생성
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        # DEFAULT 최적화: 양자화 힌트 적용 (가중치 크기 축소 등)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # 변환 실행 - TFLite 플랫버퍼(flatbuffer) 바이너리 반환
        tflite_model = converter.convert()

        # 변환된 모델을 파일로 저장
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"  Saved: {tflite_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  [ERROR] TFLite conversion failed: {e}")
        sys.exit(1)

    # ---- 3단계: TFLite FP16 양자화 버전 ----
    # float16 양자화: 가중치를 16비트로 줄여 모델 크기 약 절반으로 감소
    # GPU 델리게이트(delegate) 사용 시 FP16 하드웨어 가속 가능
    print(f"\n[3/3] Converting to TFLite (float16 quantized)...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # 타겟 타입을 float16으로 지정하여 가중치를 16비트로 양자화
        converter.target_spec.supported_types = [tf.float16]
        tflite_fp16_model = converter.convert()

        with open(tflite_fp16_path, "wb") as f:
            f.write(tflite_fp16_model)

        size_mb = os.path.getsize(tflite_fp16_path) / (1024 * 1024)
        print(f"  Saved: {tflite_fp16_path} ({size_mb:.1f} MB)")
    except Exception as e:
        # FP16 양자화 실패는 치명적이지 않으므로 경고만 출력하고 계속 진행
        print(f"  [WARN] FP16 quantization failed: {e}")
        print(f"  Continuing with float32 version only.")

    print(f"\n[OK] TFLite conversion complete for {model_name}")


def main():
    """CLI 진입점: 명령줄 인자를 파싱하여 ONNX→TFLite 변환을 실행."""
    parser = argparse.ArgumentParser(description="Convert ONNX to TFLite")
    parser.add_argument("--onnx-path", required=True, help="Path to ONNX model")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    # ONNX 파일 존재 여부 확인
    if not os.path.exists(args.onnx_path):
        print(f"[ERROR] ONNX file not found: {args.onnx_path}")
        sys.exit(1)

    convert_onnx_to_tflite(args.onnx_path, args.output_dir)


if __name__ == "__main__":
    main()
