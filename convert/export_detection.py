#!/usr/bin/env python3
# =============================================================================
# export_detection.py — YOLOv8n 4 포맷 동시 export (PC 전용)
# =============================================================================
# 역할: ultralytics.YOLO 를 사용해 YOLOv8n 가중치(.pt)를 4개 런타임 포맷으로 변환.
#       Jetson에서 12런타임 전체 커버리지 확보에 필요한 아티팩트를 한 번에 생성한다.
#
# 출력 아티팩트:
#   1) yolov8n.onnx              — ONNX RT / TFLite / ncnn / TRT 공용 (raw output)
#   2) yolov8n.torchscript       — PyTorch 런타임 (ultralytics 의존성 없이 로드)
#   3) yolov8n.tflite            — TFLite CPU/GPU
#   4) yolov8n_ncnn_model/       — ncnn CPU/Vulkan
#
# 핵심 파라미터:
#   - nms=False: raw 출력 [1,84,8400] 유지 → 공통 Python 후처리 가능
#     (ARM64 Linux TFLite NMS 제약 이슈 #10303 회피)
#   - opset=11: Jetson JetPack 4.6 TRT 8.2 호환
#   - imgsz=640: YOLOv8 기본 입력 크기
#
# 사용법:
#   conda activate jetsonbench
#   python convert/export_detection.py --output-dir models/
# =============================================================================

"""Export YOLOv8n to ONNX/TorchScript/TFLite/ncnn for Jetson benchmarking."""

import argparse
import glob
import os
import shutil
import sys


def export_yolov8n(output_dir="models", weights="yolov8n.pt"):
    """YOLOv8n 가중치를 4개 런타임 포맷으로 내보낸다.

    ultralytics 는 export 결과물을 현재 작업 디렉토리에 저장하므로,
    각 단계 후 shutil.move 로 output_dir 로 이동시킨다.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics 패키지가 필요합니다. "
              "`pip install ultralytics==8.3.0` 로 설치하세요.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 가중치 로드 (자동 다운로드)
    print(f"[1/4] Loading weights: {weights}")
    yolo = YOLO(weights)

    # ------------------------------------------------------------------
    # 1) ONNX — raw output [1,84,8400] (외부 Python NMS 필수)
    # simplify=True: onnx-simplifier 로 그래프 최적화
    # ------------------------------------------------------------------
    print("[2/4] Exporting ONNX (nms=False, opset=11)...")
    yolo.export(format="onnx", nms=False, opset=11, imgsz=640, simplify=True)
    _move_if_exists("yolov8n.onnx", os.path.join(output_dir, "yolov8n.onnx"))

    # ------------------------------------------------------------------
    # 2) TorchScript — Jetson PyTorch 런타임 (ultralytics 미설치 환경 지원)
    # torch.jit.load 로 직접 로드 가능
    # ------------------------------------------------------------------
    print("[3/4] Exporting TorchScript (imgsz=640)...")
    yolo.export(format="torchscript", imgsz=640)
    _move_if_exists("yolov8n.torchscript",
                    os.path.join(output_dir, "yolov8n.torchscript"))

    # ------------------------------------------------------------------
    # 3) TFLite — ARM64 Linux 제약 우회 (nms=False, int8=False)
    # Ultralytics 이슈 #10303: ARM64에서 TFLite NMS export 미지원
    # 해결: nms=False 로 raw 출력만 내보내고 외부 Python NMS 사용
    # ------------------------------------------------------------------
    print("[4/5] Exporting TFLite (nms=False, float32)...")
    try:
        yolo.export(format="tflite", nms=False, int8=False, imgsz=640)
        tflite_src = _find_tflite_artifact("yolov8n")
        if tflite_src:
            _move_if_exists(tflite_src, os.path.join(output_dir, "yolov8n.tflite"))
        else:
            print("[WARN] TFLite 아티팩트를 찾지 못했습니다. 수동 확인 필요.")
    except Exception as e:
        # TFLite 변환 실패가 발생해도 ncnn export 단계는 계속 시도한다.
        print(f"[WARN] TFLite export failed: {e}")

    # ------------------------------------------------------------------
    # 4) ncnn — ultralytics >= 8.0.129 공식 지원
    # 결과: yolov8n_ncnn_model/ 디렉토리 (model.param + model.bin + metadata.yaml)
    # ------------------------------------------------------------------
    print("[5/5] Exporting ncnn (imgsz=640)...")
    ncnn_dst = os.path.join(output_dir, "yolov8n_ncnn_model")
    try:
        yolo.export(format="ncnn", imgsz=640)
        ncnn_src = _find_ncnn_export_dir("yolov8n")
        if ncnn_src and os.path.isdir(ncnn_src):
            if os.path.isdir(ncnn_dst):
                shutil.rmtree(ncnn_dst)
            shutil.move(ncnn_src, ncnn_dst)
            print(f"  moved ncnn dir → {ncnn_dst}")
        else:
            print("[WARN] ncnn export 디렉토리를 찾지 못했습니다.")
    except Exception as e:
        print(f"[WARN] ncnn export failed: {e}")

    # 요약
    print("\n[DONE] YOLOv8n 4-format export complete:")
    for name in ("yolov8n.onnx", "yolov8n.torchscript", "yolov8n.tflite"):
        path = os.path.join(output_dir, name)
        ok = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{ok}] {path}")
    ncnn_ok = "OK" if os.path.isdir(ncnn_dst) else "MISSING"
    print(f"  [{ncnn_ok}] {ncnn_dst}")


def _move_if_exists(src, dst):
    """src 파일이 존재하면 dst 로 이동 (기존 dst 덮어쓰기)."""
    if not os.path.exists(src):
        print(f"[WARN] 이동할 파일이 없습니다: {src}")
        return
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(src, dst)
    size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"  moved: {src} → {dst} ({size_mb:.1f} MB)")


def _find_tflite_artifact(model_stem):
    """Ultralytics 버전별 TFLite 출력명을 폭넓게 탐색한다."""
    candidate_tflite = [
        f"{model_stem}.tflite",
        f"{model_stem}_float32.tflite",
        os.path.join(f"{model_stem}_saved_model", f"{model_stem}_float32.tflite"),
        os.path.join(f"{model_stem}_saved_model", f"{model_stem}.tflite"),
    ]
    for cand in candidate_tflite:
        if os.path.isfile(cand):
            return cand

    # 최후 fallback: 현재 작업 디렉토리 하위에서 <model>*.tflite 탐색
    globbed = sorted(glob.glob(f"**/{model_stem}*.tflite", recursive=True))
    for cand in globbed:
        if os.path.isfile(cand):
            return cand
    return None


def _find_ncnn_export_dir(model_stem):
    """Ultralytics/pnnx 출력 디렉토리(<name>_ncnn_model)를 탐색한다."""
    direct = f"{model_stem}_ncnn_model"
    if _has_ncnn_pair(direct):
        return direct

    # 버전/실행 경로 차이로 상대 경로가 달라질 수 있어 재탐색
    for d in sorted(glob.glob("**/*_ncnn_model", recursive=True)):
        if _has_ncnn_pair(d):
            return d
    return None


def _has_ncnn_pair(dir_path):
    if not os.path.isdir(dir_path):
        return False
    # 최신 Ultralytics 기본
    if os.path.isfile(os.path.join(dir_path, "model.param")) and os.path.isfile(
        os.path.join(dir_path, "model.bin")
    ):
        return True
    # pnnx 계열: <name>.ncnn.param/.bin
    for param in glob.glob(os.path.join(dir_path, "*.param")):
        stem = param[:-6]
        if os.path.isfile(stem + ".bin"):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8n to 4 formats")
    parser.add_argument("--output-dir", default="models",
                        help="Destination directory (default: models)")
    parser.add_argument("--weights", default="yolov8n.pt",
                        help="YOLO weights file or name (default: yolov8n.pt)")
    args = parser.parse_args()

    export_yolov8n(args.output_dir, args.weights)


if __name__ == "__main__":
    main()
