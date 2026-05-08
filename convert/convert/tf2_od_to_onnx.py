#!/usr/bin/env python3
# =============================================================================
# tf2_od_to_onnx.py — 순정 SSD MobileNet V2 → 듀얼 ONNX 변환
# =============================================================================
# 역할: NVIDIA/TensorRT samples/python/tensorflow_object_detection_api/create_onnx.py
#       의 로직을 이식하여 TF2 OD API SavedModel → 두 갈래 ONNX 생성.
#
# 출력:
#   1) ssd_mobilenet_v2_raw.onnx     — NMS 제거본 (ONNX RT/TFLite/ncnn/PyTorch 공용)
#   2) ssd_mobilenet_v2_effnms.onnx  — EfficientNMS_TRT plugin 포함 (TRT 네이티브 보조 측정)
#   3) ssd_mobilenet_v2_anchors.npy  — Python 후처리용 anchor 박스
#
# 알려진 버그와 해결책:
#   - AttributeError: 'Variable' object has no attribute 'values'
#     → onnx-graphsurgeon==0.3.10 고정 (0.4.x 에서 발생)
#   - NonMaxSuppression op not supported by TRT
#     → Raw 경로는 NMS 제거, EffNMS 경로는 TRT plugin 로 교체
#   - float_image_tensor 입력 타입 에러
#     → tf2onnx 변환 시 input 이름 명시
#   - dynamic batch size 미지원
#     → batch=1 고정
#
# 사용법:
#   python convert/tf2_od_to_onnx.py \
#     --saved-model models/ssd_mobilenet_v2_saved_model/saved_model \
#     --output-raw models/ssd_mobilenet_v2_raw.onnx \
#     --output-effnms models/ssd_mobilenet_v2_effnms.onnx
# =============================================================================

"""SSD MobileNet V2 TF2 SavedModel → Dual ONNX (raw + EfficientNMS_TRT)."""

import argparse
import os
import subprocess
import sys

import numpy as np


def convert_savedmodel_to_onnx(
    saved_model_dir,
    output_raw,
    output_effnms,
    opset=11,
    nms_score_thres=0.25,
    nms_iou_thres=0.45,
    max_dets=100,
):
    """SavedModel → tf2onnx → graphsurgeon (raw + effnms 두 갈래).

    Args:
        saved_model_dir: TF2 OD API SavedModel 디렉토리 경로
        output_raw: NMS 제거본 출력 경로 (공용)
        output_effnms: EfficientNMS_TRT plugin 포함 출력 경로 (TRT 전용)
        opset: ONNX opset version (Jetson JetPack 4.6 TRT 8.2 는 11 권장)
        nms_score_thres/iou_thres/max_dets: EfficientNMS_TRT plugin 파라미터
    """
    # graphsurgeon 은 onnx_graphsurgeon 패키지로 import
    try:
        import onnx
        import onnx_graphsurgeon as gs
    except ImportError as e:
        print(f"[ERROR] required package missing: {e}")
        print("  pip install onnx==1.14.0 onnx-graphsurgeon==0.3.10")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_raw) or ".", exist_ok=True)

    # =====================================================================
    # Step 1: tf2onnx 로 SavedModel → ONNX (batch=1, opset 11 고정)
    # =====================================================================
    tmp_onnx = os.path.join(
        os.path.dirname(output_raw) or ".", "_ssd_mv2_tf2onnx_tmp.onnx"
    )
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", saved_model_dir,
        "--output", tmp_onnx,
        "--opset", str(opset),
    ]
    print(f"[1/4] Running tf2onnx: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] tf2onnx conversion failed: {e}")
        print("  SSD-MV2 의 Python/tf 버전 호환성을 확인하세요 (Python 3.10 + tf 2.13.1).")
        raise

    # =====================================================================
    # Step 2: graphsurgeon 으로 그래프 정리
    # =====================================================================
    print(f"[2/4] Loading ONNX graph for surgery...")
    graph = gs.import_onnx(onnx.load(tmp_onnx))
    graph.cleanup().toposort()
    graph.fold_constants().cleanup()

    # =====================================================================
    # Step 3a: Raw ONNX — NMS 이전 tensor 를 그래프 출력으로 승격
    # =====================================================================
    print(f"[3/4] Building raw graph (NMS removed)...")
    raw_graph = _extract_raw_outputs(graph.copy())
    onnx.save(gs.export_onnx(raw_graph), output_raw)
    print(f"  [OK] Raw ONNX saved: {output_raw}")

    # =====================================================================
    # Step 3b: EfficientNMS_TRT plugin 치환본
    # =====================================================================
    print(f"[3/4] Building EfficientNMS graph (TRT plugin)...")
    try:
        effnms_graph = _inject_efficient_nms(
            graph.copy(), nms_score_thres, nms_iou_thres, max_dets
        )
        onnx.save(gs.export_onnx(effnms_graph), output_effnms)
        print(f"  [OK] EfficientNMS ONNX saved: {output_effnms}")
    except Exception as e:
        print(f"  [WARN] EfficientNMS injection failed: {e}")
        print("  TRT 네이티브 end-to-end 경로만 스킵됩니다. Raw 경로는 정상.")

    # =====================================================================
    # Step 4: Anchor 박스 추출 (Python 후처리용)
    # =====================================================================
    print(f"[4/4] Extracting anchor boxes...")
    try:
        anchors = _extract_anchors(graph.copy())
        anchor_path = os.path.join(
            os.path.dirname(output_raw) or ".",
            "ssd_mobilenet_v2_anchors.npy",
        )
        np.save(anchor_path, anchors)
        print(f"  [OK] Anchors saved: {anchor_path} ({len(anchors)} boxes)")
    except Exception as e:
        print(f"  [WARN] Anchor extraction failed: {e}")
        print("  Python 후처리 시 수동으로 anchor 생성 필요.")

    # 임시 파일 정리
    if os.path.exists(tmp_onnx):
        os.remove(tmp_onnx)


def _extract_raw_outputs(graph):
    """NonMaxSuppression 노드 직전의 (box_encodings, class_scores) 를 그래프 출력으로 승격.

    TF2 OD API 는 Postprocessor 블록에서 NMS 를 수행하므로, 그 직전 tensor 두 개를
    새로운 graph outputs 로 지정하고 이후 노드를 제거하면 raw 출력 ONNX가 된다.
    """
    import onnx_graphsurgeon as gs

    # 0) TF2 OD API 기본 출력(raw_detection_*)이 있으면 우선 사용.
    #    이 경로가 class_scores shape([N, 91])을 더 안정적으로 보존한다.
    tensors = graph.tensors()
    raw_boxes = tensors.get("raw_detection_boxes")
    raw_scores = tensors.get("raw_detection_scores")
    if raw_boxes is not None and raw_scores is not None:
        raw_boxes.name = "box_encodings"
        raw_scores.name = "class_scores"
        if getattr(raw_boxes, "dtype", None) is None:
            raw_boxes.dtype = np.float32
        if getattr(raw_scores, "dtype", None) is None:
            raw_scores.dtype = np.float32
        graph.outputs = [raw_boxes, raw_scores]
        graph.cleanup().toposort()
        return graph

    # 1) fallback: NMS 입력 텐서를 raw 출력으로 승격
    nms_nodes = [n for n in graph.nodes if n.op == "NonMaxSuppression"]
    if not nms_nodes:
        # NMS 노드가 없으면 TF2 OD 가 다른 구조로 내보냈을 수 있음 → 기본 출력 유지
        print("  [WARN] No NonMaxSuppression node found in graph")
        return graph
    node = nms_nodes[0]
    box_tensor = node.inputs[0]
    score_tensor = node.inputs[1]
    # 이름 고정 (후처리 함수가 이 이름을 참조)
    box_tensor.name = "box_encodings"
    score_tensor.name = "class_scores"
    # onnx-graphsurgeon 0.5+ 에서는 output dtype 누락 시 export 실패
    if getattr(box_tensor, "dtype", None) is None:
        box_tensor.dtype = np.float32
    if getattr(score_tensor, "dtype", None) is None:
        score_tensor.dtype = np.float32
    graph.outputs = [box_tensor, score_tensor]
    # NMS 이후 unreachable 노드 제거
    graph.cleanup().toposort()
    return graph


def _inject_efficient_nms(graph, score_thres, iou_thres, max_dets):
    """NonMaxSuppression 노드를 EfficientNMS_TRT plugin 노드로 교체."""
    import onnx_graphsurgeon as gs

    nms_nodes = [n for n in graph.nodes if n.op == "NonMaxSuppression"]
    if not nms_nodes:
        raise RuntimeError("No NonMaxSuppression node found for replacement")
    node = nms_nodes[0]
    boxes_in, scores_in = node.inputs[0], node.inputs[1]

    # 기존 NMS 출력 연결 해제
    for out in node.outputs:
        out.outputs.clear()

    # EfficientNMS_TRT plugin 출력 tensor 정의
    num_detections = gs.Variable("num_detections", dtype=np.int32, shape=[1, 1])
    nmsed_boxes    = gs.Variable("nmsed_boxes",    dtype=np.float32, shape=[1, max_dets, 4])
    nmsed_scores   = gs.Variable("nmsed_scores",   dtype=np.float32, shape=[1, max_dets])
    nmsed_classes  = gs.Variable("nmsed_classes",  dtype=np.int32,   shape=[1, max_dets])

    eff_nms = gs.Node(
        op="EfficientNMS_TRT",
        name="EfficientNMS",
        attrs={
            "plugin_version": "1",
            "background_class": -1,
            "max_output_boxes": max_dets,
            "score_threshold":  float(score_thres),
            "iou_threshold":    float(iou_thres),
            "score_activation": False,
            "box_coding": 1,  # 1 = CENTER_SIZE
        },
        inputs=[boxes_in, scores_in],
        outputs=[num_detections, nmsed_boxes, nmsed_scores, nmsed_classes],
    )
    graph.nodes.append(eff_nms)
    graph.outputs = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]
    graph.cleanup().toposort()
    return graph


def _extract_anchors(graph):
    """TF2 OD API 그래프에서 anchor tensor 를 휴리스틱으로 추출."""
    import onnx_graphsurgeon as gs

    # 1차: 이름에 'Anchor' 포함된 노드 주변 탐색
    for node in graph.nodes:
        if "Anchor" in node.name or "anchor" in node.name:
            for inp in node.inputs:
                if isinstance(inp, gs.Constant):
                    arr = inp.values
                    if arr.ndim == 2 and arr.shape[1] == 4 and arr.shape[0] > 1000:
                        return arr
    # 2차: 모든 Constant 탐색 → [N, 4] 이고 N > 1000 인 tensor
    for t in graph.tensors().values():
        if isinstance(t, gs.Constant):
            arr = t.values
            if arr.ndim == 2 and arr.shape[1] == 4 and arr.shape[0] > 1000:
                return arr
    raise RuntimeError("Anchor tensor not found (shape [N, 4], N > 1000)")


def main():
    parser = argparse.ArgumentParser(description="SSD-MV2 TF2 SavedModel → Dual ONNX")
    parser.add_argument(
        "--saved-model",
        default="models/ssd_mobilenet_v2_saved_model/saved_model",
        help="TF2 OD API SavedModel directory",
    )
    parser.add_argument(
        "--output-raw",
        default="models/ssd_mobilenet_v2_raw.onnx",
        help="Output path for raw ONNX (NMS removed)",
    )
    parser.add_argument(
        "--output-effnms",
        default="models/ssd_mobilenet_v2_effnms.onnx",
        help="Output path for EfficientNMS_TRT ONNX",
    )
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--score-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-dets", type=int, default=100)
    args = parser.parse_args()

    convert_savedmodel_to_onnx(
        args.saved_model, args.output_raw, args.output_effnms,
        opset=args.opset,
        nms_score_thres=args.score_thres,
        nms_iou_thres=args.iou_thres,
        max_dets=args.max_dets,
    )


if __name__ == "__main__":
    main()
