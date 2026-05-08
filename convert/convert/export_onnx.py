#!/usr/bin/env python3
# ============================================================================
# export_onnx.py - PyTorch 모델을 ONNX 형식으로 변환하는 스크립트
# ============================================================================
# 역할: 변환 파이프라인의 첫 번째 단계
#   - torchvision에서 사전학습(pretrained) 모델을 로드
#   - 더미 입력으로 추론 테스트를 수행하여 모델 정상 동작 확인
#   - ONNX 형식으로 내보내기 (export)
#   - 내보낸 ONNX 파일의 유효성 검증
#
# 사용법:
#   python export_onnx.py --model mobilenetv3_small --output-dir ./models
#
# 지원 모델:
#   - mobilenetv3_small: 경량 모바일 모델 (MobileNetV3-Small)
#   - resnet50: 범용 분류 모델 (ResNet-50)
#
# 파이프라인 흐름:
#   [이 스크립트] → build_trt.sh (TensorRT)
#                 → convert_tflite.py (TFLite)
#                 → convert_ncnn.sh (ncnn)
# ============================================================================
"""Export PyTorch models to ONNX format for benchmarking."""

import argparse
import os
import sys

import torch
import torchvision.models as models
import onnx


# 지원하는 모델 설정 딕셔너리
# - factory: torchvision에서 모델을 생성하는 팩토리 함수
# - input_size: 모델 입력 텐서 크기 (배치, 채널, 높이, 너비)
# - opset: ONNX 오퍼레이터 셋 버전 (호환성에 영향)
MODEL_CONFIGS = {
    "mobilenetv3_small": {
        "factory": models.mobilenet_v3_small,
        "input_size": (1, 3, 224, 224),
        "opset": 11,
    },
    "resnet50": {
        "factory": models.resnet50,
        "input_size": (1, 3, 224, 224),
        "opset": 11,
    },
    "efficientnet_b0": {
        "factory": models.efficientnet_b0,
        "input_size": (1, 3, 224, 224),
        "opset": 11,
    },
    "shufflenet_v2_x1_0": {
        "factory": models.shufflenet_v2_x1_0,
        "input_size": (1, 3, 224, 224),
        "opset": 11,
    },
}


def export_model(model_name, output_dir):
    """사전학습된 PyTorch 모델을 ONNX 형식으로 내보내는 함수.

    4단계로 진행:
      [1/4] 모델 로드 → [2/4] 추론 테스트 → [3/4] ONNX 내보내기 → [4/4] 유효성 검증
    """
    # 모델 이름이 지원 목록에 있는지 확인
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    config = MODEL_CONFIGS[model_name]
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}.onnx")

    # [1/4] 사전학습된 가중치를 포함한 모델 로드
    print(f"[1/4] Loading {model_name} (pretrained)...")
    model = config["factory"](pretrained=True)
    # eval() 모드로 전환: 드롭아웃/배치정규화 등이 추론 모드로 동작
    model.eval()

    # 모델 파라미터 수 계산 (모델 크기 파악용)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # ONNX 내보내기에 필요한 더미(dummy) 입력 텐서 생성
    # 랜덤 값으로 채워진 텐서 - 실제 이미지가 아니라 모델 그래프 추적용
    input_size = config["input_size"]
    dummy_input = torch.randn(*input_size)
    print(f"  Input shape: {input_size}")

    # [2/4] 내보내기 전 모델이 정상 동작하는지 추론 테스트
    print("[2/4] Running test inference...")
    # inference_mode: 자동미분(autograd) 비활성화로 메모리 절약
    with torch.inference_mode():
        output = model(dummy_input)
    print(f"  Output shape: {tuple(output.shape)}")

    # [3/4] PyTorch 모델을 ONNX 형식으로 내보내기
    print(f"[3/4] Exporting to ONNX (opset {config['opset']})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=config["opset"],
        # ONNX 그래프에서 사용할 입출력 이름 지정
        input_names=["input"],
        output_names=["output"],
        # 동적 축(dynamic axes) 설정: 배치 크기를 가변으로 허용
        # 이렇게 하면 추론 시 배치 크기를 자유롭게 변경 가능
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # 내보낸 파일 크기 출력 (MB 단위)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size:.1f} MB)")

    # [4/4] ONNX 모델 유효성 검증
    # 그래프 구조가 올바른지, 노드/엣지에 문제가 없는지 확인
    print("[4/4] Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # 검증된 ONNX 모델의 입출력 텐서 형상(shape) 정보 출력
    graph = onnx_model.graph
    print(f"  Inputs:")
    for inp in graph.input:
        # 각 차원(dim)의 값을 리스트로 추출
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")
    print(f"  Outputs:")
    for out in graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape}")

    print(f"\n[OK] {model_name} exported successfully to {output_path}")
    return output_path


def main():
    """CLI 진입점: 명령줄 인자를 파싱하여 모델 내보내기를 실행."""
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to export",
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Output directory (default: ./models)",
    )
    args = parser.parse_args()

    export_model(args.model, args.output_dir)


if __name__ == "__main__":
    main()
