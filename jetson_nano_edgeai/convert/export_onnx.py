#!/usr/bin/env python3
"""Export PyTorch models to ONNX format for benchmarking."""

import argparse
import os
import sys

import torch
import torchvision.models as models
import torchvision.models.detection as det_models
import onnx


MODEL_CONFIGS = {
    "mobilenetv3_small": {
        "factory": models.mobilenet_v3_small,
        "input_size": (1, 3, 224, 224),
        "opset": 11,
        "task": "classification",
    },
    "resnet50": {
        "factory": models.resnet50,
        "input_size": (1, 3, 224, 224),
        "opset": 11,
        "task": "classification",
    },
    "shufflenet_v2": {
        "factory": models.shufflenet_v2_x1_0,
        "input_size": (1, 3, 224, 224),
        "opset": 11,
        "task": "classification",
    },
    "ssd_mobilenet_v2": {
        "factory": det_models.ssdlite320_mobilenet_v3_large,
        "input_size": (1, 3, 320, 320),
        "opset": 11,
        "task": "detection",
    },
}


def export_model(model_name, output_dir):
    """Export a pretrained model to ONNX."""
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    config = MODEL_CONFIGS[model_name]
    is_detection = config.get("task") == "detection"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}.onnx")

    # Load pretrained model
    print(f"[1/4] Loading {model_name} (pretrained, {'detection' if is_detection else 'classification'})...")
    model = config["factory"](pretrained=True)
    model.eval()

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Create dummy input
    input_size = config["input_size"]
    dummy_input = torch.randn(*input_size)
    print(f"  Input shape: {input_size}")

    # Quick inference test
    print("[2/4] Running test inference...")
    with torch.inference_mode():
        if is_detection:
            output = model([dummy_input[0]])
            print(f"  Output keys: {list(output[0].keys())}")
        else:
            output = model(dummy_input)
            print(f"  Output shape: {tuple(output.shape)}")

    # Export to ONNX
    print(f"[3/4] Exporting to ONNX (opset {config['opset']})...")
    if is_detection:
        # SSD/SSDLite detection models cannot be traced directly via torch.onnx.export
        # because their forward() accesses image.shape which becomes None during tracing.
        # Instead, export only the backbone+feature extractor as a feature encoder,
        # or wrap the model to accept a pre-batched tensor with a fixed size.
        class DetectionWrapper(torch.nn.Module):
            def __init__(self, det_model, img_h, img_w):
                super().__init__()
                self.model = det_model
                self.img_h = img_h
                self.img_w = img_w

            def forward(self, x):
                # x: (1, C, H, W) — pass as a list of single images
                images = [x[i] for i in range(x.shape[0])]
                detections = self.model(images)
                boxes  = detections[0]["boxes"]
                scores = detections[0]["scores"]
                labels = detections[0]["labels"].float()
                return boxes, scores, labels

        wrapper = DetectionWrapper(model, input_size[2], input_size[3])
        wrapper.eval()
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            opset_version=config["opset"],
            input_names=["input"],
            output_names=["boxes", "scores", "labels"],
        )
    else:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=config["opset"],
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size:.1f} MB)")

    # Verify ONNX model
    print("[4/4] Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Print model info
    graph = onnx_model.graph
    print(f"  Inputs:")
    for inp in graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")
    print(f"  Outputs:")
    for out in graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape}")

    print(f"\n[OK] {model_name} exported successfully to {output_path}")
    return output_path


def main():
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
