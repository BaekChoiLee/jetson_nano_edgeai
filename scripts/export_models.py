# scripts/export_models.py
import os
import torch
import torchvision.models as models
import argparse

def export_to_onnx(model, model_name, output_dir):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    
    print(f"Exporting {model_name} to ONNX...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[Done] Saved ONNX to {onnx_path}")
    return onnx_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=====================================")
    print("Exporting PyTorch Models to ONNX")
    print("=====================================")

    # 1. MobileNetV3-Small
    mbv3 = models.mobilenet_v3_small(pretrained=True)
    export_to_onnx(mbv3, "mobilenetv3s", args.output_dir)

    # 2. EfficientNet-B0
    effb0 = models.efficientnet_b0(pretrained=True)
    export_to_onnx(effb0, "efficientnetb0", args.output_dir)

    # 3. ShuffleNetV2 (1.0x)
    shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
    export_to_onnx(shufflenet, "shufflenetv2", args.output_dir)

    # 4. ResNet-50
    resnet50 = models.resnet50(pretrained=True)
    export_to_onnx(resnet50, "resnet50", args.output_dir)
    
    print("=====================================")
    print("Export Complete! TFLite 및 TensorRT 엔진 변환은 젯슨 기기에서 아래 명령어를 참고하세요.")
    print("TFLite: python -m tf2onnx.convert --onnx model.onnx --output model.tflite")
    print("TensorRT: trtexec --onnx=model.onnx --saveEngine=model.engine --fp16")
    print("ncnn: onnx2ncnn model.onnx model.param model.bin")

if __name__ == "__main__":
    main()
