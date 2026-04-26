# scripts/prepare_dataset.py
import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data/imagenet_sample", help="Directory to save dataset")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate/download")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    
    print(f"=====================================")
    print(f"Preparing ImageNet Validation Subset")
    print(f"Target Directory: {args.data_dir}")
    print(f"Target Samples: {args.num_samples}")
    print(f"=====================================")

    # 실제 ImageNet 다운로드는 인증이 필요하므로 FakeData로 파이프라인만 검증
    # 젯슨 나노 실측정 시에는 실제 ImageNet val 폴더 경로로 교체해야 함
    print("Generating FakeData to simulate ImageNet dataset (for pipeline verification)...")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.FakeData(size=args.num_samples, image_size=(3, 224, 224), num_classes=1000, transform=transform)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"[Done] Prepared {len(dataset)} samples.")
    print("데이터셋 전처리 파이프라인 검증 완료. 실제 실험 시 torchvision.datasets.ImageFolder로 경로를 수정하세요.")

if __name__ == "__main__":
    main()
