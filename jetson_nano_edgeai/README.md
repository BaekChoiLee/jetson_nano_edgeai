# Jetson Nano Edge AI Runtime Benchmark

Jetson Nano에서 11종 런타임의 추론 속도/전력/정확도를 자동 측정하는 벤치마크 파이프라인.

## 폴더 구조

```
jetson-benchmark/
├── run_all.sh              ← 전체 실행 (이것만 돌리면 됨)
├── benchmark/              ← 측정 코드
│   ├── benchmark.py            메인 오케스트레이터 (워밍업10+측정100회)
│   ├── accuracy_eval.py        정확도 평가 (Top-1/Top-5)
│   ├── tegra_parser.py         tegrastats 전력 파서
│   ├── layer_analyzer.py       레이어별 latency 분석
│   ├── run_pytorch.py          PyTorch CPU/CUDA
│   ├── run_tensorrt.py         TensorRT FP32/FP16/INT8
│   ├── run_onnxrt.py           ONNX Runtime CUDA/TRT EP
│   ├── run_tflite.py           TFLite CPU/GPU
│   ├── run_ncnn.py             ncnn CPU/Vulkan
│   └── calc_carbon.py          탄소 배출량 계산
├── convert/                ← 모델 변환
│   ├── export_onnx.py          PyTorch → ONNX
│   ├── build_trt.sh            ONNX → TensorRT (FP32/FP16/INT8)
│   ├── convert_tflite.py       ONNX → TFLite
│   └── convert_ncnn.sh         ONNX → ncnn
├── setup/                  ← 환경 설치
├── data/imagenet_val/      ← ImageNet 1K 검증셋 (클래스당 1장, 총 1000장)
├── models/                 ← 변환된 모델 파일
├── results/                ← 측정 결과 CSV
├── charts/                 ← 시각화 차트 (PNG/PDF)
└── report/                 ← 분석 보고서
```

## 팀원 가이드: 새 모델 추가하기

### 1. 모델 등록

`benchmark/accuracy_eval.py`의 `MODEL_FACTORIES`에 추가:
```python
MODEL_FACTORIES = {
    "mobilenetv3_small": models.mobilenet_v3_small,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,    # ← 추가
}
```

### 2. 실행

```bash
# 특정 모델만 실행
./run_all.sh --model efficientnet_b0

# 주차별 모델 세트 실행
./run_all.sh --week 4    # EfficientNet-B0 + ShuffleNetV2
./run_all.sh --week 6    # YOLOv8n + SSD-MobileNetV2

# 먼저 뭘 하는지 확인만
./run_all.sh --model efficientnet_b0 --dry-run
```

### 3. 주차별 모델 배정

| 주차 | `--week` | 모델 | 담당 |
|:---:|:---:|------|:---:|
| 3 | 3 | MobileNetV3-Small | P1 |
| 4 | 4 | EfficientNet-B0, ShuffleNetV2 | P2, P3 |
| 5 | 5 | ResNet-50 | P1 |
| 6 | 6 | YOLOv8n, SSD-MobileNetV2 | P2, P3 |

### 4. 파이프라인이 자동으로 하는 것

1. 환경 검증 (`setup/verify_env.py`)
2. 모델 변환 (ONNX → TRT/TFLite/ncnn)
3. 11종 런타임 벤치마크 (10W + 5W, 런타임 간 60초 쿨다운)
4. 정확도 평가 (ImageNet 1000장)
5. 레이어별 분석

결과는 `results/` 폴더에 JSON + CSV로 저장됨.

### 5. 주의사항

- TRT `.engine` 파일은 기기별로 다시 빌드해야 함 (`--skip-convert` 쓰지 말 것)
- 실행 전 다른 프로세스 종료할 것 (백그라운드 빌드가 벤치마크를 오염시킴)
- 5W 모드에서 ResNet-50급 무거운 모델은 극심한 요동 발생 — 정상임
