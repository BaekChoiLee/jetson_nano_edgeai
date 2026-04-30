# Jetson Nano Edge AI Runtime Benchmark

Jetson Nano에서 **6종 모델 × 11종 런타임 × 2종 전력모드 = 132가지** 추론 속도/전력/정확도를 자동 측정하는 벤치마크 파이프라인.

## 측정 항목

| 측정 항목 | 도구 | 단위 | 비고 |
|---------|------|------|------|
| 추론 레이턴시 | time.perf_counter / CUDA Event | ms | warm-up 10회, 측정 100회 |
| GPU 메모리 | tegrastats | MB | CPU·GPU 공유 메모리 |
| 전력 소비 | tegrastats (INA3221) | mW | 5W / 10W 각각 측정 |
| 레이어별 latency | forward_hook (분류) / autograd.profiler (탐지) | ms/layer | 6모델 전부 |
| 탄소 배출량 | calc_carbon.py | gCO₂eq | 전력(Wh) × 탄소 계수 |
| 모델 정확도 | Top-1/Top-5 (분류) / mAP (탐지) | %, mAP | 런타임별 정확도 차이 검증 |

## 모델 매트릭스

| 모델 | 종류 | 입력 크기 | 파라미터 | 주차 |
|------|------|----------|---------|------|
| MobileNetV3-Small | Classification | 224×224 | 2.5M | 3 |
| ResNet-50 | Classification | 224×224 | 25.6M | 5 |
| EfficientNet-B0 | Classification | 224×224 | 5.3M | 4 |
| ShuffleNetV2 x1.0 | Classification | 224×224 | 2.3M | 4 |
| YOLOv8n | Detection | 640×640 | 3.2M | 6 |
| SSD-MobileNetV2 | Detection | 320×320 | 3.4M | 6 |

### 런타임 (11종)

| # | 런타임 | 백엔드 |
|---|--------|--------|
| 1 | pytorch_cpu | PyTorch CPU |
| 2 | pytorch_cuda | PyTorch CUDA |
| 3 | tensorrt_fp32 | TensorRT FP32 |
| 4 | tensorrt_fp16 | TensorRT FP16 |
| 5 | tensorrt_int8 | TensorRT INT8 |
| 6 | onnxrt_cuda | ONNX Runtime CUDA EP |
| 7 | onnxrt_trt | ONNX Runtime TRT EP |
| 8 | tflite_cpu | TFLite CPU |
| 9 | tflite_gpu | TFLite GPU Delegate |
| 10 | ncnn_cpu | ncnn CPU |
| 11 | ncnn_vulkan | ncnn Vulkan |

### 132 = 6 × 11 × 2 (skip 없음)

모든 조합을 실행하며, 특정 런타임이 실패하면 에러 결과로 기록하고 다음으로 진행합니다.

> **하드웨어 제약 참고**:
> - ShuffleNetV2의 `tflite_gpu`는 channel shuffle ops가 GPU delegate에서 미지원 → CPU 폴백 (실행은 됨)
> - `ncnn_vulkan`은 Jetson Nano Maxwell GPU에서 Vulkan 미지원(`vkCreateInstance -9`) → CPU 폴백
> - `tflite_gpu`는 `libdelegate_gpu.so`가 없으면 CPU 폴백

## 폴더 구조

```
jetson-benchmark/
├── run_all.sh              ← 전체 실행 (이것만 돌리면 됨)
├── benchmark/              ← 측정 코드
│   ├── benchmark.py            메인 오케스트레이터 (워밍업10+측정100회+tegrastats 전력)
│   ├── accuracy_eval.py        분류 정확도 (Top-1/Top-5, 11개 런타임별)
│   ├── run_detection_accuracy.py   탐지 정확도 (mAP, COCO 기반)
│   ├── detection_infer.py         탐지 추론 래퍼
│   ├── detection_postprocess.py   탐지 후처리 (NMS 등)
│   ├── detection_eval.py          탐지 평가 유틸리티
│   ├── run_pytorch.py          PyTorch CPU/CUDA
│   ├── run_tensorrt.py         TensorRT FP32/FP16/INT8
│   ├── run_onnxrt.py           ONNX Runtime CUDA/TRT EP
│   ├── run_tflite.py           TFLite CPU/GPU
│   ├── run_ncnn.py             ncnn CPU/Vulkan
│   ├── tegra_parser.py         tegrastats 전력 파서
│   ├── layer_analyzer.py       레이어별 latency (분류: forward hooks, 탐지: autograd profiler)
│   ├── calc_carbon.py          탄소 배출량 계산 (전력 데이터 기반)
│   └── collect_memory.py       메모리 사용량 수집
├── convert/                ← 모델 변환
│   ├── export_onnx.py          PyTorch → ONNX (분류)
│   ├── export_detection.py     PyTorch → ONNX (탐지)
│   ├── build_trt.sh            ONNX → TensorRT (FP32/FP16/INT8)
│   ├── build_trt_int8.py       INT8 캘리브레이션
│   ├── convert_tflite.py       ONNX → TFLite
│   └── convert_ncnn.sh         ONNX → ncnn
├── setup/                  ← 환경 설치
│   ├── setup_env.sh            기본 패키지
│   ├── install_onnxrt.sh       ONNX Runtime (CUDA/TRT EP 포함)
│   ├── install_ncnn.sh         ncnn (Python + benchncnn)
│   ├── install_tflite.sh       TFLite Runtime
│   └── verify_env.py           설치 검증
├── data/
│   ├── imagenet_val/           ImageNet 검증셋 (분류 정확도용, Git LFS)
│   └── coco_val/               ⚠️ 별도 준비 필요 (아래 참조)
├── models/                 ← 변환된 모델 파일 (tar에서 복원)
└── results/                ← 측정 결과 CSV/JSON
```

## 빠른 시작

```bash
# 1. clone
git clone -b jh https://github.com/BaekChoiLee/jetson_nano_edgeai.git
cd jetson_nano_edgeai

# 2. 모델 복원 (split parts → tar → models/)
cat models.tar.gz.part_* > models.tar.gz
tar xzf models.tar.gz
rm models.tar.gz    # 디스크 절약 (선택)

# 3. 최신 결과 복원 (선택, JSON/CSV만 포함)
tar xzf results.tar.gz

# 4. 환경 설치 (Jetson Nano에서)
bash setup/setup_env.sh
bash setup/install_onnxrt.sh
bash setup/install_ncnn.sh
bash setup/install_tflite.sh

# 5. 추가 의존성 (수동 설치 필요)
pip3 install pycocotools pycuda psutil

# 6. COCO 데이터 준비 (detection accuracy에 필요)
# → 아래 "COCO 데이터 준비" 섹션 참조

# 7. 확인 (dry-run)
bash run_all.sh --dry-run

# 8. 풀 벤치마크 실행 (132가지, 약 6-7시간 소요)
bash run_all.sh
```

### COCO 데이터 준비

Detection accuracy 평가를 위해 COCO val2017 데이터가 필요합니다. **GitHub에는 포함되어 있지 않으므로 별도로 준비해야 합니다.**

```bash
mkdir -p data/coco_val/{images,annotations}

# 이미지 (COCO val2017에서 500장 서브셋 또는 전체)
# 방법 1: 전체 val2017 다운로드
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d data/coco_val/images/

# 방법 2: 이미 val2017이 있으면 심볼릭 링크
ln -s /path/to/val2017 data/coco_val/images

# 어노테이션 (500장 서브셋)
# instances_val2017_subset500.json을 data/coco_val/annotations/에 배치
```

> COCO 데이터가 없으면 detection accuracy 22건은 자동 skip되며, 나머지 132 latency + 44 classification accuracy + 6 layer analysis는 정상 실행됩니다.

## 실행 옵션

```bash
# 특정 모델만 실행
./run_all.sh --model efficientnet_b0

# 주차별 모델 세트 실행
./run_all.sh --week 3    # MobileNetV3-Small
./run_all.sh --week 4    # EfficientNet-B0 + ShuffleNetV2
./run_all.sh --week 5    # ResNet-50
./run_all.sh --week 6    # YOLOv8n + SSD-MobileNetV2

# 변환 건너뛰기 (이미 .engine 파일이 있을 때)
./run_all.sh --skip-convert

# 상세 레이어 런타임 분석까지 함께 실행
./run_all.sh --detailed-layers

# 매우 느린 레이어별 전력/에너지 분석과 타임라인까지 포함
./run_all.sh --detailed-layers --include-layer-power --include-layer-timelines

# 먼저 뭘 하는지 확인만
./run_all.sh --dry-run
```

## 파이프라인 Step별 설명

`run_all.sh`가 자동으로 실행하는 단계:

1. **Step 0: 환경 검증** — `verify_env.py`로 PyTorch/TRT/ONNXRT/TFLite/ncnn 설치 확인
2. **Step 1: 모델 변환** — ONNX export → TRT engine 빌드 (FP32/FP16/INT8) → TFLite 변환 → ncnn 변환
3. **Step 2: 132가지 벤치마크** — 6모델 × 11런타임 × 2전력모드, 런타임 간 60초 쿨다운, tegrastats 전력 병렬 로깅
4. **Step 3: 정확도 평가**
   - 분류 (4모델): Top-1/Top-5 accuracy, ImageNet 검증셋, 11개 런타임별
   - 탐지 (2모델): mAP@0.5 / mAP@0.5:0.95, COCO val2017, 11개 런타임별
5. **Step 4: 레이어별 분석** — 6모델 전부 (분류: PyTorch forward hooks, 탐지: torch.autograd.profiler)

결과는 `results/` 폴더에 JSON + CSV로 저장됨.

## 상세 레이어 데이터 추출

기본 `run_all.sh`의 Step 4는 모델별 요약 레이어 CSV를 만듭니다. 보고서용으로 런타임별 병목을 더 자세히 보려면 새 진입점인 `run_detailed_layers.sh`를 사용합니다.

```bash
# 6모델 × 11런타임 상세 layer/node/operator latency
bash run_detailed_layers.sh

# 특정 모델만
bash run_detailed_layers.sh --model yolov8n

# 특정 런타임만
bash run_detailed_layers.sh --runtime tensorrt_fp16

# 전력/에너지까지 포함 (느림, Layer Amplification Loop)
bash run_detailed_layers.sh --include-power

# Chrome trace / TensorRT profile JSON까지 포함
bash run_detailed_layers.sh --include-timelines
```

출력 구조:

```text
results/detailed_layers_<timestamp>/
├── architecture/                         # PyTorch/TorchScript module summary
│   └── <model>_layers.csv                # layer_name/type/params/input/output/latency
├── layer_runtime/
│   ├── <model>_<runtime>.csv             # rank, layer/node/op, mean/std/min/max, samples
│   └── <model>_<runtime>.json            # same records + metadata
├── consolidated_layer_runtime_detailed.csv
├── layer_power/                          # --include-power 사용 시
│   └── <model>_layer_power.csv/json      # latency, power, energy_per_run_mj
└── layer_timelines/                      # --include-timelines 사용 시
    └── *.json                            # chrome://tracing 또는 TRT profile viewer용
```

`layer_runtime`의 granularity는 런타임별로 다릅니다.

| 런타임 | granularity |
|--------|-------------|
| pytorch_cpu / pytorch_cuda | PyTorch operator |
| tensorrt_fp32 / fp16 / int8 | TensorRT layer |
| onnxrt_cuda / onnxrt_trt | ONNX Runtime node |
| tflite_cpu / tflite_gpu | TFLite node |
| ncnn_cpu / ncnn_vulkan | ncnn layer |

TFLite/ncnn 레이어 단위 profiling은 별도 도구가 필요합니다.

```bash
bash setup/build_layer_tools.sh
export TFLITE_BENCHMARK_MODEL_BIN=$HOME/tensorflow-2.13.0/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model
export NCNN_BENCHNCNN_BIN=$HOME/ncnn/build_bench/benchmark/benchncnn
```

## 모델 파일 (tar 내)

| 모델 | .onnx | .tflite | .param/.bin | .ncnn.param/.bin | .torchscript | 기타 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| mobilenetv3_small | O | O | O | O | - | |
| resnet50 | O | O | O | O | - | |
| efficientnet_b0 | O | O | O | O | - | |
| shufflenet_v2_x1_0 | O | O | O | O | - | |
| yolov8n | O | O | O | O | O | |
| ssd_mobilenet_v2 | _raw_fpinput.onnx | O | O | O | O | _raw.onnx, _anchors.npy |

> - `.engine` 파일은 기기별로 빌드해야 하므로 tar에 포함되지 않습니다.
> - SSD의 primary ONNX는 `_raw_fpinput.onnx` (float32 입력). `_raw.onnx`는 uint8 입력 버전.

## 주의사항

- TRT `.engine` 파일은 기기별로 다시 빌드해야 함 (첫 실행 시 `--skip-convert` 쓰지 말 것)
- 실행 전 다른 프로세스 종료할 것 (백그라운드 빌드가 벤치마크를 오염시킴)
- 5W 모드에서 ResNet-50급 무거운 모델은 극심한 요동 발생 — 정상임
- 탐지 모델(YOLOv8n)은 `ultralytics` 패키지 필요: `pip3 install ultralytics`
- SSD의 TRT 엔진은 `_raw_fpinput.onnx`로부터 빌드 후 `ssd_mobilenet_v2_{precision}.engine`으로 자동 rename됨
- **COCO 데이터가 없으면** detection accuracy만 skip, 나머지 전부 실행됨
