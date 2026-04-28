# Jetson Nano Edge AI Runtime Benchmark
**멀티 런타임 × 비전 모델 성능·전력·메모리 비교 분석**

본 프로젝트는 동일한 비전 AI 모델을 **NVIDIA Jetson Nano** 온디바이스 환경에서 다양한 런타임 프레임워크로 실행하여 추론 속도($Latency$), 메모리 사용량, 전력 소비를 정밀하게 측정하고 비교 분석합니다. 단순한 성능 비교를 넘어 레이어 수준($Layer-level$)의 심화 분석과 탄소 배출량 환산을 통해 엣지 AI 시스템 설계의 실증적 가이드라인을 제시하는 것을 목표로 합니다.

---

## 🛠 Hardware Specifications
실험에 사용된 Jetson Nano의 하드웨어 사양입니다. 4GB 공유 메모리 환경이므로 원활한 실험을 위해 **스왑(Swap) 메모리 추가 설정**을 권장합니다.

| 항목 | 사양 | 비고 |
| :--- | :--- | :--- |
| **CPU** | Quad-core ARM Cortex-A57 @ 1.43GHz | 4코어 |
| **GPU** | 128-core Maxwell @ 921MHz | CUDA 지원 |
| **RAM** | 4GB LPDDR4 (CPU·GPU 공유) | 스왑 추가 권장 |
| **Power Mode** | 10W MODE | J48 점퍼로 전환 |
| **OS / SDK** | Ubuntu 18.04 + JetPack 4.x | CUDA·cuDNN·TRT 포함 |

---

## 🧪 Experimental Design

### 1. Framework Matrix
총 5개(필요시 6개)의 런타임 환경에서 순차 실행하여 동일 조건 하에 성능을 비교합니다.
* **PyTorch (CPU/CUDA):** 공식 지원 기반의 베이스라인 및 레이어 분석.
* **TensorRT:** FP32/FP16/INT8 양자화 적용 및 최고 수준의 최적화 달성.
* **ONNX Runtime:** CUDA EP 및 TRT EP 비교를 통한 크로스플랫폼 기준 정립.
* **TFLite:** 모바일 경량 런타임 비교.
* **ncnn:** Vulkan 가속을 포함한 임베디드 전용 가속기 성능 확인.

### 2. Experimental Models (P1~P3)
경량부터 중량 모델까지 다양한 스펙트럼을 가진 6개 모델을 선정하였습니다.
* **분류(Classification):** MobileNetV3-Small, ResNet-50, EfficientNet-B0, ShuffleNet V2.
* **탐지(Detection):** YOLOv8n, SSD-MobileNet V2.

### 3. Measurement Metrics
공정한 비교를 위해 모든 실험은 **Warm-up 10회 후 본 측정 100회 평균값**을 사용합니다.
* **추론 레이턴시:** $ms$ 단위 측정 (perf_counter / CUDA Event).
* **메모리/전력:** `tegrastats`를 활용한 리얼타임 로깅.
* **에너지 효율:** 전력(Wh) 대비 성능 분석 및 탄소 배출량($gCO_2eq$) 환산.
* **심화 분석:** $ms/layer$ 단위의 레이어별 Latency 분석.

---

## 🏃 Workflow
하나의 모델을 모든 런타임에서 완벽히 실행한 후 다음 모델로 이동하는 '1모델 완결 원칙'을 준수합니다.

1.  **모델 준비:** PyTorch 원본 모델 로드 및 정확도 기록.
2.  **모델 변환:** ONNX Export 및 각 런타임용 Engine/Model 빌드.
3.  **런타임별 추론:** PyTorch부터 ncnn까지 순차적 추론 실행.
4.  **로깅 및 분석:** 추론 중 전력·메모리 병렬 로깅 및 레이어 분석(Hook/Profiler) 수행.
5.  **모드 전환:** 5W 전력 모드에서 동일 실험 재반복 측정.
6.  **결과 저장:** CSV / JSON 형태로 구조화하여 데이터 축적.

---

## MacBook / Jetson Split Pipeline

TensorRT engine은 Jetson Nano의 GPU, CUDA, TensorRT, JetPack 버전에 종속되므로 Jetson에서 생성합니다. ONNX, TFLite, ncnn 모델은 MacBook에서 미리 만들고 Git LFS로 전달할 수 있습니다.

### MacBook

필요 도구 예시:

```bash
brew install git-lfs ncnn

# TensorFlow/TFLite 변환은 Python 3.10/3.11 가상환경 권장
python3.11 -m venv .venv-convert
source .venv-convert/bin/activate
pip install --upgrade pip
pip install tensorflow tf-keras onnx onnx-tf onnx2tf
```

`brew install ncnn` 후 `onnx2ncnn`이 PATH에 없으면 경로를 확인해 환경변수로 넘깁니다.

```bash
find /opt/homebrew -name onnx2ncnn 2>/dev/null
ONNX2NCNN_PATH=/path/to/onnx2ncnn bash run_pipeline_mac.sh
```

```bash
git lfs install
bash run_pipeline_mac.sh
git add .gitattributes .gitignore scripts run_pipeline*.sh models
git commit -m "Add converted model artifacts"
git push
```

### Jetson Nano

```bash
git lfs install
git pull
git lfs pull
bash run_pipeline_jetson.sh
```

`run_pipeline_jetson.sh`는 기본적으로 준비된 런타임만 실행합니다. MacBook에서 `.tflite`, `.param`, `.bin`이 생성되어 있고 Jetson에 `benchmark_model`, `benchncnn`이 설치되어 있으면 `tflite_cpu`, `ncnn_vulkan`도 자동으로 포함됩니다.

수동으로 전체 런타임을 지정할 수도 있습니다.

```bash
RUNTIMES=pytorch_cuda,tensorrt_fp32,tensorrt_fp16,tensorrt_int8,onnxrt_cpu,tflite_cpu,ncnn_vulkan bash run_pipeline_jetson.sh
```

도구가 PATH 밖에 있으면 경로를 환경변수로 지정합니다.

```bash
TFLITE_BENCHMARK_MODEL_PATH=/path/to/benchmark_model \
BENCHNCNN_PATH=/path/to/benchncnn \
bash run_pipeline_jetson.sh
```

기존 진입점인 `bash run_pipeline.sh`는 Jetson용 파이프라인으로 연결됩니다.

---

## 📅 Project Roadmap (8 Weeks) 
* **1~2주차:** 환경 구축(`setup.sh`) 및 자동화 측정 파이프라인 개발 .
* **3~6주차:** 모델별 전체 런타임 실험 실행 및 원시 데이터셋 완성.
* **7주차:** 데이터 시각화(산점도, 히트맵, 폭포수 차트) 및 통계 분석.
* **8주차:** 학술 보고서 작성 및 GitHub 코드/데이터 공개.

---

## 💡 Expected Outcomes 
* TensorRT INT8 적용 시 PyTorch CPU 대비 **5~15배**의 추론 속도 향상 예상.
* 엣지 AI 개발자를 위한 멀티 런타임 체계적 데이터셋 및 가이드라인 최초 공개.
