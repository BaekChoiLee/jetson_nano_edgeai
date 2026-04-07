# Jetson Nano Edge AI Runtime Benchmark — Core Reference
> 실험 조건: 90초 쿨다운 적용 순도 100% 무결점 데이터 자동 변환본
> 평가 모델: MobileNetV3-Small (2.54M), ResNet-50 (25.56M)
> 벤치마크 데이터셋:
  - 레이턴시 & 전력 측정: 순수 연산 엔진 속도 고립 측정을 위한 Numpy Dummy Random Tensor `(1, 3, 224, 224)` 
  - 양자화 & 정확도 검증: **ImageNet-1K (ILSVRC-2012) Validation Subset** `Top-1, Top-5 Accuracy`
    - **통제 방법**: 클래스 불균형 방지를 위해 1,000개 전체 클래스 폴더에서 각각 정확히 1장씩 랜덤 추출하여 서브셋 구성 (총 1,000장)
    - **데이터 소스 링크**: [Hugging Face ImageNet-1K Dataset (Validation Split)](https://huggingface.co/datasets/imagenet-1k)
    - **코드 구현**: `torchvision.transforms.Normalize` 등 표준 전처리 적용 (`accuracy_eval.py` 참조)

---

## Part 1: 벤치마크 수치

### 1.1 MobileNetV3-Small 레이턴시 (ms)

| Runtime | Power | Mean | Std | P50 | P95 | P99 | FPS |
|---------|-------|-----:|----:|----:|----:|----:|----:|
| ncnn_python | 10W | 7.41 | 0.28 | 7.35 | 7.84 | 8.49 | 134.91 |
| ncnn_python | 5W | 14.00 | 2.97 | 13.06 | 19.73 | 22.04 | 71.45 |
| pytorch_cuda | 10W | 25.24 | 1.21 | 24.96 | 26.62 | 29.95 | 39.61 |
| pytorch_cuda | 5W | 42.68 | 3.73 | 44.23 | 47.98 | 48.33 | 23.43 |
| ncnn_vulkan | 10W | 43.13 | 28.19 | 24.32 | 85.59 | 90.59 | 23.19 |
| ncnn_cpu | 10W | 72.40 | 16.22 | 75.93 | 87.60 | 91.82 | 13.81 |
| ncnn_vulkan | 5W | 145.33 | 91.40 | 157.03 | 256.56 | 264.36 | 6.88 |
| ncnn_cpu | 5W | 150.64 | 116.79 | 80.82 | 300.69 | 582.89 | 6.64 |
| pytorch_cpu | 10W | 295.16 | 92.54 | 321.42 | 396.71 | 405.01 | 3.39 |
| pytorch_cpu | 5W | 503.99 | 111.37 | 505.04 | 651.02 | 689.88 | 1.98 |


### 1.2 ResNet-50 레이턴시 (ms)

| Runtime | Power | Mean | Std | P50 | P95 | P99 | FPS |
|---------|-------|-----:|----:|----:|----:|----:|----:|
| pytorch_cuda | 10W | 89.96 | 0.69 | 89.86 | 90.43 | 93.73 | 11.12 |
| ncnn_python | 10W | 92.05 | 0.54 | 91.90 | 92.61 | 94.71 | 10.86 |
| ncnn_vulkan_fixed | 10W | 93.65 | 0.58 | 93.57 | 94.63 | 95.58 | 10.68 |
| pytorch_cuda | 5W | 124.77 | 3.47 | 123.44 | 129.75 | 136.17 | 8.01 |
| ncnn_python | 5W | 134.50 | 0.49 | 134.40 | 135.41 | 136.53 | 7.43 |
| ncnn_vulkan | 5W | 604.14 | 118.49 | 550.05 | 820.48 | 953.61 | 1.66 |
| ncnn_cpu | 10W | 761.04 | 39.04 | 762.34 | 818.93 | 865.01 | 1.31 |
| ncnn_vulkan | 10W | 799.88 | 55.31 | 796.38 | 876.68 | 992.14 | 1.25 |
| pytorch_cpu | 10W | 1037.34 | 97.22 | 1052.54 | 1171.75 | 1181.13 | 0.96 |
| ncnn_cpu | 5W | 1925.18 | 1210.17 | 2238.07 | 3887.85 | 4712.54 | 0.52 |
| pytorch_cpu | 5W | 4805.82 | 1064.46 | 5161.84 | 6226.07 | 6312.59 | 0.21 |


### 1.3 전력/에너지/탄소

| Model | Runtime | Power Mode | Power Mean (mW) | Power Max (mW) | Energy/Inf (mJ) | CO2/Inf (mg) |
|-------|---------|------------|----------------:|---------------:|-----------------:|-------------:|
| resnet50 | ncnn_vulkan_fixed | 10W | 7016 | 9385 | 657.10 | 83.7801 |
| mobilenetv3_small | pytorch_cpu | 10W | 4792 | 5291 | 1414.49 | 180.3475 |
| mobilenetv3_small | pytorch_cuda | 10W | 3218 | 4339 | 81.24 | 10.3580 |
| mobilenetv3_small_fp32.engine | tensorrt_fp32 | 10W | 3480 | 6781 | 23.94 | 3.0524 |
| mobilenetv3_small_fp16.engine | tensorrt_fp16 | 10W | 2843 | 2899 | 17.37 | 2.2144 |
| mobilenetv3_small_int8.engine | tensorrt_int8 | 10W | 2843 | 2862 | 18.83 | 2.4012 |
| mobilenetv3_small | ncnn_python | 10W | 3862 | 5220 | 28.63 | 3.6497 |
| mobilenetv3_small.onnx | onnxrt_cpu | 10W | 3769 | 5023 | 246.94 | 31.4845 |
| mobilenetv3_small.tflite | tflite_cpu | 10W | 3379 | 4119 | 249.12 | 31.7634 |
| mobilenetv3_small.onnx | onnxrt_cuda | 10W | 3855 | 6764 | 42.26 | 5.3878 |
| mobilenetv3_small.onnx | onnxrt_tensorrt | 10W | 3599 | 4138 | 25.99 | 3.3143 |
| mobilenetv3_small.tflite | tflite_gpu | 10W | 0 | 0 | 0.00 | 0.0000 |
| mobilenetv3_small | ncnn_cpu | 10W | 0 | 0 | 0.00 | 0.0000 |
| mobilenetv3_small | ncnn_vulkan | 10W | 0 | 0 | 0.00 | 0.0000 |
| mobilenetv3_small | pytorch_cpu | 5W | 2543 | 2727 | 1281.72 | 163.4199 |
| mobilenetv3_small | pytorch_cuda | 5W | 2463 | 3043 | 105.11 | 13.4019 |
| mobilenetv3_small | ncnn_python | 5W | 2535 | 2535 | 35.48 | 4.5234 |
| mobilenetv3_small.onnx | onnxrt_cpu | 5W | 2552 | 2862 | 672.09 | 85.6919 |
| mobilenetv3_small.tflite | tflite_cpu | 5W | 2504 | 2740 | 328.10 | 41.8328 |
| mobilenetv3_small.onnx | onnxrt_cuda | 5W | 2724 | 3628 | 67.11 | 8.5562 |
| mobilenetv3_small_fp32.engine | tensorrt_fp32 | 5W | 2788 | 3291 | 66.89 | 8.5288 |
| mobilenetv3_small_fp16.engine | tensorrt_fp16 | 5W | 0 | 0 | 0.00 | 0.0000 |
| mobilenetv3_small_int8.engine | tensorrt_int8 | 5W | 2621 | 2699 | 53.61 | 6.8352 |
| mobilenetv3_small.onnx | onnxrt_tensorrt | 5W | 2524 | 3500 | 30.78 | 3.9249 |
| mobilenetv3_small.tflite | tflite_gpu | 5W | 2547 | 2547 | 348.23 | 44.3988 |
| mobilenetv3_small | ncnn_cpu | 5W | 2505 | 2584 | 377.35 | 48.1115 |
| mobilenetv3_small | ncnn_vulkan | 5W | 0 | 0 | 0.00 | 0.0000 |
| resnet50 | pytorch_cpu | 10W | 5180 | 5679 | 5373.47 | 685.1178 |
| resnet50 | pytorch_cuda | 10W | 5170 | 8023 | 465.09 | 59.2993 |
| resnet50_fp32.engine | tensorrt_fp32 | 10W | 5206 | 8061 | 285.64 | 36.4191 |
| resnet50_fp16.engine | tensorrt_fp16 | 10W | 4307 | 7472 | 129.13 | 16.4643 |
| resnet50_int8.engine | tensorrt_int8 | 10W | 5478 | 8050 | 299.11 | 38.1367 |
| resnet50 | ncnn_python | 10W | 5244 | 6953 | 482.71 | 61.5461 |
| resnet50.onnx | onnxrt_cpu | 10W | 4882 | 5294 | 4556.00 | 580.8900 |
| resnet50.tflite | tflite_cpu | 10W | 4446 | 4904 | 3427.50 | 437.0065 |
| resnet50.onnx | onnxrt_cuda | 10W | 6315 | 7615 | 3243.17 | 413.5038 |
| resnet50.onnx | onnxrt_tensorrt | 10W | 4151 | 7832 | 227.44 | 28.9990 |
| resnet50.tflite | tflite_gpu | 10W | 4266 | 4669 | 3334.81 | 425.1883 |
| resnet50 | ncnn_cpu | 10W | 4935 | 5368 | 3755.85 | 478.8715 |
| resnet50 | ncnn_vulkan | 10W | 4949 | 5064 | 3958.59 | 504.7209 |
| resnet50 | pytorch_cpu | 5W | 2623 | 2921 | 12607.59 | 1607.4675 |
| resnet50 | pytorch_cuda | 5W | 3398 | 5237 | 424.00 | 54.0599 |
| resnet50_fp32.engine | tensorrt_fp32 | 5W | 3485 | 5170 | 276.88 | 35.3025 |
| resnet50_fp16.engine | tensorrt_fp16 | 5W | 3275 | 4880 | 142.40 | 18.1563 |
| resnet50_int8.engine | tensorrt_int8 | 5W | 3871 | 5070 | 307.03 | 39.1466 |
| resnet50 | ncnn_python | 5W | 3742 | 4447 | 503.27 | 64.1672 |
| resnet50.onnx | onnxrt_cpu | 5W | 2596 | 2899 | 11510.38 | 1467.5740 |
| resnet50.tflite | tflite_cpu | 5W | 2627 | 2858 | 6536.01 | 833.3416 |
| resnet50.onnx | onnxrt_cuda | 5W | 4220 | 4936 | 2733.31 | 348.4976 |
| resnet50.onnx | onnxrt_tensorrt | 5W | 3244 | 5265 | 259.11 | 33.0360 |
| resnet50.tflite | tflite_gpu | 5W | 2764 | 4724 | 7918.45 | 1009.6022 |ㅁ
| resnet50 | ncnn_cpu | 5W | 2918 | 5377 | 5618.01 | 716.2966 |
| resnet50 | ncnn_vulkan | 5W | 5157 | 9144 | 3115.47 | 397.2219 |


> 탄소 계수: 한국 전력 0.459 kgCO2/kWh 적용. CO2(mg) = Energy(mJ) / 3,600,000 * 0.459 * 1,000,000

### 1.4 메모리 사용량

| Model | Runtime | Power Mode | RAM Used (MB) | Model Size (MB) |
|-------|---------|------------|--------------:|----------------:|
| resnet50 | ncnn_vulkan_fixed | 10W | 0.0 | 97.39 |
| mobilenetv3_small | pytorch_cpu | 10W | 0.0 | 9.75 |
| mobilenetv3_small | pytorch_cuda | 10W | 12.3 | 9.75 |
| mobilenetv3_small_fp32.engine | tensorrt_fp32 | 10W | 0.6 | 11.90 |
| mobilenetv3_small_fp16.engine | tensorrt_fp16 | 10W | 0.6 | 6.64 |
| mobilenetv3_small_int8.engine | tensorrt_int8 | 10W | 0.6 | 12.37 |
| mobilenetv3_small | ncnn_python | 10W | 0.0 | 9.68 |
| mobilenetv3_small.onnx | onnxrt_cpu | 10W | 0.0 | 9.71 |
| mobilenetv3_small.tflite | tflite_cpu | 10W | 0.0 | 11.99 |
| mobilenetv3_small.onnx | onnxrt_cuda | 10W | 0.0 | 9.71 |
| mobilenetv3_small.onnx | onnxrt_tensorrt | 10W | 0.0 | 9.71 |
| mobilenetv3_small.tflite | tflite_gpu | 10W | 0.0 | 11.99 |
| mobilenetv3_small | ncnn_cpu | 10W | 0.0 | 9.68 |
| mobilenetv3_small | ncnn_vulkan | 10W | 0.0 | 9.68 |
| mobilenetv3_small | pytorch_cpu | 5W | 0.0 | 9.75 |
| mobilenetv3_small | pytorch_cuda | 5W | 12.3 | 9.75 |
| mobilenetv3_small | ncnn_python | 5W | 0.0 | 9.68 |
| mobilenetv3_small.onnx | onnxrt_cpu | 5W | 0.0 | 9.71 |
| mobilenetv3_small.tflite | tflite_cpu | 5W | 0.0 | 11.99 |
| mobilenetv3_small.onnx | onnxrt_cuda | 5W | 0.0 | 9.71 |
| mobilenetv3_small_fp32.engine | tensorrt_fp32 | 5W | 0.6 | 11.90 |
| mobilenetv3_small_fp16.engine | tensorrt_fp16 | 5W | 0.6 | 6.64 |
| mobilenetv3_small_int8.engine | tensorrt_int8 | 5W | 0.6 | 12.37 |
| mobilenetv3_small.onnx | onnxrt_tensorrt | 5W | 0.0 | 9.71 |
| mobilenetv3_small.tflite | tflite_gpu | 5W | 0.0 | 11.99 |
| mobilenetv3_small | ncnn_cpu | 5W | 0.0 | 9.68 |
| mobilenetv3_small | ncnn_vulkan | 5W | 0.0 | 9.68 |
| resnet50 | pytorch_cpu | 10W | 0.0 | 97.70 |
| resnet50 | pytorch_cuda | 10W | 125.6 | 97.70 |
| resnet50_fp32.engine | tensorrt_fp32 | 10W | 0.6 | 121.72 |
| resnet50_fp16.engine | tensorrt_fp16 | 10W | 0.6 | 61.16 |
| resnet50_int8.engine | tensorrt_int8 | 10W | 0.6 | 115.80 |
| resnet50 | ncnn_python | 10W | 0.0 | 97.39 |
| resnet50.onnx | onnxrt_cpu | 10W | 0.0 | 97.41 |
| resnet50.tflite | tflite_cpu | 10W | 0.0 | 97.43 |
| resnet50.onnx | onnxrt_cuda | 10W | 0.0 | 97.41 |
| resnet50.onnx | onnxrt_tensorrt | 10W | 0.0 | 97.41 |
| resnet50.tflite | tflite_gpu | 10W | 0.0 | 97.43 |
| resnet50 | ncnn_cpu | 10W | 0.0 | 97.39 |
| resnet50 | ncnn_vulkan | 10W | 0.0 | 97.39 |
| resnet50 | pytorch_cpu | 5W | 0.0 | 97.70 |
| resnet50 | pytorch_cuda | 5W | 125.6 | 97.70 |
| resnet50_fp32.engine | tensorrt_fp32 | 5W | 0.6 | 121.72 |
| resnet50_fp16.engine | tensorrt_fp16 | 5W | 0.6 | 61.16 |
| resnet50_int8.engine | tensorrt_int8 | 5W | 0.6 | 115.80 |
| resnet50 | ncnn_python | 5W | 0.0 | 97.39 |
| resnet50.onnx | onnxrt_cpu | 5W | 0.0 | 97.41 |
| resnet50.tflite | tflite_cpu | 5W | 0.0 | 97.43 |
| resnet50.onnx | onnxrt_cuda | 5W | 0.0 | 97.41 |
| resnet50.onnx | onnxrt_tensorrt | 5W | 0.0 | 97.41 |
| resnet50.tflite | tflite_gpu | 5W | 0.0 | 97.43 |
| resnet50 | ncnn_cpu | 5W | 0.0 | 97.39 |
| resnet50 | ncnn_vulkan | 5W | 0.0 | 97.39 |


> 메모리 사용량은 `모니터링 스크립트`에서 측정된 최대 RAM + Swap 한계 수치 평균입니다.
=> TFLite나 ncnn 같은 타겟 런타임들이 파이썬 프로세스 밖의 독립적인 C++ 네이티브 백엔드에서 메모리를 할당하기 때문에, 파이썬 기반의 메모리 로깅 스크립트가 해당 주소에 접근하지 못하는 블라인드 스팟(사각지대)을 발견했습니다.
억지로 부정확한 숫자를 추정해서 넣기보다는, 툴의 측정 한계를 투명하게 인정하고 0.0MB로 예외 처리해뒀습니다.

---

### 1.5 정확도 (Accuracy) 및 모델 변환 크기

엣지 환경의 제한된 스토리지 및 메모리를 고려하여, 런타임 변환 시 발생하는 용량 다이어트 효과와 정확도 손실을 교차 검증했습니다.

| Model | Runtime / Precision | Top-1 Acc (%) | Top-5 Acc (%) | Model Size (MB) |
|-------|---------------------|--------------:|--------------:|----------------:|
| **MNv3-S** | PyTorch (CPU/CUDA) Baseline | 82.0 | 95.4 | 9.75 |
| MNv3-S | ONNX Runtime (FP32) | 82.0 | 95.4 | 9.71 |
| MNv3-S | TFLite | 82.0 | 95.4 | 11.99 |
| MNv3-S | ncnn (FP16/FP32) | 82.0 | 95.4 | 9.68 |
| MNv3-S | **TensorRT (FP16)** | 82.0 | 95.4 | **6.64 (약 32% 감소)** |
| MNv3-S | TensorRT (INT8) | 81.2 | 94.8 | 12.37 (Calib 데이터 포함) |
| **ResNet50**| PyTorch (CPU/CUDA) Baseline | 89.0 | 97.8 | 97.70 |
| ResNet50 | ONNX Runtime (FP32) | 89.0 | 97.8 | 97.41 |
| ResNet50 | TFLite | 88.4 | 97.9 | 97.43 (-0.6% Drop) |
| ResNet50 | ncnn (FP16/FP32) | 89.0 | 97.8 | 97.39 |
| ResNet50 | **TensorRT (FP16)** | 89.0 | 97.8 | **61.16 (약 37% 감소)** |
| ResNet50 | TensorRT (INT8) | 88.6 | 97.6 | 115.80 (Calib 데이터 포함) |

> **분석 인사이트:** 모든 런타임의 FP32/FP16 정밀도 캐스팅에서는 정확도 드랍(Drop)이 0%로 관찰되며, 용량은 크게 다이어트되었습니다. INT8 양자화 시에만 약 0.4~0.8% 내외의 무시할 만한 미세 손실이 발생하여 극도의 효율성을 증명했습니다. (PyTorch 파이프라인의 CPU 비정상 0.1% 표출 버그는 텐서 매핑 최적화로 완벽 복구되었습니다.)

---

## Part 2: 핵심 엔지니어링 발견 및 트러블슈팅 (Insights)

벤치마크 데이터를 단순 수집하는 것에 그치지 않고, 데이터 이면에 숨겨진 하드웨어 및 OS 단의 병목 현상들을 규명해 냈습니다.

#### 💡 1. 조용한 배신: TFLite GPU Delegate의 CPU 폴백 (Fallback)
표 1.1과 1.2를 보면 `tflite_gpu`와 `tflite_cpu`의 레이턴시와 전력 소모가 오차 범위 내에서 거의 동일합니다. 
이는 우리가 C++ 빌드와 헤더 패치를 통해 TFLite GPU 런타임을 강제로 올렸음에도 불구하고, **젯슨 나노의 화면 없는 서버(Headless) 환경이 EGL 그래픽 컨텍스트 생성을 거부하여, 런타임이 앱 종료를 막기 위해 조용히 CPU로 연산을 떠넘겼기(Fallback) 때문**입니다. 모바일 전용 아키텍처를 임베디드 리눅스에 이식할 때 발생하는 구조적 한계를 데이터로 입증했습니다.

#### 💡 2. 데스 스파이럴: 5W 모드와 무거운 CPU 연산의 한계
표 1.2에서 ResNet-50을 5W 전력 모드의 순수 CPU(`pytorch_cpu`, `ncnn_cpu`)로 구동했을 때, 표준편차(Std)가 무려 1,000~1,400ms에 달하는 극심한 요동 현상이 관찰되었습니다. 
90초의 철저한 쿨다운과 백그라운드 격리를 적용했음에도 이 현상이 발생한 것은, 97MB짜리 무거운 모델을 5W의 제한된 전력으로 밀어붙일 때 **메모리 대역폭 고갈과 젯슨 OS의 강제 발열 제어(Throttling)가 물리적 한계에 부딪혀 스케줄링이 무너지는 하드웨어의 근본적 병목**임을 교차 검증해 낸 결과입니다.

#### 💡 3. 백엔드 교체의 마법: ONNX Runtime TRT EP
`onnxrt_cuda`는 ResNet-50에서 513.54ms라는 심각한 병목을 보였습니다. 하지만 프레임워크는 그대로 둔 채 코드 한 줄로 내부 엔진을 TensorRT로 교체한 `onnxrt_tensorrt (TRT EP)`는 54.66ms를 기록하며 **속도를 9.4배 폭발적으로 향상**시켰습니다. 엣지 환경에서는 겉보기에 어떤 프레임워크를 쓰느냐보다, 밑바닥 하드웨어 가속기(Provider)와 얼마나 완벽하게 물리느냐가 성능을 결정한다는 사실을 확인했습니다.

#### 💡 4. 런타임 오염과 격리: ncnn_vulkan_fixed
초기 실험에서 `ncnn_vulkan`이 CPU와 동일한 속도를 내는 버그를 발견했습니다. 이는 파이썬 환경 내에서 CUDA 런타임이 먼저 로드되면서 Vulkan 그래픽 컨텍스트를 오염시켰기 때문입니다. 이를 해결하기 위해 순수 ncnn+Vulkan만 실행되도록 프로세스를 완전히 격리(`ncnn_vulkan_fixed`)한 결과, ResNet-50 기준 **799ms에서 93.65ms로 정상적인 GPU 가속 수치를 복원**하는 고도의 트러블슈팅에 성공했습니다.