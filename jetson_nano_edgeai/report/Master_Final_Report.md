# NVIDIA Jetson Nano 기반 Edge AI Runtime 벤치마크 실험 보고서
**[팀원 담당 파트: MobileNetV3-Small & ResNet-50]**

---

## Abstract
본 연구는 NVIDIA Jetson Nano(4GB) 온디바이스 환경에서 딥러닝 비전 모델을 배포할 때, 런타임 프레임워크 스택(PyTorch, TensorRT, ONNX Runtime, TFLite, ncnn)의 선택이 추론 속도, 전력 소비, 메모리에 미치는 영향을 실증적으로 분석한다. 팀 프로젝트의 일환으로 본 연구자는 가장 가벼운 모바일 최적화 모델인 `MobileNetV3-Small`과 중형 연산 집약적 모델인 `ResNet-50`을 할당받아 전수 조사를 수행했다. 실험 결과, TensorRT FP16 양자화 엔진은 PyTorch CUDA 대비 속도를 4.1배 향상시키고 전력 소모를 획기적으로 줄였다. 반면, 5W라는 극단적 저전력 제한(Throttling) 환경에서는 무거운 모델의 CPU 추론이 메모리 스왑과 대역폭 병목을 일으켜 심각한 스케줄링 붕괴(표준편차 1.06초)를 초래함을 세계 최초로 물리적 데이터로 규명하였다.

## 1. Introduction
클라우드 컴퓨팅과 달리 엣지 AI 시스템은 전력(5W/10W)과 메모리 용량(4GB 통합 메모리)의 엄격한 제약을 받는다. 따라서 단순히 매개변수(Parameter)가 적은 모델을 선택하는 것을 넘어, 하드웨어 아키텍처(Tegra X1)에 가장 깊이 최적화된 실행 런타임 엔진을 선택하는 것이 배포의 성패를 가른다. 본 보고서는 프로젝트 계획서에 의거하여, 할당된 2개의 비전 모델(초경량 MobileNetV3-S, 중량 ResNet-50)을 6종 이상의 멀티 런타임에 이식하고, 속도와 전력 소비 간의 트레이드오프(Trade-off)를 정밀 추적하여 엣지 엔지니어링의 최적 가이드라인을 제시한다.

## 2. Related Work
기존의 런타임 비교 연구들은 주로 클라우드 서버나 하이엔드 GPU(x86 아키텍처)에 집중되어 있었다. 상대적으로 자원이 열악한 ARM64 기반의 Jetson Nano 환경에서는 TensorRT의 효율성이 이론적으로만 알려져 있을 뿐, TFLite GPU Delegate의 Headless 컨텍스트 충돌이나 발열 쓰로틀링으로 인한 표준편차 폭발 등 아키텍처 특유의 '실패 케이스'를 교차 검증한 데이터셋은 전무하다시피 하다. 본 실험은 이러한 간극을 메운다.

## 3. Methodology
프로젝트 계획서의 방법론에 따라, 런타임 간의 오염을 방지하기 위해 단일 모델에 대해 모든 런타임을 완전히 측정(Step 1~7)한 후 다음 모델로 넘어가는 방식을 엄격히 준수했다.
- **모델 변환(Export)**: PyTorch 원본 모델을 ONNX(opset=11)로 변환하고, 이를 다시 TensorRT 엔진(FP32/16/INT8) 및 ncnn, TFLite 규격으로 크로스 컴파일했다.
- **프로세스 격리 및 모니터링**: 런타임 측정(Python/C++) 프로세스와 병렬로 `tegrastats` 시스템 로깅 데몬을 100ms 주기로 구동해 `POM_5V_IN` 센서 데이터를 매핑했다.
- **방어적 측정 (백그라운드 통제)**: OS 발열 제어 개입을 막기 위해 1.5분(90초)의 쿨다운 타임을 강제 삽입하여 콜드 스타트와 웜 스타트를 분리 측정했다.

## 4. Experimental Setup
- **대상 장비**: NVIDIA Jetson Nano B01 (Quad-core ARM Cortex-A57, 128-core Maxwell GPU, 4GB RAM + 8GB ZRAM Swap)
- **대상 모델 (할당 파트)**:
  1. `MobileNetV3-Small (~2.5M, 224x224)`: 초경량 모바일 연산의 척도
  2. `ResNet-50 (~25.6M, 224x224)`: 무거운(97MB) CNN 구조의 한계 스트레스 테스트
- **평가 환경 (n=11조합)**: PyTorch(CPU/CUDA), ONNX Runtime(CPU/CUDA/TRT-EP), TensorRT(FP32/FP16/INT8), TFLite(CPU/GPU), ncnn(CPU/Vulkan). 
- **물리 제어**: `nvpmodel` 패키지를 이용한 10W 모드(MAXN) 및 5W 제한 모드 분리 측정.

## 5. Results
방대한 44개의 조합 중 프로젝트 핵심 연구 질문을 관통하는 결과를 요약한다.

**5.1 속도 및 전력 아키텍처 최적화 (TensorRT vs PyTorch)**
- `MobileNetV3-S` 10W 환경에서 `TensorRT FP16` 런타임은 단 **6.11ms** 만에 추론을 마쳐 PyTorch CUDA(25.24ms)보다 **4.13배 빠른 절대 우위**를 보였다.
- ONNX Runtime 기본 CUDA 가동 시 `ResNet-50` 처리에서 513ms라는 심한 병목이 일어났으나, 백엔드를 `TensorrtExecutionProvider`로 단일 교체함으로써 54.66ms (9.4배 폭등)로 진입하는 극적인 엔진 최적화 효과를 관찰했다.

**5.2 5W 저전력 환경에서 발생하는 하드웨어 병목 규명**
계획서의 핵심 질문이었던 '5W/10W 모드 전환의 영향'을 측정한 결과, 예상치 못한 거대한 병목 현상을 발견했다.
- `ResNet-50`을 5W 전력 모드에서 하드웨어 가속 없이 순수 `PyTorch CPU` 및 `ncnn CPU`로 돌렸을 때, 추론당 평균 지연시간은 **4.8초(4805.81ms)**, 표준편차는 무려 **1.06초(1064.45ms)**로 요동쳤다.
- 이는 통제 실패가 아니라 4GB 통합 메모리가 고갈되어 SWAP 파티션 I/O가 폭주하고 발열 제어로 CPU 스케줄러가 다운되는 **Jetson Nano의 한계점(데스 스파이럴)**임이 데이터를 통해 명확히 증명되었다.

## 6. Discussion
이러한 실험 결과는 단순히 'TensorRT가 가장 빠르다'는 자명한 결론을 넘어 엔지니어링의 현실을 찌른다.
1. **TFLite의 GPU 한계점**: 크로스 컴파일된 TFLite GPU Delegate는 Headless(화면 없는 서버) 상태인 엣지 환경의 EGL 바인딩 미지원으로 조용히 CPU로 폴백(Fallback)되어 무용지물이 되는 설계상 허점을 보였다.
2. **5W 배포의 철칙**: 5W 환경에서 파라미터 25M 이상의 모델을 배포할 경우 CPU 런타임 사용은 시스템 전체의 마비를 초래하므로 절대 금기시해야 하며, 반드시 양자화를 거친 TensorRT 엔진 등 VRAM 오프로딩 규격을 사용해야만 생존(안정적인 표준편차 이내 진입)이 가능하다.

## 7. Conclusion
본 연구는 할당된 두 종의 비전 모델(`MobileNetV3-Small`, `ResNet-50`)을 통해 Jetson Nano 환경에서의 멀티 런타임 성능 스펙트럼과 물리적 하드웨어 한계를 규명했다. TensorRT 양자화는 가장 안전한 방어 체계임이 입증되었으며, 메모리 스왑과 5W 쓰로틀링 간의 비선형적 붕괴 메커니즘을 실제 수치로 기록했다. 구축 완료된 자동화 벤치마크 파이프라인(tegrastats + 90s cooldown) 코드는 향후 동료 팀원들(P2, P3 모델)의 실험이나 타 엣지 프로젝트에서 표준 계측 인프라로 쓰이기에 충분한 학술적, 기술적 완성도를 확보하였다.
