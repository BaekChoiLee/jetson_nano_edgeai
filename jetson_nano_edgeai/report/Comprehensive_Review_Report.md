# Jetson Nano 벤치마크 프로젝트 종합 검토 보고서

> **검토 일시**: 2026-04-03
> **검토 방법**: 로컬 데이터 + Jetson Nano SSH 직접 접속 교차검증 (`jetson@192.168.1.133`)
> **데이터 소스**: `all_summaries_latest.json` (44개 실험), `report/core_reference.md`, `results/*.csv`, 소스코드

---

## 1. 프로젝트 폴더 구조

```
jetson-benchmark/
├── benchmark/                    ← 핵심 측정 코드 (13개 파일)
│   ├── benchmark.py                  메인 오케스트레이터
│   ├── accuracy_eval.py              정확도 평가 (PyTorch/ONNX/TFLite/TRT/ncnn)
│   ├── tegra_parser.py               tegrastats 파서
│   ├── layer_analyzer.py             레이어별 분석
│   ├── run_pytorch.py                PyTorch CPU/CUDA 벤치마크
│   ├── run_tensorrt.py               TensorRT 벤치마크
│   ├── run_onnxrt.py                 ONNX Runtime 벤치마크
│   ├── run_ort_trt_ep.py             ONNX TRT EP 벤치마크
│   ├── run_tflite.py                 TFLite 벤치마크
│   ├── run_ncnn.py                   ncnn 벤치마크
│   ├── calc_carbon.py                탄소 배출량 계산
│   └── collect_memory.py             메모리 수집
├── report/                       ← 보고서 (5개 파일)
│   ├── core_reference.md             최신 순도100% 데이터 문서 (3/30)
│   ├── Master_Final_Report.md        학술 보고서 형식
│   ├── Week8_Final_Report_New.md     8주차 종합 보고서
│   ├── Detailed_Experimental_Methodology.md  방법론 상세
│   └── report.md                     초기 보고서
├── results/                      ← 정확도 CSV
│   ├── mobilenetv3_small_accuracy_complete.csv  (n=1000, 7개 런타임)
│   └── resnet50_accuracy_complete.csv           (n=1000, 7개 런타임)
├── charts/                       ← 시각화 차트 (6종 × PDF+PNG = 12파일)
│   ├── 01_latency_heatmap.{png,pdf}
│   ├── 02_power_speed_scatter.{png,pdf}
│   ├── 03_layer_waterfall_mobilenetv3_small.{png,pdf}
│   ├── 03_layer_waterfall_resnet50.{png,pdf}
│   ├── 04_carbon_emissions.{png,pdf}
│   ├── 05_trt_quantization.{png,pdf}
│   └── 06_5w_vs_10w.{png,pdf}
├── models/                       ← 변환된 모델 파일
│   ├── *.onnx, *.tflite, *_saved_model/
├── data/imagenet_val/            ← ImageNet 1K 검증 데이터셋 (1000 클래스 × 1장)
├── convert/                      ← 모델 변환 도구 (4개)
├── setup/                        ← 환경 설정 스크립트 (4개)
├── analysis/                     ← 분석 스크립트 (2개)
├── run_all.sh                    ← 전체 자동화 파이프라인 (297줄)
├── all_summaries_latest.json     ← 44개 실험 조합 원시 데이터
├── jetson_nano_project_plan.docx.pdf  ← 프로젝트 계획서
├── project_plan_extracted.txt    ← 계획서 텍스트 추출본
└── mobilenetv3_small_accuracy.csv ← ⚠️ 구버전 버그 데이터 (n=10, 삭제 대상)
```

**평가**: 프로젝트 구조가 체계적. 계획서 산출물(benchmark.py, tegra_parser.py, layer_analyzer.py, CSV)이 모두 구현됨. `charts/` 폴더에 6종 차트가 독립적으로 존재하여 .gemini 경로 의존 문제도 이미 해결됨.

---

## 2. 프로젝트 계획서 대비 달성 현황

### 2.1 달성 매트릭스

| 계획서 항목 | 상태 | 상세 |
|-----------|:---:|------|
| **런타임 4-6종** | ✅ 초과달성 | 11종 조합 (PyTorch CPU/CUDA, TRT FP32/FP16/INT8, ONNX RT CPU/CUDA/TRT-EP, TFLite CPU/GPU, ncnn CPU/Vulkan) |
| **Warm-up 10회 + 측정 100회** | ✅ | `benchmark.py:123-124` — `--num-warmup=10, --num-runs=100` |
| **tegrastats 병렬 로깅 100ms** | ✅ | `benchmark.py:169` — `interval_ms=100` |
| **5W / 10W 듀얼 모드** | ✅ | `run_all.sh:213-217` — `nvpmodel -m 0/1` |
| **ImageNet 정확도 검증** | ✅ | 1,000장 (1000 클래스 × 1장) Top-1/Top-5 |
| **레이어별 latency 분석** | ✅ | `layer_analyzer.py` 실행 완료. 젯슨 `results/`에 `mobilenetv3_small_layers.csv` (52개 레이어), `resnet50_layers.csv` (53개 레이어) + PNG 시각화 존재 |
| **탄소 배출량 환산** | ✅ 방식변경 | 계획: codecarbon → 실제: 수동 계산 (한국 전력계수 0.459 kgCO2/kWh) |
| **쿨다운 타임** | ✅ 초과 | 계획 60초 → 실제 `run_all.sh:11` COOL_DOWN=60, `core_reference.md`에서 90초 적용 언급 |
| **프로세스 격리** | ✅ | `ncnn_vulkan_fixed` 사례에서 격리 효과 입증 |
| **모델 2개 (P1 담당분)** | ✅ | MobileNetV3-Small + ResNet-50 완료 |
| **나머지 4개 모델** | ❌ 미완 | P2(EfficientNet-B0, YOLOv8n), P3(ShuffleNetV2, SSD-MobileNetV2) 팀원 담당 |
| **시각화 차트** | ✅ | `charts/` 폴더에 6종 × 2포맷(PNG+PDF) = 12파일 독립 존재 |
| **GitHub 코드 정리** | ❌ 미완 | 브랜치 푸시 미완료 (카톡에서 논의 중) |

### 2.2 Step 1~7 실험 절차 준수 여부

| Step | 내용 | 준수 | 근거 |
|------|------|:---:|------|
| Step 1 | 모델 준비 (PyTorch 원본 + 정확도) | ✅ | `accuracy_eval.py` — pretrained=True 로드 |
| Step 2 | 모델 변환 (ONNX→TRT/TFLite/ncnn) | ✅ | `run_all.sh:141-190` Step 1 자동 변환 |
| Step 3 | 런타임별 추론 순차 실행 | ✅ | `benchmark.py:20-32` ALL_RUNTIMES 11종 |
| Step 4 | tegrastats 병렬 로깅 | ⚠️ | 구현됨. 일부 런타임에서 수집 실패 (0 또는 빈 값) |
| Step 5 | 레이어 분석 | ✅ | 젯슨 SSH 확인: `*_layers.csv` + `*_layers.png` 존재 (MNv3-S 52레이어, ResNet-50 53레이어) |
| Step 6 | 5W 모드 재측정 | ✅ | JSON에 5W 데이터 22개 실험 존재 |
| Step 7 | 결과 CSV/JSON 저장 | ✅ | all_summaries_latest.json + results/*.csv |

---

## 3. 할루시네이션(데이터 조작) 검증

### 3.1 교차검증: JSON 원시 데이터 vs 보고서 수치

`all_summaries_latest.json`의 원시 수치와 `core_reference.md` 테이블 값을 소수점 단위까지 대조했다.

| 항목 | JSON 원시값 | core_reference.md | 일치 |
|------|-----------|-------------------|:---:|
| MNv3-S ncnn_python 10W mean | 7.41247518... | 7.41ms | ✅ (반올림 일치) |
| MNv3-S pytorch_cuda 10W mean | 25.24389163... | 25.24ms | ✅ |
| MNv3-S tensorrt_fp16 10W mean | 6.10943173... | 6.11ms, FPS 163.68 | ✅ (JSON fps=163.68) |
| MNv3-S pytorch_cpu 10W mean | 295.16208986... | 295.16ms | ✅ |
| MNv3-S tensorrt_fp32 10W mean | 6.87934216... | — (core에 TRT 별도 섹션) | ✅ |
| ResNet50 pytorch_cuda 10W mean | 89.96241783... | 89.96ms | ✅ |
| ResNet50 onnxrt_cuda 10W mean | 513.53883598... | core=513.54ms | ✅ |
| ResNet50 tensorrt_fp16 10W mean | 29.98514739... | 29.99ms | ✅ |
| ResNet50 pytorch_cpu 5W mean | 4805.81993549... | 4805.82ms | ✅ |
| ResNet50 ncnn_vulkan_fixed 10W mean | 93.65053399... | 93.65ms | ✅ |

**검증 결과**: 10개 표본 전수 일치. 소수점 2자리 반올림 범위 내에서 JSON↔보고서 수치가 정확히 대응한다.

### 3.2 보고서 간 수치 불일치 (Week8 vs core_reference vs report.md)

| 항목 | core_reference | report.md | Week8 | JSON 원시값 |
|------|:---:|:---:|:---:|:---:|
| ResNet50 ONNX CUDA 10W | 513.54ms | 513.54ms | **515.05ms** | 513.539ms |

**판정**: Week8 보고서의 515.05ms는 JSON과 1.5ms 차이. 다른 측정 세션의 결과로 추정된다. core_reference.md와 report.md는 JSON과 정확히 일치하므로, **core_reference.md를 정본(canonical source)으로 사용해야 한다**.

### 3.3 정확도 CSV 교차검증

`results/*.csv`와 `core_reference.md` 1.5절 대조:

| 모델 | 런타임 | CSV (n=1000) | core_reference | 일치 |
|------|--------|:---:|:---:|:---:|
| MNv3-S | PyTorch CPU | 82.0 / 95.4 | 82.0 / 95.4 | ✅ |
| MNv3-S | PyTorch CUDA | 82.0 / 95.4 | — (동일) | ✅ |
| MNv3-S | TRT FP16 | 82.0 / 95.4 | 82.0 / 95.4 | ✅ |
| MNv3-S | TRT INT8 | 81.2 / 94.8 | 81.2 / 94.8 | ✅ |
| MNv3-S | ncnn | 82.0 / 95.4 | 82.0 / 95.4 | ✅ |
| ResNet50 | PyTorch CPU | 89.0 / 97.8 | 89.0 / 97.8 | ✅ |
| ResNet50 | TFLite | 88.4 / 97.9 | 88.4 / 97.9 | ✅ |
| ResNet50 | TRT INT8 | 88.6 / 97.6 | 88.6 / 97.6 | ✅ |

**검증 결과**: 정확도 CSV 16개 수치 전수 일치. ��이터 조작 흔적 없음.

### 3.4 젯슨 나노 원본 vs 로컬 정확도 교차검증 (SSH 직접 확인)

젯슨 나노의 `results/` 디렉토리에 있는 **원본 정확도 CSV**와 로컬 **수정본 CSV**를 대조했다.

| 모델 | 런타임 | 젯슨 원본 (버그) | 로컬 수정본 (`_complete`) | 비고 |
|------|--------|:---:|:---:|------|
| MNv3-S | pytorch_cpu | **0.1% / 0.5%** | 82.0% / 95.4% | nn.Hardswish 버그 → mkldnn.enabled=False로 수정 |
| MNv3-S | pytorch_cuda | 82.0% / 95.4% | 82.0% / 95.4% | ✅ 일치 |
| ResNet50 | pytorch_cpu | **0.1% / 0.5%** | 89.0% / 97.8% | 동일 버그 |
| ResNet50 | pytorch_cuda | 89.0% / 97.8% | 89.0% / 97.8% | ✅ 일치 |

**판정**: 젯슨 원본에서 `pytorch_cpu`가 0.1%인 것은 `accuracy_eval.py:89-99` 주석에 기록된 **PyTorch 1.10 ARM aarch64의 nn.Hardswish 플랫폼 버그**의 명확한 증거다. CUDA 및 다른 런타임은 젯슨↔로컬 정확히 일치하므로, 로컬의 `_complete.csv`는 버그 수정 후 재측정한 정당한 데이터임이 확인된다.

**추가 발견**: 젯슨의 `results/`에 40개 이상의 실험 디렉토리가 존재하여 (예: `mobilenetv3_small_10w_20260327_191002` ~ `20260329_174811`), 3일간(3/27~3/29) 반복 실험을 수행한 흔적이 확인됨. 이는 데이터의 실측 신뢰도를 높인다.

---

## 4. RED FLAGS (주의 사항)

### RED FLAG 1: 정확도 수치가 공식 벤치마크보다 상당히 높음

| 모델 | torchvision 공식 Top-1 | 측정값 Top-1 | 차이 |
|------|:--------------------:|:----------:|:---:|
| MobileNetV3-Small | 67.7% (V1) / 74.0% (V2) | **82.0%** | +8~14%p |
| ResNet-50 | 76.1% (V1) / 80.9% (V2) | **89.0%** | +8~13%p |

**원인 분석** (`accuracy_eval.py:29-59`):

```python
def load_imagenet_val(data_dir, max_images=500):
    classes = sorted(os.listdir(val_dir))
    for cls in classes:
        for fname in sorted(os.listdir(cls_dir)):  # ← 정렬된 첫 번째 이미지만
            if count >= max_images: break
            ...
```

데이터셋 확인 결과 **1000개 클래스 폴더에 각각 정확히 1장**만 존재. 즉 전체 50,000장 중 2%만 사용한 서브셋이다. 각 클래스에서 "첫 번째 이미지"는 해당 클래스를 대표하는 전형적(prototypical) 이미지일 가능성이 높아 **표본 편향(sampling bias)**이 발생한다.

**판정**: 데이터 조작(할루시네이션) **아님**. 실제 측정값이지만 **표본 편향으로 인한 과대 측정**. 보고서에 "1,000-image stratified subset (클래스당 1장)" 임을 명시해야 한다. core_reference.md 7행에 이미 명시되어 있음 — 양호.

### RED FLAG 2: 구버전 정확도 파일 잔존

**파일**: `mobilenetv3_small_accuracy.csv` (프로젝트 루트)

```csv
runtime,top1,top5,n
pytorch_cpu,10.0,50.0,10    ← PyTorch 1.10 ARM nn.Hardswish 버그 흔적
pytorch_cuda,70.0,90.0,10
onnxrt_cuda,70.0,90.0,10
tflite,70.0,90.0,10
```

n=10, pytorch_cpu Top-1=**10.0%** — 이것은 `accuracy_eval.py:89-99` 주석에 설명된 "SafeHardswish Hook 버그" 수정 전의 잔존 파일이다. 정상 파일은 `results/mobilenetv3_small_accuracy_complete.csv` (n=1000, 82.0%).

**조치 필요**: 이 파일을 삭제하거나 `archive/` 폴더로 이동해야 혼동을 방지할 수 있다.

### RED FLAG 3: 일부 전력 데이터 누락 (빈 문자열 또는 0)

JSON에서 `power_avg_mw`가 빈 문자열(`""`)인 런타임:

| 런타임 | 전력모드 | power_avg_mw | 원인 |
|--------|:------:|:-----------:|------|
| tflite_gpu | 10W | "" | tegrastats 로깅 미수집 |
| ncnn_cpu | 10W | "" | 동일 |
| ncnn_vulkan | 10W | "" | 동일 |
| tensorrt_fp16 | 5W | "" | 동일 |
| ncnn_vulkan | 5W | "" | 동일 |

`benchmark.py:168-172`에서 tegra_logger 시작 실패 시 None으로 폴백하는 로직 확인. 데이터 조작이 아닌 **로깅 실패**. core_reference.md에서도 해당 셀을 0으로 표기하며 투명하게 처리함.

### RED FLAG 4: ResNet-50 INT8이 FP32보다 느리지 않지만 거의 동일

| 정밀도 | 10W Mean | 5W Mean |
|--------|--------:|--------:|
| TRT FP32 | 54.87ms | 79.45ms |
| TRT INT8 | 54.61ms | 79.31ms |
| TRT FP16 | 29.99ms | 43.48ms |

일반적으로 INT8 >> FP32 속도가 기대되나, Jetson Nano의 **Maxwell GPU는 INT8 전용 Tensor Core가 없다**. INT8 연산이 FP32로 디-퀀타이즈/리-퀀타이즈되면서 오버헤드가 상쇄되어 비슷한 속도가 나오는 것은 하드웨어 제약상 합리적이다. FP16은 Maxwell이 네이티브 지원하므로 확실한 이점이 있다.

### RED FLAG 5: Week8 보고서 내 이미지 경로 문제

`Week8_Final_Report_New.md`에서 차트 참조가 `.gemini` 경로를 사용:
```markdown
![Latency Heatmap](C:/Users/baikj/.gemini/antigravity/brain/aa9362d6-.../01_latency_heatmap.png)
```

이 경로는 Gemini AI 도구의 임시 캐시 디렉토리로, **다른 환경이나 팀원과 공유 시 깨진다**. `charts/` 폴더에 동일 파일이 이미 존재하므로 상대 경로로 교체해야 한다.

---

## 5. 할루시네이션 종합 판정

| 검증 항목 | 판정 | 근거 |
|----------|:---:|------|
| 레이턴시 데이터 | ✅ 신뢰 | JSON↔보고서 10개 표본 전수 일치. 통계 분포(std, p50/p95/p99)가 자연스러움 |
| 전력 데이터 | ⚠️ 부분 신뢰 | 5개 조합에서 누락(빈 값). 수집된 값은 2500~7000mW 범위로 합리적 |
| 정확도 데이터 | ⚠️ 편향 주의 | CSV↔보고서 전수 일치. 실측이지만 1000-image subset 편향으로 과대 측정 |
| 탄소 배출량 | ✅ 신뢰 | 공식 기반 계산 (Power × Latency × 한국 전력계수 0.459) |
| 메모리 데이터 | ⚠️ 한계 인정 | 0.0MB는 C++ 백엔드 측정 사각지대. core_reference.md에서 투명 고지 |

**결론**: 데이터는 실제 Jetson Nano에서 측정된 것으로 판단되며, AI가 지어낸 할루시네이션은 아니다. JSON 원시 데이터→보고서→CSV 전 경로에서 수치 일관성이 확인됨. 다만 **정확도 Top-1 수치는 표본 편향**이 있으므로 "ImageNet full validation set 기준"으로 해석하면 안 된다.

---

## 6. 주요 성과 요약

1. **TensorRT FP16 압도적 1위**: MNv3-S 6.11ms (163 FPS), ResNet-50 29.99ms (33 FPS)
2. **ONNX TRT EP 발견**: CUDA EP 513.54ms → TRT EP 54.79ms (9.4배 개선)
3. **TFLite GPU Fallback 규명**: Headless EGL 문제 → CPU 폴백 현상 데이터로 증명
4. **5W Death Spiral 규명**: ResNet-50 5W CPU에서 Std 1,064ms 극심한 요동
5. **ncnn Vulkan 격리 성공**: 800ms → 94ms로 8.4배 성능 복원
6. **PyTorch 1.10 ARM 버그 발견 및 우회**: nn.Hardswish 0.1% 정확도 문제 → mkldnn.enabled=False

---

## 7. 카톡 내용 분석 + 벤치마크 방법 설명

### 7.1 상황 파악

> 최호범: "벤치마크, 측정 프레임워크 만드는 부분인데 서로 코드가 다를거 같아서 다른분들 코드 보면서 더 좋은부분 있으면 수정하면서 진행"

- 팀 3명이 각자 벤치마크 측정 프레임워크를 개발 중
- 최호범이 코드 공유를 위해 **Git 브랜치 푸시** 요청
- 서로 다른 접근방식의 코드를 리뷰하며 **최선의 코드를 선별/통합**하려는 의도

### 7.2 현재 벤치마크 구현 방식 (팀에 공유할 핵심)

**자동화 파이프라인 아키텍처**:
```
run_all.sh (마스터 스크립트, 297줄)
  ├── Step 0: verify_env.py (환경 검증)
  ├── Step 1: export_onnx.py → build_trt.sh → convert_tflite.py → convert_ncnn.sh
  ├── Step 2: benchmark.py (11종 런타임 순차 벤치마크)
  │           ├── run_pytorch.py (CPU/CUDA)
  │           ├── run_tensorrt.py (FP32/FP16/INT8)
  │           ├── run_onnxrt.py + run_ort_trt_ep.py (CUDA EP / TRT EP)
  │           ├── run_tflite.py (CPU/GPU)
  │           └── run_ncnn.py (CPU/Vulkan)
  ├── Step 3: accuracy_eval.py (정확도 평가)
  └── Step 4: layer_analyzer.py (레이어 분석)
```

**측정 프로토콜**:
- 발열 제어: 런타임 간 60~90초 쿨다운 강제 삽입
- 워밍업: 10회 더미 추론 (캐시 히트 안정화)
- 본 측정: 100회 반복, `time.perf_counter()` 사용
- 통계 집계: Mean, Std, P50, P95, P99, FPS 자동 산출
- 전력 모니터링: tegrastats 100ms 간격 병렬 로깅 (`POM_5V_IN` 센서)
- 전력 모드 분리: `nvpmodel -m 0` (10W) / `-m 1` (5W)

**팀원에게 추천할 사항**:
- `run_all.sh`를 공유 브랜치에 푸시하면 P2/P3 모델 담당자가 `--model` 인자만 바꿔서 사용 가능
- `benchmark.py`의 `ALL_RUNTIMES` 리스트에 모델 추가만 하면 됨
- `accuracy_eval.py`는 `MODEL_FACTORIES`에 모델 팩토리만 등록하면 확장 가능
- `run_all.sh --week 4`나 `--week 6`으로 팀원별 모델 세트 자동 선택 가능

---

## 8. 즉시 조치 필요 사항

### 우선순위 HIGH

| # | 항목 | 상세 | 난이도 |
|---|------|------|:---:|
| 1 | **Git 브랜치 생성 & 푸시** | 최호범 요청에 따라 현재 코드를 브랜치에 올려야 함 | 쉬움 |
| 2 | **Week8 보고서 이미지 경로 수정** | `.gemini/` → `../charts/` 상대 경로로 5곳 교체 | 쉬움 |
| 3 | **구버전 파일 정리** | 루트의 `mobilenetv3_small_accuracy.csv` (n=10, 버그 데이터) 삭제 | 쉬움 |

### 우선순위 MEDIUM

| # | 항목 | 상세 | 난이도 |
|---|------|------|:---:|
| 4 | **누락된 전력 데이터 재측정** | tflite_gpu, ncnn_cpu, ncnn_vulkan의 10W + tensorrt_fp16/ncnn_vulkan의 5W | 중간 (젯슨 접속 필요) |
| 5 | **정확도 해석 주의 문구 보강** | Week8 보고서에 "서브셋 편향" 경고 추가 | 쉬움 |

### 우선순위 LOW

| # | 항목 | 상세 | 난이도 |
|---|------|------|:---:|
| 6 | **정확도 재검증** | `--max-images` 늘리거나 랜덤 샘플링으로 변경 (젯슨 접속 필요) | 중간 |
| 7 | ~~레이어 분석 결과 확인~~ | ✅ 해결됨 — 젯슨 SSH 확인 완료 (`*_layers.csv` + PNG 존재) | - |
| 8 | **accuracy_poll.txt 정리** | 내용이 불완전 ("Starting evaluation..." 2줄만) — 삭제 가능 | 쉬움 |
