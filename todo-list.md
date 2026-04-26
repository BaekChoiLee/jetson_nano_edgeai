# Jetson Nano Edge AI 벤치마크 프로젝트 TODO

## 1주차 — 환경 구축 및 검증

- [x] JetPack 4.x 플래싱 및 Ubuntu 18.04 기본 세팅
- [x] 4GB 스왑 메모리 추가 설정 (OOM 방지)
- [x] PyTorch, TensorRT, ONNX Runtime (CPU), TFLite, ncnn 설치
  - [x] ONNX Runtime GPU wheel 링크 만료 → CPU 버전(`pip install onnxruntime`)으로 대체 결정
- [x] Python 환경에서 CUDA 정상 동작 테스트
- [x] `tegrastats` 전력·메모리 로깅 정상 확인
- [x] SSH 원격 접속 환경 구성 (맥북 VSCode Remote SSH)
- [x] `setup.sh` 스크립트 작성 및 문서화 (버전 기록, 스크린샷, 패키지 버전 고정)

---

## 2주차 — 측정 파이프라인 개발 (맥북에서 구현 → git pull로 젯슨 적용)

### 개별 모듈 작성

- [x] `benchmark.py` — warm-up 10회 / 본 측정 100회 / 평균·표준편차·중앙값·min·max 통계 계산
- [x] `tegra_parser.py` — 100ms 간격 tegrastats 서브프로세스 실행 및 파싱: VDD_IN·VDD_CPU·VDD_GPU(mW), RAM(MB), GR3D_FREQ(%), 온도(℃)
- [x] `layer_analyzer.py` — PyTorch `register_forward_pre/post_hook` 레이어별 실행 시간(ms) + tegrastats 타임스탬프 매핑으로 전력 추정
- [x] TensorRT IProfiler 구현 — TRT 엔진 레이어별 latency breakdown 추출
- [x] `result_saver.py` — 실험 결과 CSV(분석용) + JSON(원본) 이중 저장
  - [x] 컬럼: `timestamp, model, runtime, precision, task_type, power_mode, mean_ms, std_ms, power_total_mw, power_gpu_mw, ram_mb, gpu_util_pct, co2_kg, metric(top1/mAP)`
- [x] codecarbon 연동 — 추론 중 전력 소비를 gCO₂eq로 자동 환산

### ncnn C++ 래퍼

- [x] ncnn 소스 빌드: Vulkan 포함, `jetson.toolchain.cmake` 사용 (빌드 1~2시간, 백그라운드 실행)
- [x] `ncnn_bench.cpp` 작성 — warm-up/반복 측정 후 JSON stdout 출력, Python에서 subprocess로 호출

### 통합 실행 진입점

- [x] `run_all.py` 작성 — 단일 파일 실행으로 전체 실험 파이프라인 자동화
  - [x] CLI 인자 설계: `--model` (all / 모델명) `--runtime` (all / 런타임명) `--task` (cls / det / all) `--power-mode` (5w / 10w) `--runs` (반복 횟수)
  - [x] 실행 예시: `python run_all.py --model all --runtime all --task cls --power-mode 10w`
  - [x] 모델 변환 → 런타임별 추론 → tegrastats 병렬 로깅 → 레이어 분석 → CSV/JSON 저장 순서 자동 실행
  - [x] 실험 진행 상태 출력 (진행률 표시, 각 단계 소요 시간 로깅)
  - [x] 특정 런타임/모델 변환 실패 시 해당 조합만 스킵 후 나머지 계속 진행 (에러 로그 저장)
  - [x] 이미 측정 완료된 조합은 CSV 확인 후 중복 실행 스킵 (재시작 안전성)
- [x] 전체 파이프라인 통합 테스트 — MobileNetV3-S 더미 입력으로 `run_all.py` end-to-end 동작 확인

---

## 3~4주차 — 분류 모델 실험 (Classification)

### 데이터셋 준비 (공통, 1회)

- [x] ImageNet validation subset 준비 — 클래스당 10장 × 100클래스 = 1,000장 샘플링 (약 300MB)
- [x] 전처리 파이프라인 — Resize(256) → CenterCrop(224) → Normalize, 런타임별 입력 형식(NCHW/NHWC) 변환

### MobileNetV3-Small [P1] [분류]

- [x] PyTorch 원본 로드 및 ImageNet Top-1 정확도 기록
- [x] ONNX export → TFLite → ncnn(.param/.bin) → TRT 엔진(FP32/FP16/INT8) 변환
- [x] `python run_all.py --model mobilenetv3s --task cls` 로 6개 런타임 전체 자동 측정
  - [x] 각 런타임 실행 중 tegrastats 병렬 로깅 → 전력·메모리·GPU 사용률 기록
  - [x] PyTorch hook + TRT Profiler 레이어별 실행 시간 + 전력 추정값 추출 → `layer_results.csv` 저장
- [x] 런타임별 정확도 검증 (INT8 양자화 시 Top-1 drop 기록)

### EfficientNet-B0 [P2] [분류]

- [x] PyTorch 원본 로드 및 ImageNet Top-1 정확도 기록
- [x] ONNX export → TFLite → ncnn → TRT 엔진(FP32/FP16/INT8) 변환
- [x] `python run_all.py --model efficientnetb0 --task cls` 로 6개 런타임 전체 자동 측정
  - [x] tegrastats 병렬 로깅, 레이어 분석 포함
- [x] MobileNetV3-S 결과와 1차 비교 — 모델 크기 2배 시 전력·속도 변화 분석

### ShuffleNetV2 [P3] [분류]

- [x] PyTorch 원본 로드 및 ImageNet Top-1 정확도 기록
- [x] ONNX export → TFLite → ncnn → TRT 엔진 변환
  - [x] channel shuffle 연산 ncnn 변환 특이사항 체크
- [x] `python run_all.py --model shufflenetv2 --task cls` 로 6개 런타임 전체 자동 측정

### ResNet-50 [P1] [분류]

- [x] PyTorch 원본 로드 및 ImageNet Top-1 정확도 기록
- [x] ONNX export → TFLite → ncnn → TRT 엔진 변환
  - [x] OOM 대비 배치 크기 1 고정, 스왑 여유 확인
- [x] `python run_all.py --model resnet50 --task cls` 로 6개 런타임 전체 자동 측정
- [x] 4개 분류 모델 레이어 분석 비교 — Conv/BN/Activation/Pooling 유형별 런타임 간 시간 차이 분석

---

## 5~6주차 — 탐지 모델 실험 (Object Detection)

### 데이터셋 준비 (공통, 1회)

- [ ] COCO 2017 validation subset 준비 — 500장 샘플링 (약 500MB)
- [ ] 전처리 파이프라인
  - [ ] YOLOv8용: letterbox resize(640×640), 런타임별 입력 형식 변환
  - [ ] SSD용: resize(300×300) + SSD normalize, 런타임별 입력 형식 변환
- [ ] 후처리 파이프라인 — NMS 포함 박스 디코딩, mAP@0.5 계산 스크립트 (pycocotools)

### YOLOv8n [P2] [탐지]

- [ ] ultralytics 공식 export로 ONNX → TFLite → ncnn → TRT 변환
- [ ] `python run_all.py --model yolov8n --task det` 로 6개 런타임 전체 자동 측정
  - [ ] tegrastats 병렬 로깅
  - [ ] 레이어 분석 — C2f, SPPF, Detect head 등 YOLO 고유 레이어 실행 시간·전력 추정
- [ ] 런타임별 mAP@0.5 검증 (NMS 후처리 런타임 포함 여부 명시)
- [ ] 열 스로틀링 방지 — 측정 전 5분 대기, GPU 온도 함께 기록

### SSD-MobileNetV2 [P3] [탐지]

- [ ] ONNX export → 각 런타임 변환 (SSD 특화 레이어 변환 이슈 체크)
- [ ] `python run_all.py --model ssd_mv2 --task det` 로 6개 런타임 전체 자동 측정
  - [ ] tegrastats 병렬 로깅
  - [ ] backbone(MobileNetV2) vs SSD head 분리 레이어 분석
- [ ] YOLOv8n 결과와 비교 — 동일 탐지 태스크에서 모델 구조 차이에 따른 런타임별 특성 비교

---

## 7주차 — 데이터 분석 및 시각화

### 환경 세팅

- [ ] Jupyter Notebook(`analysis.ipynb`) 세팅 및 전체 CSV 로드, 데이터 정합성 검증

### 런타임 비교 (전체 모델 × 런타임)

- [ ] 런타임 × 모델 추론 속도 히트맵 — 행: 6개 모델, 열: 6개 런타임, 셀: mean latency(ms), 색상: 속도 빠를수록 진함
- [ ] 전력-속도 트레이드오프 산점도 — x: mean_ms, y: VDD_IN(mW), 색상: 런타임 구분
- [ ] 런타임별 탄소 배출량(gCO₂eq) 막대그래프 — 모델별 서브플롯, 런타임 색상 통일
- [ ] TensorRT 양자화(FP32→FP16→INT8) 정확도-효율 곡선 — x: latency, y: Top-1/mAP, 모델별 라인

### 레이어 단위 분석

- [ ] 레이어별 latency 폭포수 차트(Waterfall) — 모델 1개 × 런타임 2개(PyTorch vs TRT) 비교, 병목 레이어 시각적 강조
- [ ] 레이어 유형별(Conv/BN/Act/Pooling) 평균 실행 시간 — 런타임별 grouped bar chart
- [ ] 레이어별 전력 추정 추이 — tegrastats 폴링 데이터를 추론 타임라인에 overlay, 전력 spike 구간과 레이어 매핑
- [ ] 모델별 레이어 수 vs 총 레이턴시 상관관계 — 분류 4개 모델 scatter

### 태스크 유형별 비교

- [ ] 분류 vs 탐지 전력 소비 비교 — 동일 런타임 기준, 입력 해상도(224 vs 640) 차이 영향 분리
- [ ] ONNX Runtime CPU vs TensorRT GPU 속도·전력 비교 표 — 런타임 선택 가이드라인 근거 자료

### 산출물

- [ ] 고해상도 그래프 (PDF/PNG 300dpi) 일괄 export
- [ ] 핵심 통계 요약표 — 모델×런타임 전체 mean/std/power/co2 정리