#!/bin/bash
# =============================================================================
# rebenchmark_detection_2models.sh
# =============================================================================
# 목적:
#   YOLOv8n + SSD-MobileNetV2 검출 2모델을 Jetson에서 안정적으로 재측정한다.
#
# 수행 단계:
#   1) (옵션) detection 아티팩트 준비/보정
#   2) latency: 2모델 × 11런타임 × 2전력모드 (MAXN/5W)
#   3) accuracy: COCO mAP (2모델 × 11런타임)
#
# 결과:
#   results/detection2_<timestamp>/
#     latency/{MAXN,5W}/*.json
#     accuracy/*_detection.json
#     run.log
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
# 사용자 로컬 바이너리(onnx2tf/onnxsim/pnnx 등) 우선 탐색
export PATH="$HOME/.local/bin:$PATH"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="results/detection2_${TIMESTAMP}"
LOG_FILE="${RESULTS_ROOT}/run.log"

DETECTION_MODELS=("yolov8n" "ssd_mobilenet_v2")
YOLO_RUNTIMES=(
  "pytorch_cpu" "pytorch_cuda"
  "tensorrt_fp32" "tensorrt_fp16" "tensorrt_int8"
  "onnxrt_cuda" "onnxrt_trt"
  "tflite_cpu" "tflite_gpu"
  "ncnn_cpu" "ncnn_vulkan"
)
# SSD도 동일하게 전 런타임을 전부 시도한다.
# (실패 조합은 JSON error로 사유를 남기고 후속 N/A 분류)
SSD_RUNTIMES=(
  "pytorch_cpu" "pytorch_cuda"
  "tensorrt_fp32" "tensorrt_fp16" "tensorrt_int8"
  "onnxrt_cuda" "onnxrt_trt"
  "tflite_cpu" "tflite_gpu"
  "ncnn_cpu" "ncnn_vulkan"
)
POWER_MODES=("MAXN" "5W")

NUM_WARMUP=10
NUM_RUNS=100
COOLDOWN_SEC=0
MAX_IMAGES=500

SKIP_PREP=0
SKIP_LATENCY=0
SKIP_ACCURACY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --num-warmup) NUM_WARMUP="$2"; shift 2 ;;
    --num-runs) NUM_RUNS="$2"; shift 2 ;;
    --cooldown) COOLDOWN_SEC="$2"; shift 2 ;;
    --max-images) MAX_IMAGES="$2"; shift 2 ;;
    --skip-prepare) SKIP_PREP=1; shift ;;
    --skip-latency) SKIP_LATENCY=1; shift ;;
    --skip-accuracy) SKIP_ACCURACY=1; shift ;;
    -h|--help)
      echo "Usage: bash rebenchmark_detection_2models.sh [options]"
      echo "  --results-root <dir>"
      echo "  --num-warmup <int> --num-runs <int>"
      echo "  --cooldown <sec> --max-images <int>"
      echo "  --skip-prepare --skip-latency --skip-accuracy"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

LOG_FILE="${RESULTS_ROOT}/run.log"
mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "Detection 2-model benchmark"
echo "  results:    $RESULTS_ROOT"
echo "  warmup/runs ${NUM_WARMUP}/${NUM_RUNS}"
echo "  yolo_rt:    ${#YOLO_RUNTIMES[@]} (${YOLO_RUNTIMES[*]})"
echo "  ssd_rt:     ${#SSD_RUNTIMES[@]} (${SSD_RUNTIMES[*]})"
echo "  cooldown:   ${COOLDOWN_SEC}s"
echo "============================================================"

clean_system() {
  echo "[clean] drop caches + cooldown ${COOLDOWN_SEC}s"
  sync || true
  if [ -w /proc/sys/vm/drop_caches ]; then
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  fi
  sleep "$COOLDOWN_SEC"
}

set_power_mode() {
  local mode="$1"
  if command -v nvpmodel >/dev/null 2>&1; then
    if [ "$mode" = "MAXN" ]; then
      sudo nvpmodel -m 0 || true
    else
      sudo nvpmodel -m 1 || true
    fi
    sudo jetson_clocks || true
  fi
  echo "[power] $mode"
}

get_runtimes_for_model() {
  local model="$1"
  if [ "$model" = "ssd_mobilenet_v2" ]; then
    echo "${SSD_RUNTIMES[@]}"
  else
    echo "${YOLO_RUNTIMES[@]}"
  fi
}

has_yolo_tflite() {
  local cands=(
    "models/yolov8n.tflite"
    "models/yolov8n_float32.tflite"
    "models/yolov8n_saved_model/yolov8n_float32.tflite"
    "models/yolov8n_saved_model/yolov8n.tflite"
  )
  local f
  for f in "${cands[@]}"; do
    if [ -f "$f" ]; then
      return 0
    fi
  done
  return 1
}

has_yolo_ncnn() {
  # ultralytics 기본: yolov8n_ncnn_model/model.{param,bin}
  if [ -f "models/yolov8n_ncnn_model/model.param" ] && [ -f "models/yolov8n_ncnn_model/model.bin" ]; then
    return 0
  fi
  # pnnx/onnx2ncnn 계열 flat 파일
  if [ -f "models/yolov8n.param" ] && [ -f "models/yolov8n.bin" ]; then
    return 0
  fi
  if [ -f "models/yolov8n.ncnn.param" ] && [ -f "models/yolov8n.ncnn.bin" ]; then
    return 0
  fi
  return 1
}

normalize_yolo_artifacts() {
  # 런타임 파이프라인 호환용 canonical 이름 보정
  if [ ! -f "models/yolov8n.tflite" ]; then
    local src=""
    for src in \
      "models/yolov8n_float32.tflite" \
      "models/yolov8n_saved_model/yolov8n_float32.tflite" \
      "models/yolov8n_saved_model/yolov8n.tflite"; do
      if [ -f "$src" ]; then
        cp -f "$src" "models/yolov8n.tflite"
        echo "  [prep] normalized TFLite -> models/yolov8n.tflite (from $src)"
        break
      fi
    done
  fi
}

has_yolo_artifacts() {
  [ -f "models/yolov8n.onnx" ] || return 1
  [ -f "models/yolov8n.torchscript" ] || return 1
  has_yolo_tflite || return 1
  has_yolo_ncnn || return 1
  return 0
}

has_ssd_raw_onnx() {
  [ -f "models/ssd_mobilenet_v2_raw.onnx" ]
}

ssd_raw_onnx_has_valid_class_shape() {
  python3 - <<'PY' >/dev/null 2>&1
import os, sys
try:
    import onnx
except Exception:
    sys.exit(1)
p = "models/ssd_mobilenet_v2_raw.onnx"
if not os.path.exists(p):
    sys.exit(1)
m = onnx.load(p)
outs = {o.name: o for o in m.graph.output}
cls = outs.get("class_scores")
if cls is None:
    sys.exit(1)
dims = []
for d in cls.type.tensor_type.shape.dim:
    dims.append(int(d.dim_value) if getattr(d, "dim_value", 0) > 0 else -1)
if len(dims) != 3:
    sys.exit(1)
c1, c2 = dims[1], dims[2]
ok = not (min(c1, c2) in (0, 1, 2) and max(c1, c2) >= 100)
sys.exit(0 if ok else 1)
PY
}

has_ssd_torchscript() {
  [ -f "models/ssd_mobilenet_v2.torchscript" ]
}

has_nonempty_file() {
  [ -f "$1" ] && [ -s "$1" ]
}

has_ssd_trt() {
  local ok_fp32=0 ok_fp16=0 ok_int8=0
  if has_nonempty_file "models/ssd_mobilenet_v2_fp32.engine" || has_nonempty_file "models/ssd_mobilenet_v2_raw_fp32.engine" || has_nonempty_file "models/ssd_mobilenet_v2_raw_fpinput_fp32.engine" || has_nonempty_file "models/ssd_mobilenet_v2_raw3_fpinput_fp32.engine"; then ok_fp32=1; fi
  if has_nonempty_file "models/ssd_mobilenet_v2_fp16.engine" || has_nonempty_file "models/ssd_mobilenet_v2_raw_fp16.engine" || has_nonempty_file "models/ssd_mobilenet_v2_raw_fpinput_fp16.engine" || has_nonempty_file "models/ssd_mobilenet_v2_raw3_fpinput_fp16.engine"; then ok_fp16=1; fi
  if has_nonempty_file "models/ssd_mobilenet_v2_int8.engine" || has_nonempty_file "models/ssd_mobilenet_v2_raw_int8.engine" || has_nonempty_file "models/ssd_mobilenet_v2_raw_fpinput_int8.engine" || has_nonempty_file "models/ssd_mobilenet_v2_raw3_fpinput_int8.engine"; then ok_int8=1; fi
  [ "$ok_fp32" -eq 1 ] && [ "$ok_fp16" -eq 1 ] && [ "$ok_int8" -eq 1 ]
}

has_ssd_tflite() {
  [ -f "models/ssd_mobilenet_v2.tflite" ] \
    || [ -f "models/ssd_mobilenet_v2_raw.tflite" ] \
    || [ -f "models/ssd_mobilenet_v2_float32.tflite" ] \
    || [ -f "models/ssd_mobilenet_v2_raw_float32.tflite" ]
}

has_ssd_ncnn() {
  if [ -f "models/ssd_mobilenet_v2_raw.ncnn.status.json" ]; then
    if ! python3 - <<'PY' >/dev/null 2>&1
import json
p="models/ssd_mobilenet_v2_raw.ncnn.status.json"
with open(p, "r", encoding="utf-8") as f:
    j = json.load(f)
raise SystemExit(0 if bool(j.get("ok", False)) else 1)
PY
    then
      return 1
    fi
  fi
  { has_nonempty_file "models/ssd_mobilenet_v2.param" && has_nonempty_file "models/ssd_mobilenet_v2.bin"; } \
    || { has_nonempty_file "models/ssd_mobilenet_v2_raw.param" && has_nonempty_file "models/ssd_mobilenet_v2_raw.bin"; } \
    || { has_nonempty_file "models/ssd_mobilenet_v2.ncnn.param" && has_nonempty_file "models/ssd_mobilenet_v2.ncnn.bin"; } \
    || { has_nonempty_file "models/ssd_mobilenet_v2_raw.ncnn.param" && has_nonempty_file "models/ssd_mobilenet_v2_raw.ncnn.bin"; }
}

normalize_ssd_trt_engines() {
  # 생성 엔진명이 제각각일 수 있어 canonical 이름으로 정규화
  local src=""
  if ! has_nonempty_file "models/ssd_mobilenet_v2_fp32.engine"; then
    for src in \
      "models/ssd_mobilenet_v2_raw_fp32.engine" \
      "models/ssd_mobilenet_v2_raw_fpinput_fp32.engine" \
      "models/ssd_mobilenet_v2_raw3_fpinput_fp32.engine" \
      "models/ssd_mobilenet_v2_effnms_fpinput_fp32.engine"; do
      if has_nonempty_file "$src"; then
        cp -f "$src" "models/ssd_mobilenet_v2_fp32.engine"
        echo "  [prep] normalized TRT FP32 -> models/ssd_mobilenet_v2_fp32.engine (from $src)"
        break
      fi
    done
  fi
  if ! has_nonempty_file "models/ssd_mobilenet_v2_fp16.engine"; then
    for src in \
      "models/ssd_mobilenet_v2_raw_fp16.engine" \
      "models/ssd_mobilenet_v2_raw_fpinput_fp16.engine" \
      "models/ssd_mobilenet_v2_raw3_fpinput_fp16.engine" \
      "models/ssd_mobilenet_v2_effnms_fpinput_fp16.engine"; do
      if has_nonempty_file "$src"; then
        cp -f "$src" "models/ssd_mobilenet_v2_fp16.engine"
        echo "  [prep] normalized TRT FP16 -> models/ssd_mobilenet_v2_fp16.engine (from $src)"
        break
      fi
    done
  fi
  if ! has_nonempty_file "models/ssd_mobilenet_v2_int8.engine"; then
    for src in \
      "models/ssd_mobilenet_v2_raw_int8.engine" \
      "models/ssd_mobilenet_v2_raw_fpinput_int8.engine" \
      "models/ssd_mobilenet_v2_raw3_fpinput_int8.engine" \
      "models/ssd_mobilenet_v2_effnms_fpinput_int8.engine"; do
      if has_nonempty_file "$src"; then
        cp -f "$src" "models/ssd_mobilenet_v2_int8.engine"
        echo "  [prep] normalized TRT INT8 -> models/ssd_mobilenet_v2_int8.engine (from $src)"
        break
      fi
    done
  fi
}

prepare_detection_artifacts() {
  if [ "$SKIP_PREP" -eq 1 ]; then
    echo "[1/3] prepare artifacts — SKIPPED"
    return
  fi

  echo ""
  echo "[1/3] prepare detection artifacts"

  # YOLOv8n
  if ! has_yolo_artifacts; then
    echo "  [prep] YOLOv8n export (onnx/torchscript/tflite/ncnn)"
    python3 convert/export_detection.py --output-dir models || true
  fi
  normalize_yolo_artifacts
  if ! has_yolo_ncnn && [ -f "models/yolov8n.onnx" ]; then
    echo "  [prep] YOLOv8n ncnn fallback (onnx2ncnn)"
    bash convert/convert_ncnn.sh "models/yolov8n.onnx" || true
  fi
  if ! has_yolo_tflite && [ -f "models/yolov8n.onnx" ]; then
    echo "  [prep] YOLOv8n tflite fallback (onnx->tflite)"
    python3 convert/convert_tflite.py --onnx-path "models/yolov8n.onnx" --output-dir "models" || true
    normalize_yolo_artifacts
  fi
  if [ -f "models/yolov8n.onnx" ]; then
    if [ ! -f "models/yolov8n_fp32.engine" ] || [ ! -f "models/yolov8n_fp16.engine" ] || [ ! -f "models/yolov8n_int8.engine" ]; then
      echo "  [prep] YOLOv8n TRT engines"
      bash convert/build_trt.sh "models/yolov8n.onnx" || true
    fi
  fi

  # SSD-MobileNetV2
  if ! has_ssd_raw_onnx || ! ssd_raw_onnx_has_valid_class_shape; then
    if [ -d "models/ssd_mobilenet_v2_saved_model/saved_model" ]; then
      if has_ssd_raw_onnx && ! ssd_raw_onnx_has_valid_class_shape; then
        echo "  [prep] SSD raw ONNX class shape invalid -> rebuild"
      else
        echo "  [prep] SSD raw ONNX missing -> build"
      fi
      # TFOD 공식 경로(raw_detection_scores 보존) 우선
      python3 convert/tf2_od_to_onnx.py || true
      # 그래도 raw가 없으면 direct tf2onnx로 마지막 재시도
      if [ ! -f "models/ssd_mobilenet_v2_raw.onnx" ]; then
        echo "  [prep] SSD raw ONNX (tf2onnx direct fallback)"
        python3 -m tf2onnx.convert \
          --saved-model "models/ssd_mobilenet_v2_saved_model/saved_model" \
          --output "models/ssd_mobilenet_v2_raw.onnx" \
          --opset 11 || true
      fi
    else
      echo "  [warn] SSD saved_model missing: models/ssd_mobilenet_v2_saved_model/saved_model"
    fi
  fi

  # Jetson ORT 1.11 호환: IR version 상한(8) 보정
  if [ -f "models/ssd_mobilenet_v2_raw.onnx" ]; then
    python3 - <<'PY' || true
import onnx
p="models/ssd_mobilenet_v2_raw.onnx"
m=onnx.load(p)
if getattr(m, "ir_version", 0) > 8:
    m.ir_version = 8
    onnx.save(m, p)
    print("[prep] SSD ONNX IR downgraded to 8 for ORT 1.11")
PY
  fi

  # SSD 추가 아티팩트 체인: raw ONNX -> TorchScript/TRT/TFLite/NCNN
  if has_ssd_raw_onnx && ! has_ssd_torchscript; then
    echo "  [prep] SSD TorchScript (onnx_to_pytorch)"
    python3 convert/onnx_to_pytorch.py \
      --onnx "models/ssd_mobilenet_v2_raw.onnx" \
      --output "models/ssd_mobilenet_v2.torchscript" || true
  fi
  if has_ssd_raw_onnx && ! has_ssd_trt; then
    echo "  [prep] SSD TRT engines"
    bash convert/build_trt.sh "models/ssd_mobilenet_v2_raw.onnx" || true
  fi
  normalize_ssd_trt_engines
  if has_ssd_raw_onnx && ! has_ssd_tflite; then
    echo "  [prep] SSD TFLite (onnx->tflite)"
    python3 convert/convert_tflite.py \
      --onnx-path "models/ssd_mobilenet_v2_raw.onnx" \
      --output-dir "models" || true
  fi
  if has_ssd_raw_onnx && ! has_ssd_ncnn; then
    echo "  [prep] SSD ncnn (onnx2ncnn)"
    bash convert/convert_ncnn.sh "models/ssd_mobilenet_v2_raw.onnx" || true
    if [ -f "models/ssd_mobilenet_v2_raw.ncnn.status.json" ]; then
      echo "  [prep] SSD ncnn status: models/ssd_mobilenet_v2_raw.ncnn.status.json"
    fi
  fi

  echo "  [prep] done"
}

run_latency() {
  if [ "$SKIP_LATENCY" -eq 1 ]; then
    echo "[2/3] latency — SKIPPED"
    return
  fi

  echo ""
  echo "[2/3] latency benchmark (model-specific runtimes × 2 power)"

  for mode in "${POWER_MODES[@]}"; do
    set_power_mode "$mode"
    for model in "${DETECTION_MODELS[@]}"; do
      local model_runtimes
      # shellcheck disable=SC2206
      model_runtimes=($(get_runtimes_for_model "$model"))
      for rt in "${model_runtimes[@]}"; do
        local out="${RESULTS_ROOT}/latency/${mode}/${model}_${rt}.json"
        local tegra="${RESULTS_ROOT}/latency/${mode}/${model}_${rt}.tegra.log"
        if [ -f "$out" ]; then
          echo "  [skip] $out"
          continue
        fi
        mkdir -p "$(dirname "$out")"
        clean_system
        echo "  [run] $mode / $model / $rt"
        python3 benchmark/run_detection_latency.py \
          --model "$model" \
          --runtime "$rt" \
          --num-warmup "$NUM_WARMUP" \
          --num-runs "$NUM_RUNS" \
          --model-dir "models" \
          --tegra-log "$tegra" \
          --output "$out" || echo "    [fail] $mode / $model / $rt"
      done
    done
  done
}

run_accuracy() {
  if [ "$SKIP_ACCURACY" -eq 1 ]; then
    echo "[3/3] accuracy — SKIPPED"
    return
  fi

  echo ""
  echo "[3/3] COCO mAP accuracy"

  if [ ! -f "data/coco_val/annotations/instances_val2017_subset500.json" ]; then
    echo "  [warn] annotations missing: data/coco_val/annotations/instances_val2017_subset500.json"
    return
  fi

  mkdir -p "${RESULTS_ROOT}/accuracy"
  for model in "${DETECTION_MODELS[@]}"; do
    local model_runtimes
    # shellcheck disable=SC2206
    model_runtimes=($(get_runtimes_for_model "$model"))
    local out="${RESULTS_ROOT}/accuracy/${model}_detection.json"
    if [ -f "$out" ]; then
      echo "  [skip] $out"
      continue
    fi
    echo "  [run] $model (${model_runtimes[*]})"
    python3 benchmark/run_detection_accuracy.py \
      --model "$model" \
      --runtimes "${model_runtimes[@]}" \
      --coco-dir "data/coco_val/images" \
      --annotations "data/coco_val/annotations/instances_val2017_subset500.json" \
      --model-dir "models" \
      --max-images "$MAX_IMAGES" \
      --output "$out" || echo "    [fail] $model"
  done
}

prepare_detection_artifacts
run_latency
run_accuracy

echo ""
echo "============================================================"
echo "[DONE] detection 2-model benchmark complete"
echo "  results: $RESULTS_ROOT"
echo "============================================================"
