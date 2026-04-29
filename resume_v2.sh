#!/bin/bash
set -uo pipefail

cd "$(dirname "$0")"

RUN="${1:-results/clean_v2_20260414_092817}"
LOG="${2:-results/api_resume_$(date +%Y%m%d_%H%M%S)_v2.log}"

{
echo "[resume-v2] start $(date)"
echo "[resume-v2] run=$RUN"

mkdir -p "$RUN/accuracy" "$RUN/layer_power"

# Fill missing unsupported combo as explicit N/A
for mode in MAXN 5W; do
  f="$RUN/latency/$mode/ssd_mobilenet_v2_onnxrt_trt.json"
  if [ ! -f "$f" ]; then
    python3 - <<PY
import json
p = "$f"
obj = {
  "runtime": "onnxrt_trt",
  "model": "ssd_mobilenet_v2",
  "error": "N/A: ORT TensorRT EP unsupported/unstable on Jetson Nano for this model"
}
with open(p, "w", encoding="utf-8") as w:
    json.dump(obj, w, ensure_ascii=False, indent=2)
print("saved", p)
PY
  fi
done

# Remaining detection accuracy
if [ ! -f "$RUN/accuracy/ssd_mobilenet_v2_detection.json" ]; then
  echo "[resume-v2] run ssd detection accuracy"
  python3 benchmark/run_detection_accuracy.py \
    --model ssd_mobilenet_v2 \
    --output "$RUN/accuracy/ssd_mobilenet_v2_detection.json" \
    --per-runtime-timeout-sec 1200 \
    || echo "[warn] ssd detection accuracy failed"
fi

# Layer power sweep with timeout per model
for m in mobilenetv3_small resnet50 efficientnet_b0 shufflenet_v2_x1_0 yolov8n ssd_mobilenet_v2; do
  out="$RUN/layer_power/${m}_layer_power.csv"
  if [ -f "$out" ]; then
    echo "[skip] layer_power $m"
    continue
  fi
  echo "[resume-v2] layer_power $m"
  if command -v timeout >/dev/null 2>&1; then
    timeout 45m python3 benchmark/layer_power_analyzer.py \
      --model "$m" --device cuda --num-amplify 2000 --output-dir "$RUN/layer_power" \
      || echo "[warn] layer_power failed/timeout $m"
  else
    python3 benchmark/layer_power_analyzer.py \
      --model "$m" --device cuda --num-amplify 2000 --output-dir "$RUN/layer_power" \
      || echo "[warn] layer_power failed $m"
  fi
done

echo "[resume-v2] consolidate"
python3 analysis/consolidate_v2.py --results-dir "$RUN" --output-dir "$RUN" || echo "[warn] consolidate failed"

echo "[resume-v2] done $(date)"
} >> "$LOG" 2>&1
