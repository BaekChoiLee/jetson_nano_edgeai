#!/bin/bash
# =============================================================================
# download_tf2_ssd.sh — 순정 SSD MobileNet V2 (TF2 OD Model Zoo) 다운로드
# =============================================================================
# 역할: NVIDIA TRT 공식 샘플이 지원하는 SavedModel 버전을 받아 압축 해제.
#
# 출력: models/ssd_mobilenet_v2_saved_model/saved_model/
#
# 사용법:
#   bash convert/download_tf2_ssd.sh
# =============================================================================

set -euo pipefail

# 스크립트 위치 기준으로 프로젝트 루트 계산
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_DIR/models"

mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

URL="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
TARBALL="ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
SRC_DIR="ssd_mobilenet_v2_320x320_coco17_tpu-8"
DST_DIR="ssd_mobilenet_v2_saved_model"

if [ -d "$DST_DIR" ]; then
    echo "[SKIP] $DST_DIR already exists"
    exit 0
fi

echo "[1/3] Downloading SSD MobileNet V2 320x320 (TF2 OD Model Zoo)..."
wget -q --show-progress "$URL" -O "$TARBALL"

echo "[2/3] Extracting..."
tar -xzf "$TARBALL"

echo "[3/3] Moving to $DST_DIR..."
mv "$SRC_DIR" "$DST_DIR"
rm -f "$TARBALL"

echo ""
echo "[OK] $MODELS_DIR/$DST_DIR/ prepared"
ls -la "$DST_DIR/saved_model" 2>/dev/null || true
