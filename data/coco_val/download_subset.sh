#!/bin/bash
# =============================================================================
# download_subset.sh — COCO val2017 annotations + 500장 subset 이미지 준비
# =============================================================================
# 역할: pycocotools 평가에 필요한 annotation json 과 subset 이미지를 모두 준비.
#       기존 파일이 있으면 건너뜀(idempotent).
#
# 단계:
#   1) annotations_trainval2017.zip 다운로드 + 압축 해제
#      → instances_val2017.json (241MB) 추출
#   2) make_subset_json.py 실행: seed 42 로 500장 subset annotation 생성
#   3) download_selected_images.py 실행: 500장 이미지만 직접 다운로드
#
# 주의:
#   - val2017.zip 전체(약 1GB)는 받지 않음 — subset 이미지만 직접 다운로드
#   - seed 고정으로 모든 런타임이 동일 이미지 평가 보장
#
# 사용법:
#   bash data/coco_val/download_subset.sh
# =============================================================================

set -euo pipefail

# 스크립트 위치 기준 절대 경로
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ANN_DIR="$SCRIPT_DIR/annotations"
IMG_DIR="$SCRIPT_DIR/images"
SUBSET_JSON="$ANN_DIR/instances_val2017_subset500.json"
FULL_JSON="$ANN_DIR/instances_val2017.json"

mkdir -p "$ANN_DIR" "$IMG_DIR"

# =============================================================================
# Step 1: annotations 다운로드
# =============================================================================
if [ -f "$FULL_JSON" ]; then
    echo "[1/3] [SKIP] $FULL_JSON already exists"
else
    echo "[1/3] Downloading annotations_trainval2017.zip (~241 MB)..."
    cd "$ANN_DIR"
    ZIP_FILE="annotations_trainval2017.zip"
    if [ ! -f "$ZIP_FILE" ]; then
        wget -q --show-progress \
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
            -O "$ZIP_FILE"
    fi
    echo "    Extracting instances_val2017.json..."
    unzip -o -q "$ZIP_FILE" "annotations/instances_val2017.json"
    # unzip 결과는 annotations/annotations/instances_val2017.json 형태
    if [ -f "annotations/instances_val2017.json" ]; then
        mv "annotations/instances_val2017.json" "$FULL_JSON"
        rmdir "annotations" 2>/dev/null || true
    fi
    rm -f "$ZIP_FILE"
    echo "    [OK] $FULL_JSON"
fi

# =============================================================================
# Step 2: subset annotation 생성
# =============================================================================
if [ -f "$SUBSET_JSON" ]; then
    echo "[2/3] [SKIP] $SUBSET_JSON already exists"
else
    echo "[2/3] Generating subset annotation (seed 42, 500 images)..."
    cd "$PROJECT_DIR"
    python3 "$SCRIPT_DIR/make_subset_json.py" \
        --annotations "$FULL_JSON" \
        --output "$SUBSET_JSON" \
        --num-images 500 \
        --seed 42
fi

# =============================================================================
# Step 3: 500장 이미지 다운로드
# =============================================================================
echo "[3/3] Downloading 500 subset images to $IMG_DIR ..."
cd "$PROJECT_DIR"
python3 "$SCRIPT_DIR/download_selected_images.py" \
    --subset-json "$SUBSET_JSON" \
    --output-dir "$IMG_DIR" \
    --workers 4

echo ""
echo "[DONE] COCO val2017 subset prepared:"
echo "  annotations: $SUBSET_JSON"
echo "  images:      $IMG_DIR ($(ls "$IMG_DIR" 2>/dev/null | wc -l) files)"
