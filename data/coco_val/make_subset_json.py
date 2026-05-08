#!/usr/bin/env python3
# =============================================================================
# make_subset_json.py — COCO val2017 → seed 고정 subset annotation 생성
# =============================================================================
# 역할: 전체 instances_val2017.json 에서 N 장 이미지를 무작위로 선택하고
#       해당 이미지의 annotation 만 추출한 새 json 을 저장한다.
#       seed 를 고정해 모든 런타임이 동일 이미지로 평가받도록 보장.
#
# 출력:
#   - instances_val2017_subset500.json (또는 N 에 따른 이름)
#   - selected_image_ids.txt — 선택된 image_id 목록 (다운로드 스크립트에서 참조)
#
# 사용법:
#   python data/coco_val/make_subset_json.py \
#     --annotations data/coco_val/annotations/instances_val2017.json \
#     --output data/coco_val/annotations/instances_val2017_subset500.json \
#     --num-images 500 --seed 42
# =============================================================================

"""Generate seed-fixed COCO val2017 subset annotation JSON."""

import argparse
import json
import os
import random


def make_subset(annotations_path, output_path, num_images=500, seed=42):
    """전체 annotation 에서 N 장 이미지의 부분집합 json 생성.

    선택 알고리즘:
        1) image_id 정렬 (재현성)
        2) random.Random(seed).sample() 로 N 장 추출
        3) 해당 image_id 와 그에 속한 annotation 만 필터
        4) categories 는 전체 유지 (mAP 계산 시 카테고리 매칭 필수)
    """
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations not found: {annotations_path}")

    print(f"[1/4] Loading {annotations_path}...")
    with open(annotations_path, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    print(f"  total images:      {len(images)}")
    print(f"  total annotations: {len(annotations)}")
    print(f"  categories:        {len(categories)}")

    if num_images > len(images):
        print(f"[WARN] requested {num_images} > available {len(images)}, "
              f"using all")
        num_images = len(images)

    # 2) 재현 가능한 샘플링
    print(f"[2/4] Sampling {num_images} images (seed={seed})...")
    sorted_images = sorted(images, key=lambda x: x["id"])
    rng = random.Random(seed)
    selected = rng.sample(sorted_images, num_images)
    selected_ids = set(img["id"] for img in selected)

    # 3) 해당 이미지 annotation 만 필터
    print(f"[3/4] Filtering annotations...")
    filtered_anns = [a for a in annotations if a["image_id"] in selected_ids]
    print(f"  filtered annotations: {len(filtered_anns)}")

    # 4) 결과 dict 구성 (info, licenses 도 보존)
    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": selected,
        "annotations": filtered_anns,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(subset, f)

    # selected_image_ids.txt 도 함께 저장 (다운로드 스크립트가 사용)
    ids_path = os.path.join(
        os.path.dirname(output_path), "selected_image_ids.txt"
    )
    with open(ids_path, "w") as f:
        for img in selected:
            f.write(f"{img['id']}\t{img['file_name']}\n")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[4/4] Saved subset:")
    print(f"  {output_path} ({size_mb:.2f} MB)")
    print(f"  {ids_path}")
    print(f"  images:      {len(selected)}")
    print(f"  annotations: {len(filtered_anns)}")


def main():
    parser = argparse.ArgumentParser(
        description="Make seed-fixed COCO val2017 subset annotation"
    )
    parser.add_argument(
        "--annotations",
        default="data/coco_val/annotations/instances_val2017.json",
        help="Path to full COCO val2017 annotation json",
    )
    parser.add_argument(
        "--output",
        default="data/coco_val/annotations/instances_val2017_subset500.json",
        help="Output subset annotation json",
    )
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    make_subset(args.annotations, args.output, args.num_images, args.seed)


if __name__ == "__main__":
    main()
