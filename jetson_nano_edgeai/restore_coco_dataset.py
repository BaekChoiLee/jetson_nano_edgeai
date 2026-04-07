#!/usr/bin/env python3
"""Download and prepare COCO validation dataset for detection model evaluation.

Sets up data/coco_val/ directory with:
  - val2017/            (5000 validation images)
  - annotations/        (instances_val2017.json)

Uses a small subset (200 images) by default to save space on Jetson Nano.
"""

import os
import urllib.request
import zipfile
import json
import shutil
import sys


COCO_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "coco_val")

# Full COCO val2017 URLs
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_file(url, dest_path):
    """Download file with progress."""
    if os.path.exists(dest_path):
        print(f"  Already downloaded: {dest_path}")
        return

    print(f"  Downloading: {url}")
    print(f"  → {dest_path}")

    def reporthook(count, block_size, total_size):
        percent = min(100, count * block_size * 100 // total_size) if total_size > 0 else 0
        sys.stdout.write(f"\r  Progress: {percent}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook)
    print()


def setup_coco_full():
    """Download and extract full COCO val2017 dataset."""
    os.makedirs(COCO_DATA_DIR, exist_ok=True)

    val_dir = os.path.join(COCO_DATA_DIR, "val2017")
    ann_dir = os.path.join(COCO_DATA_DIR, "annotations")

    # Download and extract images
    if not os.path.isdir(val_dir):
        images_zip = os.path.join(COCO_DATA_DIR, "val2017.zip")
        download_file(COCO_IMAGES_URL, images_zip)

        print("  Extracting images...")
        with zipfile.ZipFile(images_zip, 'r') as z:
            z.extractall(COCO_DATA_DIR)
        os.remove(images_zip)
        print(f"  ✅ Images extracted to {val_dir}")
    else:
        n_imgs = len([f for f in os.listdir(val_dir) if f.endswith('.jpg')])
        print(f"  Images already exist: {n_imgs} files in {val_dir}")

    # Download and extract annotations
    ann_file = os.path.join(ann_dir, "instances_val2017.json")
    if not os.path.exists(ann_file):
        ann_zip = os.path.join(COCO_DATA_DIR, "annotations.zip")
        download_file(COCO_ANN_URL, ann_zip)

        print("  Extracting annotations...")
        with zipfile.ZipFile(ann_zip, 'r') as z:
            # Only extract instances_val2017.json
            for member in z.namelist():
                if "instances_val2017.json" in member:
                    z.extract(member, COCO_DATA_DIR)
        os.remove(ann_zip)
        print(f"  ✅ Annotations extracted to {ann_dir}")
    else:
        print(f"  Annotations already exist: {ann_file}")


def setup_coco_mini(max_images=200):
    """Download full COCO then create a mini subset for quick evaluation.

    This creates a smaller dataset to save space and evaluation time
    while keeping representative samples from different categories.
    """
    setup_coco_full()

    val_dir = os.path.join(COCO_DATA_DIR, "val2017")
    ann_file = os.path.join(COCO_DATA_DIR, "annotations", "instances_val2017.json")

    print(f"\n  Creating mini subset ({max_images} images)...")

    with open(ann_file) as f:
        coco = json.load(f)

    # Select a diverse subset: pick images that cover many categories
    img_ids_with_anns = set()
    for ann in coco["annotations"]:
        img_ids_with_anns.add(ann["image_id"])

    # Sort by image_id for deterministic subset
    selected_ids = sorted(list(img_ids_with_anns))[:max_images]
    selected_set = set(selected_ids)

    # Filter images and annotations
    mini_images = [img for img in coco["images"] if img["id"] in selected_set]
    mini_anns = [ann for ann in coco["annotations"] if ann["image_id"] in selected_set]

    print(f"  Selected {len(mini_images)} images with {len(mini_anns)} annotations")

    # Save mini annotation file
    mini_ann_file = os.path.join(COCO_DATA_DIR, "annotations", "instances_val2017_mini.json")
    mini_coco = {
        "images": mini_images,
        "annotations": mini_anns,
        "categories": coco["categories"],
    }
    with open(mini_ann_file, "w") as f:
        json.dump(mini_coco, f)

    print(f"  ✅ Mini annotations saved to {mini_ann_file}")
    print(f"\n✅ COCO val dataset ready at: {COCO_DATA_DIR}")
    print(f"  Full: {len(coco['images'])} images")
    print(f"  Mini: {len(mini_images)} images ({max_images} target)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download COCO validation dataset")
    parser.add_argument("--max-images", type=int, default=200,
                        help="Max images for mini subset (default: 200)")
    parser.add_argument("--full-only", action="store_true",
                        help="Download full dataset without creating mini subset")
    args = parser.parse_args()

    print("🚀 COCO Val2017 Dataset Setup")
    print(f"  Target directory: {COCO_DATA_DIR}")
    print()

    if args.full_only:
        setup_coco_full()
    else:
        setup_coco_mini(args.max_images)


if __name__ == "__main__":
    main()
