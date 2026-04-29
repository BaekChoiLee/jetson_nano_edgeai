#!/usr/bin/env python3
# =============================================================================
# download_selected_images.py — subset 에 포함된 COCO 이미지만 다운로드
# =============================================================================
# 역할: make_subset_json.py 가 만든 selected_image_ids.txt (또는 subset json)
#       에 포함된 file_name 만 http://images.cocodataset.org/val2017/ 에서
#       내려받는다. 전체 val2017 5000장(약 1GB) 받지 않고 500장(~80MB)만 받음.
#
# 사용법:
#   python data/coco_val/download_selected_images.py \
#     --subset-json data/coco_val/annotations/instances_val2017_subset500.json \
#     --output-dir data/coco_val/images/
# =============================================================================

"""Download only the COCO val2017 images included in the subset."""

import argparse
import json
import os
import sys
import time
from urllib.request import urlopen, Request


COCO_BASE_URL = "http://images.cocodataset.org/val2017/"


def download_one(file_name, output_dir, retries=3, timeout=30):
    """단일 이미지 다운로드 (재시도 포함)."""
    url = COCO_BASE_URL + file_name
    out_path = os.path.join(output_dir, file_name)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True, "skip"

    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "jetson-benchmark/1.0"})
            with urlopen(req, timeout=timeout) as r:
                data = r.read()
            with open(out_path, "wb") as f:
                f.write(data)
            return True, "ok"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.0 + attempt * 1.0)
                continue
            return False, str(e)
    return False, "exhausted retries"


def download_all(subset_json, output_dir, max_workers=4):
    """subset json 에서 file_name 을 모두 추출 후 다운로드."""
    if not os.path.exists(subset_json):
        print(f"[ERROR] subset json not found: {subset_json}")
        sys.exit(1)

    print(f"[1/3] Loading subset json...")
    with open(subset_json, "r") as f:
        subset = json.load(f)

    images = subset.get("images", [])
    file_names = [img["file_name"] for img in images]
    print(f"  to download: {len(file_names)} images")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[2/3] Downloading to {output_dir}...")
    n_ok = 0
    n_skip = 0
    n_fail = 0
    failures = []

    # 단순 순차 다운로드 (병렬은 옵션 — concurrent.futures 필요)
    if max_workers <= 1:
        for i, fn in enumerate(file_names, 1):
            ok, status = download_one(fn, output_dir)
            if ok and status == "skip":
                n_skip += 1
            elif ok:
                n_ok += 1
            else:
                n_fail += 1
                failures.append((fn, status))
            if i % 50 == 0:
                print(f"  progress: {i}/{len(file_names)} "
                      f"(ok={n_ok}, skip={n_skip}, fail={n_fail})")
    else:
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
        except ImportError:
            return download_all(subset_json, output_dir, max_workers=1)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(download_one, fn, output_dir): fn
                for fn in file_names
            }
            done = 0
            for fut in as_completed(futures):
                fn = futures[fut]
                try:
                    ok, status = fut.result()
                except Exception as e:
                    ok, status = False, str(e)
                if ok and status == "skip":
                    n_skip += 1
                elif ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    failures.append((fn, status))
                done += 1
                if done % 50 == 0:
                    print(f"  progress: {done}/{len(file_names)} "
                          f"(ok={n_ok}, skip={n_skip}, fail={n_fail})")

    print(f"\n[3/3] Summary:")
    print(f"  downloaded: {n_ok}")
    print(f"  skipped (already exists): {n_skip}")
    print(f"  failed: {n_fail}")
    if failures:
        print(f"\nFirst 10 failures:")
        for fn, err in failures[:10]:
            print(f"  - {fn}: {err}")
        sys.exit(1)
    print(f"[DONE] All images present in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download COCO val2017 subset images"
    )
    parser.add_argument(
        "--subset-json",
        default="data/coco_val/annotations/instances_val2017_subset500.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/coco_val/images",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel download workers (default 4, set 1 for sequential)",
    )
    args = parser.parse_args()

    download_all(args.subset_json, args.output_dir, max_workers=args.workers)


if __name__ == "__main__":
    main()
