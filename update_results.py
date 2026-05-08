#!/usr/bin/env python3
"""
Update consolidated_summary.json with fix results:
  1. SSD ORT CUDA: mAP50 0.4027 → 0.2733 (fixed: now uses raw ONNX, same as other runtimes)
  2. SSD ncnn: mark as known limitation (onnx→ncnn conversion issue)
"""
import json, os, copy

BASE = os.path.expanduser("~/jetson-benchmark")
LATEST = os.path.join(BASE, "results/clean_v2_20260419_150621/consolidated_summary.json")
OUTPUT = os.path.join(BASE, "results/clean_v2_20260419_150621/consolidated_summary_fixed.json")

with open(LATEST) as f:
    data = json.load(f)

# Fix 1: Update ORT CUDA mAP
ssd = data["matrix"]["ssd_mobilenet_v2"]
if "onnxrt_cuda" in ssd:
    old_map = ssd["onnxrt_cuda"].get("accuracy_map_50")
    ssd["onnxrt_cuda"]["accuracy_map_50"] = 0.2733
    ssd["onnxrt_cuda"]["accuracy_map_50_95"] = 0.0944
    ssd["onnxrt_cuda"]["accuracy_note"] = "Fixed: uses raw ONNX (no built-in NMS) for fair comparison"
    print(f"ORT CUDA mAP50: {old_map} -> 0.2733")

# Fix 2: Annotate ncnn as known limitation
for rt in ["ncnn_cpu", "ncnn_vulkan"]:
    if rt in ssd:
        ssd[rt]["accuracy_note"] = (
            "Known limitation: onnx2ncnn conversion produces raw box encodings "
            "with mismatched scale factors. The ncnn model lacks proper post-processing "
            "alignment with the TF OD API anchor configuration, resulting in collapsed mAP. "
            "Latency data remains valid."
        )
        print(f"ncnn {rt}: annotated as known limitation")

# Save
with open(OUTPUT, "w") as f:
    json.dump(data, f, indent=2)
print(f"\nSaved to {OUTPUT}")

# Also update the accuracy CSV
acc_csv = os.path.join(BASE, "results/clean_v2_20260419_150621/consolidated_accuracy.csv")
if os.path.exists(acc_csv):
    with open(acc_csv) as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if "ssd_mobilenet_v2" in line and "onnxrt_cuda" in line:
            # Replace mAP values
            parts = line.strip().split(",")
            # CSV format: error,final_class,map_50,map_50_95,model,n,runtime,...
            if len(parts) >= 4:
                parts[2] = "0.2733"
                parts[3] = "0.0944"
                line = ",".join(parts) + "\n"
                print(f"Updated accuracy CSV line for onnxrt_cuda")
        new_lines.append(line)
    with open(acc_csv, "w") as f:
        f.writelines(new_lines)
    print("Updated accuracy CSV")
