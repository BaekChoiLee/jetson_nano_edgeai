import json, csv, os

acc_csv = "results/clean_v2_20260419_150621/consolidated_accuracy.csv"
new_acc = json.load(open("results/shufflenet_v2_x1_0_classification.json"))

with open(acc_csv) as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    rows = list(reader)

tflite_entries = [e for e in new_acc if "tflite" in e.get("runtime", "")]
for te in tflite_entries:
    already = any(r["model"] == "shufflenet_v2_x1_0" and r["runtime"] == te["runtime"] for r in rows)
    if not already:
        row = {h: "" for h in headers}
        row["model"] = "shufflenet_v2_x1_0"
        row["runtime"] = te["runtime"]
        row["top1"] = str(te["top1"])
        row["top5"] = str(te["top5"])
        row["n"] = str(te.get("n", 500))
        row["task"] = "classification"
        row["final_class"] = "classification"
        row["status_reason"] = te.get("status_reason", "ok")
        row["error"] = te.get("error", "")
        rows.append(row)
        print("Added: " + te["runtime"] + " top1=" + str(te["top1"]))
    else:
        for r in rows:
            if r["model"] == "shufflenet_v2_x1_0" and r["runtime"] == te["runtime"]:
                r["top1"] = str(te["top1"])
                r["top5"] = str(te["top5"])
                print("Updated: " + te["runtime"] + " top1=" + str(te["top1"]))
                break

with open(acc_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

total = len(rows)
shuf = sum(1 for r in rows if r["model"] == "shufflenet_v2_x1_0")
print("Total: " + str(total) + ", ShuffleNet: " + str(shuf))
