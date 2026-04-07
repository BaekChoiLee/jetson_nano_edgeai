import os, glob, csv, json

data = []
# Grab all summaries
for f in sorted(glob.glob("jetson-benchmark/results/**/summary.csv", recursive=True)):
    with open(f, "r") as file:
        reader = csv.DictReader(file)
        data.extend(list(reader))

# Deduplicate keeping the LAST (most recent) entry
deduped = {}
for d in data:
    model = d["model"].split(".")[0] # strip .onnx, .tflite, .engine
    key = f"{model}_{d['runtime']}_{d.get('power_mode', '10w')}"
    deduped[key] = d

final_data = list(deduped.values())

with open("jetson-benchmark/all_summaries_latest.json", "w") as out:
    json.dump(final_data, out, indent=2)

print(f"Compiled {len(final_data)} unique benchmark configurations!")
