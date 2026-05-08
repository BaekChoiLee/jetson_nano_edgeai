import json, csv, os

CLEAN_DIR = "results/clean_v2_20260419_150621"
LAT_CSV = os.path.join(CLEAN_DIR, "consolidated_latency.csv")

r10w = json.load(open("results/shufflenet_v2_x1_0_10w_20260427_132852/results.json"))[0]
r5w = json.load(open("results/shufflenet_v2_x1_0_5w_20260427_132908/results.json"))[0]

CARBON_INTENSITY = 0.000475

def get_power(data):
    ts = data.get("tegrastats", {})
    cpu_avg = ts.get("pom_5v_cpu_average_mw", {}).get("mean", 0)
    gpu_avg = ts.get("pom_5v_gpu_average_mw", {}).get("mean", 0)
    cpu_max = ts.get("pom_5v_cpu_current_mw", {}).get("max", 0)
    gpu_max = ts.get("pom_5v_gpu_current_mw", {}).get("max", 0)
    return cpu_avg + gpu_avg, cpu_max + gpu_max

with open(LAT_CSV) as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    rows = list(reader)

updates = {
    ("MAXN", "tflite_cpu"): r10w,
    ("5W", "tflite_cpu"): r5w,
}

for i, row in enumerate(rows):
    if row["model"] != "shufflenet_v2_x1_0":
        continue
    key = (row["power_mode"], row["runtime"])
    
    if key in updates:
        d = updates[key]
        lat = d["latency"]
        pavg, pmax = get_power(d)
        lat_s = lat["mean"] / 1000.0
        energy_mj = pavg * lat_s
        co2_mg = (energy_mj / 3.6e6) * CARBON_INTENSITY * 1e6
        
        rows[i]["mean_ms"] = str(round(lat["mean"], 4))
        rows[i]["p50_ms"] = str(round(lat["p50"], 4))
        rows[i]["p95_ms"] = str(round(lat["p95"], 4))
        rows[i]["p99_ms"] = str(round(lat["p99"], 4))
        rows[i]["std_ms"] = str(round(lat["std"], 4))
        rows[i]["memory_mb"] = str(round(d["memory_mb"], 2))
        rows[i]["model_size_mb"] = str(round(d["model_size_mb"], 2))
        rows[i]["num_runs"] = str(d["num_runs"])
        rows[i]["power_avg_mw"] = str(round(pavg))
        rows[i]["power_max_mw"] = str(round(pmax))
        rows[i]["energy_per_inference_mj"] = str(round(energy_mj, 4))
        rows[i]["co2_per_inference_mg"] = str(round(co2_mg, 6))
        rows[i]["status"] = "ok"
        rows[i]["status_reason"] = ""
        rows[i]["error"] = ""
        rows[i]["final_class"] = "classification"
        print("Updated " + str(key) + ": mean=" + str(round(lat["mean"],2)) + "ms")
    elif row["runtime"] == "tflite_gpu":
        rows[i]["status_reason"] = "gpu_delegate_incompatible"
        rows[i]["error"] = "channel_shuffle unsupported by GPU delegate"
        print("Updated " + row["power_mode"] + "/tflite_gpu: gpu_delegate_incompatible")

with open(LAT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

ok = sum(1 for r in rows if r["status"] == "ok")
na = sum(1 for r in rows if r["status"] == "na")
print("Total: " + str(len(rows)) + " rows, ok=" + str(ok) + ", na=" + str(na))
