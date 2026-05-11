import os
import glob
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

### python dashboard_app.py


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = "/Users/hobongs/Desktop/HoBong_study/26-1/탄중/temp/jetson_nano_edgeai/results/clean_v2_20260419_150621"
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "layer_runtime", "visualizations")

# Serve static images
app.mount("/images", StaticFiles(directory=VISUALIZATIONS_DIR), name="images")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/models")
def get_models():
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "consolidated_latency.csv"))
        models = df['model'].unique().tolist()
        return {"models": models}
    except Exception as e:
        print(e)
        return {"models": []}

@app.get("/api/images/{model_name}")
def get_images(model_name: str):
    search_pattern = os.path.join(VISUALIZATIONS_DIR, f"{model_name}_*.png")
    files = glob.glob(search_pattern)
    filenames = sorted([os.path.basename(f) for f in files])
    return {"images": filenames}

@app.get("/api/metrics/{model_name}")
def get_metrics(model_name: str):
    try:
        latency_df = pd.read_csv(os.path.join(BASE_DIR, "consolidated_latency.csv"))
        accuracy_df = pd.read_csv(os.path.join(BASE_DIR, "consolidated_accuracy.csv"))

        lat_model = latency_df[latency_df['model'] == model_name]
        acc_model = accuracy_df[accuracy_df['model'] == model_name]
        
        merged = pd.merge(lat_model, acc_model[['runtime', 'map_50', 'top1']], on='runtime', how='left')
        merged = merged.fillna("")
        
        cols = ['runtime', 'power_mode', 'mean_ms', 'memory_mb', 'power_avg_mw', 'energy_per_inference_mj', 'top1', 'map_50']
        available_cols = [c for c in cols if c in merged.columns]
        result = merged[available_cols].to_dict(orient='records')
        return {"metrics": result}
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return {"metrics": []}

if __name__ == "__main__":
    print("Starting FastAPI dashboard server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
