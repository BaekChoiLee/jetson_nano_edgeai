import os
import glob
import socket
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

### python dashboard_app.py


def find_available_port(preferred_port: int, host: str = "127.0.0.1") -> int:
    for port in range(preferred_port, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No available port from {preferred_port} to {preferred_port + 19}")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.environ.get(
    "DASHBOARD_RESULTS_DIR",
    os.path.join(PROJECT_DIR, "my_results", "visualizations"),
)
VISUALIZATIONS_DIR = BASE_DIR
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

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
        models = sorted(df['model'].dropna().unique().tolist())
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
        accuracy_path = os.path.join(BASE_DIR, "consolidated_accuracy.csv")
        accuracy_df = pd.read_csv(accuracy_path) if os.path.exists(accuracy_path) else pd.DataFrame()

        lat_model = latency_df[latency_df['model'] == model_name]
        if not accuracy_df.empty and "model" in accuracy_df.columns:
            acc_model = accuracy_df[accuracy_df['model'] == model_name]
            acc_cols = [c for c in ['runtime', 'map_50', 'top1'] if c in acc_model.columns]
            merged = pd.merge(lat_model, acc_model[acc_cols], on='runtime', how='left') if acc_cols else lat_model
        else:
            merged = lat_model
        merged = merged.fillna("")
        
        cols = ['runtime', 'power_mode', 'mean_ms', 'memory_mb', 'power_avg_mw', 'energy_per_inference_mj', 'top1', 'map_50']
        available_cols = [c for c in cols if c in merged.columns]
        result = merged[available_cols].to_dict(orient='records')
        return {"metrics": result}
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return {"metrics": []}

@app.get("/api/source")
def get_source():
    return {"base_dir": BASE_DIR, "visualizations_dir": VISUALIZATIONS_DIR}

if __name__ == "__main__":
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    preferred_port = int(os.environ.get("DASHBOARD_PORT", "8001"))
    port = find_available_port(preferred_port, host=host)
    print("Starting FastAPI dashboard server...")
    print(f"Dashboard data: {BASE_DIR}")
    if port != preferred_port:
        print(f"Port {preferred_port} is busy; using {port} instead.")
    print(f"Open: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
