#!/bin/bash
# Backward-compatible entry point.
# MacBook: bash run_pipeline_mac.sh
# Jetson Nano: bash run_pipeline_jetson.sh

echo "[Info] run_pipeline.sh delegates to the Jetson pipeline."
echo "[Info] For MacBook model conversion, run: bash run_pipeline_mac.sh"

bash run_pipeline_jetson.sh
