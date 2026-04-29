#!/bin/bash
# Deploy benchmark project to Jetson Nano and run setup
# Usage: bash deploy_to_jetson.sh <JETSON_IP>
# Example: bash deploy_to_jetson.sh 192.168.0.42
set -e

JETSON_IP="${1:?Usage: bash deploy_to_jetson.sh <JETSON_IP>}"
JETSON_USER="jetson"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Jetson Nano Benchmark Deployment"
echo "============================================"
echo "  Target: ${JETSON_USER}@${JETSON_IP}"
echo "  Source: ${PROJECT_DIR}"
echo "============================================"
echo ""

# Step 1: Test SSH connection
echo "[1/4] Testing SSH connection..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes "${JETSON_USER}@${JETSON_IP}" "echo 'SSH OK'" 2>/dev/null; then
    echo "  SSH connection successful (key-based auth)."
else
    echo "  Key-based auth not set up. Will use password auth."
    echo "  Default password: jetson"
    echo ""
    echo "  TIP: Set up SSH key to avoid password prompts:"
    echo "    ssh-copy-id ${JETSON_USER}@${JETSON_IP}"
    echo ""
fi

# Step 2: Transfer project
echo "[2/4] Transferring benchmark project..."
scp -r "${PROJECT_DIR}/" "${JETSON_USER}@${JETSON_IP}:~/jetson-benchmark/"
echo "  Transfer complete."

# Step 3: Run setup on Jetson
echo "[3/4] Running environment setup on Jetson..."
echo "  This will take 10-20 minutes (package installation)."
echo ""
ssh -t "${JETSON_USER}@${JETSON_IP}" "cd ~/jetson-benchmark && bash setup/setup_env.sh"

# Step 4: Verify environment
echo ""
echo "[4/4] Verifying environment..."
ssh -t "${JETSON_USER}@${JETSON_IP}" "cd ~/jetson-benchmark && python3 setup/verify_env.py"

echo ""
echo "============================================"
echo "  Deployment Complete!"
echo "============================================"
echo ""
echo "  Next steps (on Jetson):"
echo "    ssh ${JETSON_USER}@${JETSON_IP}"
echo "    cd ~/jetson-benchmark"
echo "    bash run_all.sh --dry-run    # verify first"
echo "    bash run_all.sh              # full benchmark"
echo ""
