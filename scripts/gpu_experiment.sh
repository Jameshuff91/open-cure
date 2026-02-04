#!/bin/bash
# GPU Experiment Runner for Vast.ai
# Usage: ./scripts/gpu_experiment.sh <experiment_script.py> [--keep]
#
# This script:
# 1. Provisions a Vast.ai GPU instance (RTX 3090)
# 2. Sets up TxGNN environment
# 3. Copies and runs your experiment
# 4. Retrieves results
# 5. Destroys the instance (unless --keep flag)
#
# Example:
#   ./scripts/gpu_experiment.sh scripts/txgnn_evaluate.py
#   ./scripts/gpu_experiment.sh scripts/txgnn_finetune.py --keep

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MIN_GPU_RAM=12  # GB
MIN_DISK=50     # GB
MAX_PRICE=0.50  # $/hr
RESULTS_DIR="data/analysis/gpu_results"

# Parse arguments
EXPERIMENT_SCRIPT="$1"
KEEP_INSTANCE=false
if [[ "$2" == "--keep" ]]; then
    KEEP_INSTANCE=true
fi

if [[ -z "$EXPERIMENT_SCRIPT" ]]; then
    echo -e "${RED}Error: No experiment script specified${NC}"
    echo "Usage: $0 <experiment_script.py> [--keep]"
    exit 1
fi

if [[ ! -f "$EXPERIMENT_SCRIPT" ]]; then
    echo -e "${RED}Error: Experiment script not found: $EXPERIMENT_SCRIPT${NC}"
    exit 1
fi

echo "======================================================================"
echo "  GPU Experiment Runner"
echo "======================================================================"
echo "  Experiment: $EXPERIMENT_SCRIPT"
echo "  Keep instance: $KEEP_INSTANCE"
echo ""

# Check vastai CLI
if ! command -v vastai &> /dev/null; then
    echo -e "${RED}Error: vastai CLI not found. Install with: pipx install vastai${NC}"
    exit 1
fi

# Check balance
echo -e "${YELLOW}[1/8] Checking Vast.ai balance...${NC}"
BALANCE=$(vastai show user 2>/dev/null | tail -1 | awk '{print $1}')
echo "  Balance: \$${BALANCE}"
if (( $(echo "$BALANCE < 1.0" | bc -l) )); then
    echo -e "${RED}Warning: Balance is low (\$${BALANCE}). Consider adding funds.${NC}"
fi

# Search for instance
echo -e "${YELLOW}[2/8] Searching for GPU instance...${NC}"
SEARCH_RESULT=$(vastai search offers \
    "gpu_ram >= ${MIN_GPU_RAM} disk_space >= ${MIN_DISK} reliability > 0.95 dph_total <= ${MAX_PRICE}" \
    -o 'dph_total' --limit 1 --raw 2>/dev/null)

if [[ -z "$SEARCH_RESULT" ]] || [[ "$SEARCH_RESULT" == "[]" ]]; then
    echo -e "${RED}No suitable GPU instances found. Try increasing MAX_PRICE.${NC}"
    exit 1
fi

OFFER_ID=$(echo "$SEARCH_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['id'])" 2>/dev/null)
GPU_NAME=$(echo "$SEARCH_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['gpu_name'])" 2>/dev/null)
PRICE=$(echo "$SEARCH_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d[0]['dph_total']:.3f}\")" 2>/dev/null)

echo "  Found: $GPU_NAME @ \$${PRICE}/hr (Offer ID: $OFFER_ID)"

# Create instance
echo -e "${YELLOW}[3/8] Creating instance...${NC}"
CREATE_RESULT=$(vastai create instance "$OFFER_ID" \
    --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel \
    --disk "$MIN_DISK" \
    --raw 2>/dev/null)

INSTANCE_ID=$(echo "$CREATE_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('new_contract'))" 2>/dev/null)

if [[ -z "$INSTANCE_ID" ]] || [[ "$INSTANCE_ID" == "None" ]]; then
    echo -e "${RED}Failed to create instance. Response: $CREATE_RESULT${NC}"
    exit 1
fi

echo "  Instance ID: $INSTANCE_ID"

# Cleanup function
cleanup() {
    if [[ "$KEEP_INSTANCE" == false ]] && [[ -n "$INSTANCE_ID" ]]; then
        echo -e "${YELLOW}Destroying instance $INSTANCE_ID...${NC}"
        vastai destroy instance "$INSTANCE_ID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for instance to be ready
echo -e "${YELLOW}[4/8] Waiting for instance to be ready...${NC}"
MAX_WAIT=300  # 5 minutes
WAITED=0
while [[ $WAITED -lt $MAX_WAIT ]]; do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('actual_status', 'unknown'))" 2>/dev/null)

    if [[ "$STATUS" == "running" ]]; then
        echo "  Instance is running!"
        break
    fi

    echo "  Status: $STATUS (waited ${WAITED}s)"
    sleep 10
    WAITED=$((WAITED + 10))
done

if [[ "$STATUS" != "running" ]]; then
    echo -e "${RED}Instance failed to start within ${MAX_WAIT}s${NC}"
    exit 1
fi

# Get SSH details
echo -e "${YELLOW}[5/8] Getting SSH connection details...${NC}"
sleep 5  # Give it a moment to fully initialize

SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)
# Parse: ssh -p PORT root@HOST
SSH_PORT=$(echo "$SSH_URL" | grep -oE '\-p [0-9]+' | awk '{print $2}')
SSH_HOST=$(echo "$SSH_URL" | grep -oE 'root@[^ ]+' | cut -d@ -f2)

if [[ -z "$SSH_PORT" ]] || [[ -z "$SSH_HOST" ]]; then
    echo -e "${RED}Failed to parse SSH details from: $SSH_URL${NC}"
    exit 1
fi

echo "  SSH: ssh -p $SSH_PORT root@$SSH_HOST"

# SSH function
run_ssh() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p "$SSH_PORT" "root@$SSH_HOST" "$1"
}

# Wait for SSH to be available
echo "  Waiting for SSH..."
SSH_READY=false
for i in {1..12}; do
    if run_ssh "echo 'SSH ready'" 2>/dev/null; then
        SSH_READY=true
        break
    fi
    sleep 10
done

if [[ "$SSH_READY" == false ]]; then
    echo -e "${RED}SSH connection failed${NC}"
    exit 1
fi

# Setup TxGNN
echo -e "${YELLOW}[6/8] Setting up TxGNN environment...${NC}"
run_ssh 'apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1'
run_ssh 'pip3 install "numpy<2.0" "pandas<2.0" --quiet 2>&1 | tail -1'
run_ssh 'pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet 2>&1 | tail -1'
run_ssh 'pip3 install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu118/repo.html --quiet 2>&1 | tail -1'
run_ssh 'rm -rf TxGNN 2>/dev/null; git clone https://github.com/mims-harvard/TxGNN.git 2>&1 | tail -1'
run_ssh 'cd TxGNN && pip3 install -e . --quiet 2>&1 | tail -1'

# Verify TxGNN
if ! run_ssh 'python3 -c "from txgnn import TxGNN; print(\"TxGNN ready\")"' 2>/dev/null; then
    echo -e "${RED}TxGNN installation failed${NC}"
    exit 1
fi
echo "  TxGNN installed successfully"

# Copy experiment files
echo -e "${YELLOW}[7/8] Copying experiment files and running...${NC}"
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$EXPERIMENT_SCRIPT" "root@$SSH_HOST:~/experiment.py"

# Copy any required data files
if [[ -f "data/reference/txgnn_500epochs.pt" ]]; then
    scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "data/reference/txgnn_500epochs.pt" "root@$SSH_HOST:~/"
fi
if [[ -f "data/analysis/zero_shot_benchmark.json" ]]; then
    scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "data/analysis/zero_shot_benchmark.json" "root@$SSH_HOST:~/"
fi

# Run experiment
echo "  Running experiment..."
run_ssh 'cd ~ && python3 experiment.py 2>&1' | tee /tmp/gpu_experiment_output.txt

# Collect results
echo -e "${YELLOW}[8/8] Collecting results...${NC}"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Try to copy any result files
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "root@$SSH_HOST:~/results*.json" "$RESULTS_DIR/" 2>/dev/null || true
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "root@$SSH_HOST:~/output*.json" "$RESULTS_DIR/" 2>/dev/null || true

# Save experiment output
cp /tmp/gpu_experiment_output.txt "$RESULTS_DIR/experiment_${TIMESTAMP}.log"

echo ""
echo "======================================================================"
echo -e "${GREEN}  Experiment Complete!${NC}"
echo "======================================================================"
echo "  Results saved to: $RESULTS_DIR/"
echo "  Log file: $RESULTS_DIR/experiment_${TIMESTAMP}.log"

if [[ "$KEEP_INSTANCE" == true ]]; then
    echo ""
    echo -e "${YELLOW}  Instance kept running (ID: $INSTANCE_ID)${NC}"
    echo "  SSH: ssh -p $SSH_PORT root@$SSH_HOST"
    echo "  To destroy: vastai destroy instance $INSTANCE_ID"
    trap - EXIT  # Remove cleanup trap
else
    echo ""
    echo "  Instance will be destroyed..."
fi
