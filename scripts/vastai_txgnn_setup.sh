#!/bin/bash
# Automated Vast.ai TxGNN Setup Script
# Usage: ./vastai_txgnn_setup.sh <SSH_PORT> <SSH_HOST>
# Example: ./vastai_txgnn_setup.sh 16464 ssh3.vast.ai

set -e

PORT=${1:-16464}
HOST=${2:-ssh3.vast.ai}

echo "======================================"
echo "Vast.ai TxGNN Setup Script"
echo "======================================"
echo "Host: $HOST:$PORT"
echo ""

# Function to run SSH commands
run_ssh() {
    ssh -o StrictHostKeyChecking=no -p "$PORT" "root@$HOST" "$1"
}

echo "[1/5] Installing Python and pip..."
run_ssh 'apt-get update -qq && apt-get install -y -qq python3-pip git > /dev/null 2>&1 && echo "Done"'

echo "[2/5] Installing NumPy and Pandas (compatible versions)..."
run_ssh 'pip3 install "numpy<2.0" "pandas<2.0" --quiet 2>&1 | tail -1'

echo "[3/5] Installing PyTorch with CUDA 11.8..."
run_ssh 'pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet 2>&1 | tail -1'

echo "[4/5] Installing DGL 1.1.3 with CUDA 11.8..."
run_ssh 'pip3 install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu118/repo.html --quiet 2>&1 | tail -1'

echo "[5/5] Cloning and installing TxGNN..."
run_ssh 'rm -rf TxGNN 2>/dev/null; git clone https://github.com/mims-harvard/TxGNN.git 2>&1 | tail -1 && cd TxGNN && pip3 install -e . --quiet 2>&1 | tail -1'

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Verify with:"
echo "  ssh -p $PORT root@$HOST 'python3 -c \"from txgnn import TxGNN; print(\\\"TxGNN ready\\\")\"'"
echo ""
echo "To copy files:"
echo "  scp -P $PORT data/reference/txgnn_500epochs.pt root@$HOST:~/"
echo "  scp -P $PORT data/reference/everycure_gt_for_txgnn.json root@$HOST:~/"
