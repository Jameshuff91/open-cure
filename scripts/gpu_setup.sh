#!/bin/bash
# GPU Setup Script for TxGNN Fine-tuning
# Run this on the Vast.ai instance

set -e

echo "=============================================="
echo "Setting up TxGNN Fine-tuning Environment"
echo "=============================================="

# Update system
apt-get update -qq

# Install Python packages
echo "Installing Python dependencies..."
pip3 install --quiet "numpy<2.0" "pandas<2.0" scikit-learn tqdm loguru

# Install PyTorch with CUDA 11.8
echo "Installing PyTorch..."
pip3 install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install DGL (critical for TxGNN)
echo "Installing DGL..."
pip3 install --quiet dgl==1.1.3 -f https://data.dgl.ai/wheels/cu118/repo.html

# Clone and install TxGNN
echo "Cloning TxGNN..."
if [ ! -d "TxGNN" ]; then
    git clone https://github.com/mims-harvard/TxGNN.git
fi
cd TxGNN && pip3 install --quiet -e . && cd ..

# Verify CUDA
echo ""
echo "=============================================="
echo "Verifying CUDA..."
echo "=============================================="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Verify TxGNN
echo ""
echo "Verifying TxGNN..."
python3 -c "from txgnn import TxData, TxGNN; print('TxGNN imported successfully')"

echo ""
echo "=============================================="
echo "Setup complete! Ready for fine-tuning."
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Transfer files: scp -P PORT local_file root@ssh1.vast.ai:~/"
echo "2. Run fine-tuning: python3 finetune_txgnn_everycure.py --epochs 100"
