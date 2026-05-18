#!/bin/bash
# Ewald-Allegro one-click installation script
# Usage: bash install.sh [env_name]
set -e

ENV_NAME="${1:-ewald-allegro}"

echo "============================================"
echo " Ewald-Allegro Installation Script"
echo " Environment: $ENV_NAME"
echo "============================================"

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo ""
echo "[1/5] Creating conda environment: $ENV_NAME ..."
conda create -n $ENV_NAME python=3.10 -y

# Activate
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch
echo ""
echo "[2/5] Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
echo ""
echo "[3/5] Installing core dependencies..."
pip install e3nn nequip ase scipy matplotlib

# Install this package
echo ""
echo "[4/5] Installing Ewald-Allegro..."
pip install -e .

# Verify
echo ""
echo "[5/5] Verifying installation..."
python -c "
import torch, e3nn, nequip, ase, scipy, matplotlib
print('  PyTorch:', torch.__version__)
print('  CUDA available:', torch.cuda.is_available())
print('  e3nn:', e3nn.__version__)
print('  ase:', ase.__version__)
print('  scipy:', scipy.__version__)
print('  matplotlib:', matplotlib.__version__)
print('  ✅ All dependencies installed successfully!')
"

echo ""
echo "============================================"
echo " Installation complete!"
echo ""
echo " Activate the environment:"
echo "   conda activate $ENV_NAME"
echo ""
echo " Quick test:"
echo "   python prepare_data.py --check-deps"
echo "   python test_model_forward.py"
echo "============================================"
