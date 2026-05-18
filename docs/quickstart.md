# Quick Start Guide

This guide walks through the complete Ewald-Allegro workflow.

## Prerequisites

- Python >= 3.10
- Conda (recommended) or pip
- NVIDIA GPU with CUDA (recommended, but CPU works)

## Installation

```bash
# Clone the repository
git clone https://github.com/xiaokang126/ewald-allegro.git
cd ewald-allegro

# Create environment
conda create -n ewald-allegro python=3.10
conda activate ewald-allegro

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install e3nn nequip ase scipy matplotlib

# Install this package
pip install -e .
```

Or use the automatic script:
```bash
bash install.sh
```

## Verify Installation

```bash
python prepare_data.py --check-deps
```

Expected output shows all dependencies ✓:

```
[OK] Python 3.10.x
[OK] PyTorch 2.x (CUDA available: True)
[OK] e3nn
[OK] nequip
[OK] allegro
[OK] ase
[OK] scipy
[OK] matplotlib
```

## Test Model Forward Pass

```bash
# Uses existing data in data/train.xyz
python test_model_forward.py
```

Expected output:
```
原子: 36 (12H2O)
模型参数: XXX,XXX
✅ 前向传播成功!
✅ 反向传播成功!
```

## Train a Model

```bash
python train.py
```

This trains the Ewald-Allegro model on the water AIMD trajectory.
The best checkpoint is saved to `data/model_best.pt`.

## Analyze Results

```bash
python analyze.py
```

Generates 6 diagnostic figures in `plot/`:
- `fig1_prediction_scatter.png` — Energy prediction scatter
- `fig2_error_distribution.png` — Error distribution histogram
- `fig3_distance_bucket_mae.png` — MAE vs intermolecular distance (key evidence)
- `fig4_error_vs_longrange.png` — Error vs Ewald long-range contribution
- `fig5_size_scaling.png` — System size scaling
- `fig6_charge_analysis.png` — Charge distribution + electroneutrality

## Data Preparation

### From VASP MD Output

Place VASP output directories under `../unloaded_data/`:

```
unloaded_data/
├── hot/       (OUTCAR + XDATCAR)
├── hot2/
├── hot3/
├── MD/
└── MD2/
```

Then run:
```bash
python prepare_data.py
```

### Custom extxyz Files

Place extxyz files directly:
```bash
mkdir -p data
cp your_data.xyz data/train.xyz
cp your_data.xyz data/test.xyz
```

## Training Configuration

Key hyperparameters in the model:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r_max` | 5.0 A | Short-range cutoff radius |
| `l_max` | 1 | Maximum spherical harmonics order |
| `num_layers` | 2 | Number of Allegro interaction layers |
| `num_scalar_features` | 64 | Number of scalar features |
| `num_tensor_features` | 32 | Number of tensor features |
| `ewald_alpha` | 0.35 | Ewald splitting parameter |
| `ewald_r_cut` | 8.0 A | Real-space Ewald cutoff |
| `ewald_grid` | (32,32,32) | Reciprocal-space FFT grid |
