#!/bin/bash
# Ewald-Allegro Water Example
# Demonstrates the complete workflow: test forward → train → analyze
set -e

EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$EXAMPLE_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

echo "=== Ewald-Allegro Water Example ==="
echo "Project dir: $PROJECT_DIR"
echo ""

# 1. Check dependencies
echo "--- Step 1: Checking dependencies ---"
python prepare_data.py --check-deps
echo ""

# 2. Copy example data
echo "--- Step 2: Setting up example data ---"
mkdir -p data
cp "$EXAMPLE_DIR/example.xyz" data/train.xyz
cp "$EXAMPLE_DIR/example.xyz" data/test.xyz
echo "Data ready: 2 frames (12 H2O each)"
echo ""

# 3. Test model forward + backward
echo "--- Step 3: Testing model forward/backward ---"
python test_model_forward.py
echo ""

# 4. Quick training (just 5 epochs for demo)
echo "--- Step 4: Quick training (5 epochs) ---"
python train.py --epochs 5 2>&1 | tail -20
echo ""

echo "=== Water example completed! ==="
echo "Run 'python analyze.py' to generate diagnostic figures."
