#!/bin/bash
# Ewald-Allegro dependency installer
# Usage: bash install_deps.sh [conda_env_name]
#
# If no environment name is given, it installs to the currently active conda env.
# If "ai4phy" or another name is given, it installs to that conda environment.

set -e

if [ -n "$1" ]; then
    ENV_NAME="$1"
    ENV_PATH="$(conda info --base 2>/dev/null)/envs/$ENV_NAME"
    if [ ! -d "$ENV_PATH" ]; then
        echo "Environment '$ENV_NAME' not found at $ENV_PATH"
        echo "Creating conda environment: $ENV_NAME"
        conda create -n $ENV_NAME python=3.10 -y
    fi
    PIP="$ENV_PATH/bin/pip"
    echo "Installing dependencies to conda environment: $ENV_NAME"
else
    PIP="pip"
    echo "Installing dependencies to current environment"
fi

echo "PIP: $PIP"
echo

# Core dependencies
$PIP install numpy scipy ase 2>&1 | tail -5

# e3nn
$PIP install e3nn 2>&1 | tail -5

# nequip
$PIP install nequip 2>&1 | tail -5

echo
echo "Installation complete!"
echo "Run tests: python allegro/test_env_check.py"
