#!/bin/bash

# setup.sh
# creates a virtual environment and installs dependencies for the wheres waldo pipeline
#
# usage: bash scripts/setup.sh
#        bash scripts/setup.sh --cuda   # for GPU support (installs PyTorch with CUDA 12.6)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "========================================"
echo "  Wheres Waldo - Dependency Setup"
echo "  COMP 4360 Machine Learning"
echo "========================================"
echo ""

# create venv if it doesn't exist
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/.venv"
else
    echo "Virtual environment already exists."
fi

# activate venv
source "$PROJECT_ROOT/.venv/bin/activate"
echo "Using Python: $(python3 --version) ($(which python3))"
echo ""

# upgrade pip
pip install --upgrade pip

# install dependencies
if [ "$1" = "--cuda" ]; then
    echo ""
    echo "Installing with CUDA 12.6 support (PyTorch from pytorch.org)..."
    pip install -r "$PROJECT_ROOT/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu126
else
    echo ""
    echo "Installing CPU-only dependencies from PyPI..."
    echo "(Use 'bash scripts/setup.sh --cuda' for GPU support)"
    pip install -r "$PROJECT_ROOT/requirements.txt"
fi

echo ""
echo "========================================"
echo "  Setup complete!"
echo "  Activate the environment:  source .venv/bin/activate"
echo "  Run the pipeline with:     bash scripts/run.sh"
echo "========================================"
