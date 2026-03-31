#!/bin/bash

# demo.sh
# quick demo that loads an existing checkpoint and runs evaluation
# WITHOUT retraining. this is the fastest way for a grader to see results
#
# what it does:
#   1. checks dependencies (via setup.sh)
#   2. generates tiles if they dont exist yet
#   3. loads the saved model checkpoint
#   4. runs full evaluation with all plots and metrics
#   5. runs demo detection on sample images
#
# usage: bash scripts/demo.sh
#
# requires: a trained checkpoint in checkpoints/best_model.pth
# outputs go to results/ folder, log saved to results/demo_log.txt

export PYTHONUNBUFFERED=1

# cd to project root (one level up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p results

exec > >(tee results/demo_log.txt) 2>&1

echo "========================================"
echo "  Wheres Waldo - Quick Demo"
echo "  (no training, evaluation only)"
echo "========================================"
echo ""
echo "date: $(date)"
echo ""

# figure out which python to use
PYTHON=python3
if ! command -v python3 &> /dev/null; then
    PYTHON=python
fi

# step 1: check dependencies
echo "========================================"
echo "  Step 1: Checking dependencies"
echo "========================================"
bash "$SCRIPT_DIR/setup.sh"
if [ $? -ne 0 ]; then
    echo "ERROR: dependency setup failed."
    exit 1
fi
echo ""

# step 2: check that a trained checkpoint exists
echo "========================================"
echo "  Step 2: Checking for trained model"
echo "========================================"
if [ ! -f "checkpoints/best_model.pth" ]; then
    echo "ERROR: no trained model found at checkpoints/best_model.pth"
    echo ""
    echo "you need to train the model first. options:"
    echo "  bash scripts/run.sh                    (full pipeline, trains + evaluates)"
    echo "  python src/train.py                   (just training)"
    echo "  python experiments/run_retinanet.py   (train via experiment framework)"
    exit 1
fi
echo "found checkpoint: checkpoints/best_model.pth"
echo ""

# step 3: generate tiles if they dont exist
echo "========================================"
echo "  Step 3: Checking tile data"
echo "========================================"
if [ -d "data/processed" ] && [ "$(ls -A data/processed 2>/dev/null)" ]; then
    echo "tiles already exist in data/processed/, skipping generation"
else
    echo "generating tiles from raw images..."
    $PYTHON src/data/data_pipeline.py
fi
echo ""

# step 4: model summary
echo "========================================"
echo "  Step 4: Model summary"
echo "========================================"
$PYTHON src/scripts/print_summary.py model
echo ""

# step 5: sanity check
echo "========================================"
echo "  Step 5: Sanity check"
echo "========================================"
$PYTHON src/scripts/sanity_check.py
echo ""

# step 6: full evaluation
echo "========================================"
echo "  Step 6: Evaluation"
echo "========================================"
$PYTHON src/evaluate.py
echo ""

# step 7: results summary
echo "========================================"
echo "  Step 7: Results"
echo "========================================"
$PYTHON src/scripts/print_summary.py results
echo ""

echo "========================================"
echo "  Demo complete!"
echo ""
echo "  outputs saved to results/ folder"
echo "  full log saved to results/demo_log.txt"
echo "========================================"
