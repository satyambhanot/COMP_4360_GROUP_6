#!/bin/bash

# run.sh
# this script runs the entire wheres waldo detection pipeline from start to finish
# it generates tiles, trains the model, and evaluates the results
#
# usage: bash run.sh
#
# the results will be saved to the results/ folder
# the complete log will be saved to results/run_log.txt automatically
# the trained model will be saved to the checkpoints/ folder
#
# to change the number of training epochs edit NUM_EPOCHS in src/config.py

# make sure results directory exists before we start logging
mkdir -p results

# force python to not buffer its output
# without this, print statements dont show up until the buffer is full
# which makes it look like the script is stuck when training is actually running
export PYTHONUNBUFFERED=1

# save the complete output to a log file automatically
# tee sends output to both the screen and the file at the same time
exec > >(tee results/run_log.txt) 2>&1

echo "========================================"
echo "  Wheres Waldo Detection Pipeline"
echo "  COMP 4360 Machine Learning"
echo "  University of Manitoba"
echo "========================================"
echo ""
echo "date: $(date)"
echo ""

# show hardware info so the grader knows what we trained on
echo "hardware info:"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not available')"
    echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'not available')"
else
    echo "  GPU: not available (training will use CPU)"
fi
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"
echo ""

# figure out which python command to use
PYTHON=python3
if ! command -v python3 &> /dev/null; then
    PYTHON=python
fi

# make sure results directory exists for the log file
mkdir -p results

# step 1: install dependencies
echo "========================================"
echo "  Step 1: Installing dependencies"
echo "========================================"
pip install -r requirements.txt 2>/dev/null || pip3 install -r requirements.txt
echo ""

# step 2: generate tiles from raw images
echo "========================================"
echo "  Step 2: Generating tiles from raw images"
echo "========================================"
echo "this tiles images with 4 grid offsets and generates random crops around waldo"
echo ""
# clean out old tiles first so we dont have leftover 64x64 and 128x128 from previous runs
if [ -d "data/processed" ]; then
    echo "cleaning old tiles..."
    rm -rf data/processed
fi
$PYTHON src/data_pipeline.py
echo ""

# step 3: show model summary before training
echo "========================================"
echo "  Step 3: Model summary"
echo "========================================"
$PYTHON src/print_summary.py model
echo ""

# step 4: clean old checkpoints so we start fresh
echo "========================================"
echo "  Step 4: Cleaning old checkpoints"
echo "========================================"
if [ -f checkpoints/best_model.pth ]; then
    echo "removing old best_model.pth"
    rm checkpoints/best_model.pth
fi
if [ -f checkpoints/last_model.pth ]; then
    echo "removing old last_model.pth"
    rm checkpoints/last_model.pth
fi
if [ -f checkpoints/training_history.json ]; then
    echo "removing old training_history.json"
    rm checkpoints/training_history.json
fi
echo "checkpoints directory is clean"
echo ""

# step 5: train the model
echo "========================================"
echo "  Step 5: Training the model"
echo "========================================"
echo "this may take a while depending on your hardware"
echo "on a GTX 1080 Ti: about 3 seconds per epoch"
echo ""
$PYTHON train.py
echo ""

# step 6: quick sanity check on the trained model
echo "========================================"
echo "  Step 6: Sanity check on trained model"
echo "========================================"
echo "checking if the model learned to distinguish waldo from background..."
echo ""
$PYTHON src/sanity_check.py
echo ""

# step 7: run full evaluation
echo "========================================"
echo "  Step 7: Full evaluation"
echo "========================================"
$PYTHON evaluate.py
echo ""

# step 8: quick summary of the run
echo "========================================"
echo "  Step 8: Run summary"
echo "========================================"
$PYTHON src/print_summary.py results
echo ""

echo "========================================"
echo "  Pipeline complete!"
echo ""
echo "  outputs:"
echo "    checkpoints/best_model.pth           trained model weights"
echo "    checkpoints/training_history.json    loss per epoch"
echo "    results/evaluation_results.json      all metrics in json format"
echo "    results/training_curves.png          train vs val loss plots"
echo "    results/confidence_distribution.png  waldo vs background confidence"
echo "    results/tp_*.png                     true positive tile examples"
echo "    results/fp_*.png                     false positive tile examples"
echo "    results/fn_*.png                     false negative tile examples"
echo "    results/demo_*.png                   waldo detection on full images"
echo "    results/run_log.txt                 complete log of this run"
echo "========================================"
