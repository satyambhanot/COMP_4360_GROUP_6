#!/bin/bash

# run.sh
# runs the wheres waldo detection pipeline from start to finish
# generates tiles, trains the current MODEL_TYPE from config.py, and evaluates
#
# usage: bash run.sh
#
# results go to the results/ folder, log saved to results/run_log.txt
# trained model saved to checkpoints/ folder
# to change the model type or epochs, edit src/config.py

# force python to flush output immediately so progress shows through tee
export PYTHONUNBUFFERED=1

# make sure results directory exists before we start logging
mkdir -p results

# save the complete output to a log file automatically
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

# step 1: install dependencies
echo "========================================"
echo "  Step 1: Installing dependencies"
echo "========================================"
pip install -r requirements.txt 2>/dev/null || pip3 install -r requirements.txt
echo ""

# step 2: generate tiles from raw images
echo "========================================"
echo "  Step 2: Generating tiles"
echo "========================================"
echo "tiling images with 4 grid offsets + random crop augmentation"
echo ""
# clean old tiles so we start fresh every run
if [ -d "data/processed" ]; then
    echo "cleaning old tiles..."
    rm -rf data/processed
fi
$PYTHON src/data_pipeline.py
echo ""

# step 3: show model summary
echo "========================================"
echo "  Step 3: Model summary"
echo "========================================"
$PYTHON src/print_summary.py model
echo ""

# step 4: clean old checkpoints
echo "========================================"
echo "  Step 4: Cleaning old checkpoints"
echo "========================================"
rm -f checkpoints/best_model.pth checkpoints/last_model.pth checkpoints/training_history.json
echo "checkpoints directory is clean"
echo ""

# step 5: train the model
echo "========================================"
echo "  Step 5: Training"
echo "========================================"
echo "this takes a while. RetinaNet at 512x512 is about 5-7 min per epoch"
echo ""
$PYTHON train.py
echo ""

# step 6: sanity check
echo "========================================"
echo "  Step 6: Sanity check"
echo "========================================"
$PYTHON src/sanity_check.py
echo ""

# step 7: full evaluation
echo "========================================"
echo "  Step 7: Evaluation"
echo "========================================"
$PYTHON evaluate.py
echo ""

# step 8: summary
echo "========================================"
echo "  Step 8: Summary"
echo "========================================"
$PYTHON src/print_summary.py results
echo ""

echo "========================================"
echo "  Pipeline complete!"
echo ""
echo "  outputs:"
echo "    checkpoints/best_model.pth              trained model weights"
echo "    checkpoints/training_history.json       loss per epoch"
echo "    results/evaluation_results.json         all metrics as json"
echo "    results/confusion_matrix.png            TP/FP/FN/TN heatmap"
echo "    results/precision_recall_curve.png      PR curve"
echo "    results/f1_vs_threshold.png             F1 at different thresholds"
echo "    results/roc_curve.png                   ROC curve with AUC"
echo "    results/confidence_distribution.png     positive vs negative scores"
echo "    results/per_image_summary.png           per image detection table"
echo "    results/training_curves.png             train vs val loss"
echo "    results/tp_*.png fp_*.png fn_*.png      sample tile detections"
echo "    results/demo_*.png                      full image detection"
echo "    results/run_log.txt                     this complete log"
echo "========================================"
