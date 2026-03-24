#!/bin/bash

# run.sh
# this script runs the entire wheres waldo detection pipeline from start to finish
# it generates tiles, trains the model, and evaluates the results
#
# usage: bash run.sh
#
# what it does:
#   1. installs the required python packages
#   2. generates tiles from the raw images and creates train/val/test splits
#   3. trains the WaldoDetector model and saves checkpoints
#   4. runs the confidence test to check if the model learned properly
#   5. runs the full evaluation with metrics and visualizations
#
# the results will be saved to the results/ folder
# the trained model will be saved to the checkpoints/ folder
#
# requirements:
#   python 3.8+
#   a GPU is recommended but CPU works too (just slower)
#   the raw images should already be in data/raw/images/
#   the annotations should already be in data/raw/annotations/

echo "========================================"
echo "  Wheres Waldo Detection Pipeline"
echo "  COMP 4360 Machine Learning"
echo "========================================"
echo ""

# step 1: install dependencies
echo "step 1: installing dependencies..."
pip install -r requirements.txt
echo ""

# step 2: generate tiles from raw images
echo "step 2: generating tiles from raw images..."
python src/data_pipeline.py
echo ""

# step 3: train the model
echo "step 3: training the model..."
echo "this may take a while depending on your hardware"
python train.py
echo ""

# step 4: check confidence scores on test tiles
echo "step 4: checking confidence scores..."
python test_confidence.py
echo ""

# step 5: run full evaluation
echo "step 5: running evaluation..."
python evaluate.py
echo ""

echo "========================================"
echo "  Pipeline complete!"
echo "  check results/ folder for output images"
echo "  check checkpoints/ folder for saved model"
echo "========================================"
