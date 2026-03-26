#!/bin/bash

# run_all.sh
# runs everything needed for the final report in one shot
# this includes the main training, YOLOv8 comparison, ablation studies,
# and comparison chart generation
#
# estimated total time on GTX 1080 Ti: ~3-4 hours
#   step 1 (RetinaNet training + eval): ~90 min
#   step 2 (YOLOv8 comparison): ~15 min
#   step 3 (ablation studies): ~90 min
#   step 4 (comparison charts): ~10 sec
#
# usage: bash run_all.sh
#
# make sure you have already run:
#   python src/dataset_creation_tool.py
#   git add data/raw/images/gen_* data/raw/annotations/gen_*
#   git commit -m "Add 200 generated waldo training images"
#
# all results will be in the results/ folder when done

echo "========================================"
echo "  Running Complete Pipeline + All Experiments"
echo "  estimated time: 3-4 hours on GTX 1080 Ti"
echo "========================================"
echo ""
echo "date: $(date)"
echo ""

# figure out which python command to use
PYTHON=python3
if ! command -v python3 &> /dev/null; then
    PYTHON=python
fi

# step 1: main pipeline (RetinaNet)
# this runs tiling, training, and full evaluation
echo "========================================"
echo "  Step 1: Main Pipeline (RetinaNet)"
echo "  this is our primary model for the report"
echo "========================================"
echo ""
bash run.sh

# save the retinanet results with a model specific name
# so they dont get overwritten when we run other models
if [ -f results/evaluation_results.json ]; then
    cp results/evaluation_results.json results/retinanet_results.json
    echo "saved RetinaNet results to results/retinanet_results.json"
fi
echo ""

# step 2: YOLOv8 comparison
# standalone experiment using a completely different architecture
# this gives us a state of the art reference point for the report
echo "========================================"
echo "  Step 2: YOLOv8 Comparison"
echo "  training YOLOv8 on the same data"
echo "========================================"
echo ""
$PYTHON train_yolo.py
echo ""

# step 3: ablation studies
# trains 3 preprocessing variants (full, HSV only, RGB only)
# and compares them to show each component helps
echo "========================================"
echo "  Step 3: Ablation Studies"
echo "  testing if HSV mask and Sobel edges actually help"
echo "========================================"
echo ""
$PYTHON evaluate_ablation.py
echo ""

# step 4: comparison charts
# loads all the results and generates side by side comparison charts
echo "========================================"
echo "  Step 4: Generating Comparison Charts"
echo "========================================"
echo ""
$PYTHON compare_results.py
echo ""

# final summary
echo "========================================"
echo "  All Done!"
echo ""
echo "  results/                          main evaluation outputs"
echo "  results/retinanet_results.json    RetinaNet metrics"
echo "  results/yolo/                     YOLOv8 results"
echo "  results/ablation/                 ablation study charts"
echo "  results/comparison/               cross model comparison"
echo ""
echo "  total time: started at the top of this log"
echo "  finished at: $(date)"
echo "========================================"
