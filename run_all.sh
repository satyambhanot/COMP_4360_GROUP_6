#!/bin/bash

# run_all.sh
# runs everything needed for the final report in one shot
# this includes the main training, YOLOv8 comparison, ablation studies,
# and comparison chart generation
#
# estimated total time on GTX 1080 Ti: ~2 hours
#   step 1 (RetinaNet training + eval): ~45-60 min
#   step 2 (YOLOv8 comparison): ~15 min
#   step 3 (ablation studies): ~45-60 min
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
echo "  estimated time: ~2 hours on GTX 1080 Ti"
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
echo "  Step 1/4: Main Pipeline (RetinaNet)"
echo "  this is our primary model for the report"
echo "  started at: $(date)"
echo "========================================"
echo ""
STEP1_START=$(date +%s)
bash run.sh
STEP1_END=$(date +%s)
STEP1_TIME=$(( (STEP1_END - STEP1_START) / 60 ))
echo ""
echo "  step 1 completed in ${STEP1_TIME} minutes"

# save the retinanet results with a model specific name
# so they dont get overwritten when we run other models
if [ -f results/evaluation_results.json ]; then
    cp results/evaluation_results.json results/retinanet_results.json
    echo "  saved RetinaNet results to results/retinanet_results.json"
fi
echo ""

# step 2: YOLOv8 comparison
# standalone experiment using a completely different architecture
# this gives us a state of the art reference point for the report
echo "========================================"
echo "  Step 2/4: YOLOv8 Comparison"
echo "  training YOLOv8 on the same data"
echo "  started at: $(date)"
echo "========================================"
echo ""
STEP2_START=$(date +%s)
$PYTHON train_yolo.py
STEP2_END=$(date +%s)
STEP2_TIME=$(( (STEP2_END - STEP2_START) / 60 ))
echo ""
echo "  step 2 completed in ${STEP2_TIME} minutes"
echo ""

# step 3: ablation studies
# trains 3 preprocessing variants (full, HSV only, RGB only)
# and compares them to show each component helps
echo "========================================"
echo "  Step 3/4: Ablation Studies"
echo "  testing if HSV mask and Sobel edges actually help"
echo "  started at: $(date)"
echo "========================================"
echo ""
STEP3_START=$(date +%s)
$PYTHON evaluate_ablation.py
STEP3_END=$(date +%s)
STEP3_TIME=$(( (STEP3_END - STEP3_START) / 60 ))
echo ""
echo "  step 3 completed in ${STEP3_TIME} minutes"
echo ""

# step 4: comparison charts
# loads all the results and generates side by side comparison charts
echo "========================================"
echo "  Step 4/4: Generating Comparison Charts"
echo "========================================"
echo ""
$PYTHON compare_results.py
echo ""

# final summary
echo "========================================"
echo "  All Done!"
echo ""
TOTAL_TIME=$(( STEP1_TIME + STEP2_TIME + STEP3_TIME ))
echo "  time breakdown:"
echo "    step 1 (RetinaNet):    ${STEP1_TIME} min"
echo "    step 2 (YOLOv8):      ${STEP2_TIME} min"
echo "    step 3 (ablation):    ${STEP3_TIME} min"
echo "    total:                ${TOTAL_TIME} min"
echo ""
echo "  outputs:"
echo "    results/                          main evaluation outputs"
echo "    results/retinanet_results.json    RetinaNet metrics"
echo "    results/yolo/                     YOLOv8 results"
echo "    results/ablation/                 ablation study charts"
echo "    results/comparison/               cross model comparison"
echo ""
echo "  finished at: $(date)"
echo "========================================"
