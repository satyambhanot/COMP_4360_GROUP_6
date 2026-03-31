#!/bin/bash

# run_all_experiments.sh
# trains and evaluates all model architectures, runs ablation study,
# and generates a comparison report at the end
#
# experiments:
#   1. RetinaNet ResNet50 FPN (our best model)
#   2. Pretrained SSD300 VGG16
#   3. Custom SSD MobileNetV2
#   4. YOLOv8 Nano (state of the art comparison)
#   5. Preprocessing ablation (RGB only, RGB+HSV, RGB+HSV+Sobel)
#   6. Comparison report from all results
#
# usage: bash scripts/run_all_experiments.sh
#
# results saved to saved_results/<model_name>/ for each experiment
# final comparison charts saved to saved_results/report/

export PYTHONUNBUFFERED=1

# cd to project root (one level up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========================================"
echo "  Wheres Waldo - All Experiments"
echo "========================================"
echo ""
echo "date: $(date)"
echo ""

# figure out which python to use
PYTHON=python3
if ! command -v python3 &> /dev/null; then
    PYTHON=python
fi

# check dependencies first
echo "========================================"
echo "  Checking dependencies"
echo "========================================"
bash "$SCRIPT_DIR/setup.sh"
if [ $? -ne 0 ]; then
    echo "ERROR: dependency setup failed."
    exit 1
fi
echo ""

# generate tiles if they dont exist
if [ ! -d "data/processed" ] || [ -z "$(ls -A data/processed 2>/dev/null)" ]; then
    echo "========================================"
    echo "  Generating tiles"
    echo "========================================"
    $PYTHON src/data/data_pipeline.py
    echo ""
fi

FAILED=()

# experiment 1: RetinaNet
echo "========================================"
echo "  Experiment 1/5: RetinaNet"
echo "========================================"
$PYTHON experiments/run_retinanet.py
if [ $? -ne 0 ]; then FAILED+=("RetinaNet"); fi
echo ""

# experiment 2: Pretrained SSD300
echo "========================================"
echo "  Experiment 2/5: Pretrained SSD300"
echo "========================================"
$PYTHON experiments/run_pretrained_ssd.py
if [ $? -ne 0 ]; then FAILED+=("Pretrained SSD300"); fi
echo ""

# experiment 3: Custom SSD
echo "========================================"
echo "  Experiment 3/5: Custom SSD"
echo "========================================"
$PYTHON experiments/run_custom_ssd.py
if [ $? -ne 0 ]; then FAILED+=("Custom SSD"); fi
echo ""

# experiment 4: YOLOv8
echo "========================================"
echo "  Experiment 4/5: YOLOv8"
echo "========================================"
$PYTHON experiments/run_yolo.py
if [ $? -ne 0 ]; then FAILED+=("YOLOv8"); fi
echo ""

# experiment 5: Preprocessing ablation
echo "========================================"
echo "  Experiment 5/5: Ablation Study"
echo "========================================"
$PYTHON experiments/run_ablation.py
if [ $? -ne 0 ]; then FAILED+=("Ablation"); fi
echo ""

# generate comparison report
echo "========================================"
echo "  Generating Comparison Report"
echo "========================================"
$PYTHON experiments/generate_report.py
echo ""

# summary
echo "========================================"
echo "  All Experiments Complete!"
echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  All 5 experiments finished successfully."
else
    echo "  WARNING: ${#FAILED[@]} experiment(s) had errors:"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo ""
echo "  Results saved to:"
echo "    saved_results/retinanet/"
echo "    saved_results/pretrained_ssd/"
echo "    saved_results/custom_ssd/"
echo "    saved_results/yolo/"
echo "    saved_results/ablation/"
echo "    saved_results/report/"
echo "========================================"
