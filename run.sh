#!/bin/bash

# run.sh
# this script runs the entire wheres waldo detection pipeline from start to finish
# it generates tiles, trains the model, and evaluates the results
#
# usage: bash run.sh 2>&1 | tee results/run_log.txt
# or just: bash run.sh
#
# the results will be saved to the results/ folder
# the trained model will be saved to the checkpoints/ folder
#
# to change the number of training epochs edit NUM_EPOCHS in src/config.py
# default is 50 epochs, for better results try 500

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
echo "this reads 19 raw wheres waldo images and chops them into tiles"
echo "at three resolutions (64x64, 128x128, 256x256)"
echo ""
$PYTHON src/data_pipeline.py
echo ""

# step 3: show model summary before training
echo "========================================"
echo "  Step 3: Model summary"
echo "========================================"
$PYTHON -c "
import sys; sys.path.insert(0, 'src')
from model import WaldoDetector
model = WaldoDetector()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = total - trainable
print(f'model architecture: WaldoDetector')
print(f'  preprocessing: HSV mask + Sobel edge detection on HSV')
print(f'  backbone: MobileNetV2 (first 11 layers, frozen)')
print(f'  detection heads: 5 SSD blocks')
print(f'  anchors per location: 4')
print(f'  total anchors: 1364')
print(f'')
print(f'parameters:')
print(f'  total:     {total:,}')
print(f'  trainable: {trainable:,} (1x1 conv + SSD heads)')
print(f'  frozen:    {frozen:,} (MobileNet backbone)')
"
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
$PYTHON -c "
import sys, os; sys.path.insert(0, 'src')
import torch
import torch.nn.functional as F
from model import WaldoDetector
from data_pipeline import get_dataloader
from config import CHECKPOINTS_DIR

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
model = WaldoDetector().to(device)
cp = torch.load(os.path.join(CHECKPOINTS_DIR, 'best_model.pth'), map_location=device)
model.load_state_dict(cp['model_state_dict'])
model.eval()

# check training tiles (should have high confidence on positive tiles)
loader = get_dataloader(split='train', tile_size=256, batch_size=32)
pos_confs = []
neg_confs = []

with torch.no_grad():
    for batch_idx, (images, labels, bboxes) in enumerate(loader):
        if batch_idx >= 5: break
        images = images.to(device)
        locs, cls_scores, anchors = model(images)
        probs = F.softmax(cls_scores, dim=2)
        max_probs = probs[:, :, 1].max(dim=1).values
        for i in range(len(labels)):
            if labels[i].item() == 1:
                pos_confs.append(max_probs[i].item())
            else:
                neg_confs.append(max_probs[i].item())

avg_pos = sum(pos_confs) / len(pos_confs) if pos_confs else 0
avg_neg = sum(neg_confs) / len(neg_confs) if neg_confs else 0
max_pos = max(pos_confs) if pos_confs else 0
max_neg = max(neg_confs) if neg_confs else 0

print(f'on training tiles (first 5 batches):')
print(f'  positive tiles ({len(pos_confs)} tiles): avg confidence = {avg_pos:.4f}, max = {max_pos:.4f}')
print(f'  negative tiles ({len(neg_confs)} tiles): avg confidence = {avg_neg:.4f}, max = {max_neg:.4f}')
print(f'')

if avg_pos > avg_neg * 2:
    print(f'  the model shows higher confidence on positive tiles (good sign)')
elif len(pos_confs) == 0:
    print(f'  no positive tiles seen in these batches (try more batches)')
else:
    print(f'  WARNING: positive and negative tiles have similar confidence')
    print(f'  the model may not have learned to detect waldo properly')
    print(f'  consider training for more epochs')
"
echo ""

# step 7: run confidence test on test tiles
echo "========================================"
echo "  Step 7: Confidence test on test tiles"
echo "========================================"
$PYTHON test_confidence.py
echo ""

# step 8: run full evaluation
echo "========================================"
echo "  Step 8: Full evaluation"
echo "========================================"
$PYTHON evaluate.py
echo ""

echo "========================================"
echo "  Pipeline complete!"
echo ""
echo "  outputs:"
echo "    checkpoints/best_model.pth      trained model weights"
echo "    checkpoints/training_history.json  loss per epoch"
echo "    results/training_curves.png      train vs val loss plots"
echo "    results/confidence_distribution.png  waldo vs background confidence"
echo "    results/tp_*.png                 true positive tile examples"
echo "    results/fp_*.png                 false positive tile examples"
echo "    results/fn_*.png                 false negative tile examples"
echo "    results/demo_full_image.png      waldo detection on full image"
echo ""
echo "  to save a complete log of this output:"
echo "    bash run.sh 2>&1 | tee results/run_log.txt"
echo "========================================"
