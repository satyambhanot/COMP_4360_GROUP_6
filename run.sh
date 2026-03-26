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
$PYTHON -c "
import sys; sys.path.insert(0, 'src')
from model import create_model
from config import MODEL_TYPE
model = create_model()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = total - trainable
print(f'model type: {model.model_type}')
if MODEL_TYPE == 'retinanet':
    print(f'  architecture: RetinaNet ResNet50 FPN v2 (pretrained on COCO)')
    print(f'  preprocessing: HSV mask + Sobel edge detection on HSV')
    print(f'  backbone: ResNet50 + FPN (frozen, pretrained on COCO)')
    print(f'  detection heads: RetinaNet heads with built in focal loss')
    print(f'  input size: 512x512 (larger view for more context)')
elif MODEL_TYPE == 'pretrained_ssd':
    print(f'  architecture: SSD300 VGG16 (pretrained on COCO)')
    print(f'  preprocessing: HSV mask + Sobel edge detection on HSV')
    print(f'  backbone: VGG16 (frozen, pretrained on COCO)')
    print(f'  detection heads: SSD heads (classification head replaced for waldo)')
    print(f'  input size: 300x300')
else:
    print(f'  architecture: Custom WaldoDetector')
    print(f'  preprocessing: HSV mask + Sobel edge detection on HSV')
    print(f'  backbone: MobileNetV2 (first 11 layers, frozen)')
    print(f'  detection heads: 5 custom SSD blocks')
    print(f'  input size: 256x256')
print(f'')
print(f'parameters:')
print(f'  total:     {total:,}')
print(f'  trainable: {trainable:,}')
print(f'  frozen:    {frozen:,}')
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
from model import WaldoDetector, WaldoDetectorPretrained, WaldoDetectorRetinaNet
from data_pipeline import get_dataloader
from config import CHECKPOINTS_DIR, MODEL_TYPE

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

cp = torch.load(os.path.join(CHECKPOINTS_DIR, 'best_model.pth'), map_location=device, weights_only=False)
model_type = cp.get('model_type', MODEL_TYPE)

if model_type == 'retinanet':
    model = WaldoDetectorRetinaNet().to(device)
elif model_type == 'pretrained_ssd':
    model = WaldoDetectorPretrained().to(device)
else:
    model = WaldoDetector().to(device)

model.load_state_dict(cp['model_state_dict'])
model.eval()
model.model_type = model_type

# check training tiles (should have high confidence on positive tiles)
loader = get_dataloader(split='train', tile_size=256, batch_size=32)
pos_confs = []
neg_confs = []

with torch.no_grad():
    for batch_idx, (images, labels, bboxes) in enumerate(loader):
        if batch_idx >= 5: break
        images = images.to(device)

        if model_type in ('pretrained_ssd', 'retinanet'):
            detections = model(images)
            for i in range(len(labels)):
                det = detections[i]
                waldo_mask = det['labels'] == 1
                scores = det['scores'][waldo_mask]
                conf = scores.max().item() if scores.shape[0] > 0 else 0.0
                if labels[i].item() == 1:
                    pos_confs.append(conf)
                else:
                    neg_confs.append(conf)
        else:
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
print(f'  separation ratio: {avg_pos / max(avg_neg, 0.0001):.1f}x')
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

# step 9: quick summary of the run
echo "========================================"
echo "  Step 9: Run summary"
echo "========================================"
$PYTHON -c "
import sys, os, json; sys.path.insert(0, 'src')
from config import CHECKPOINTS_DIR, MODEL_TYPE, NUM_EPOCHS

# load results if they exist
results_path = os.path.join('results', 'evaluation_results.json')
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
    m = results['metrics']
    model_names = {'retinanet': 'RetinaNet ResNet50 FPN', 'pretrained_ssd': 'pretrained SSD300', 'custom_ssd': 'custom WaldoDetector'}
    print(f'model type: {model_names.get(MODEL_TYPE, MODEL_TYPE)}')
    print(f'epochs trained: {NUM_EPOCHS}')
    print(f'')
    print(f'test set results:')
    print(f'  accuracy:  {m[\"accuracy\"]:.4f}')
    print(f'  precision: {m[\"precision\"]:.4f}')
    print(f'  recall:    {m[\"recall\"]:.4f}')
    print(f'  f1 score:  {m[\"f1\"]:.4f}')
    print(f'  true positives:  {m[\"tp\"]}')
    print(f'  false positives: {m[\"fp\"]}')
    print(f'  true negatives:  {m[\"tn\"]}')
    print(f'  false negatives: {m[\"fn\"]}')
    print(f'')
    print(f'confidence stats:')
    print(f'  avg positive tile: {results[\"avg_confidence_positive\"]:.4f}')
    print(f'  avg negative tile: {results[\"avg_confidence_negative\"]:.4f}')
    print(f'  max positive tile: {results[\"max_confidence_positive\"]:.4f}')
    print(f'  max negative tile: {results[\"max_confidence_negative\"]:.4f}')
else:
    print('no evaluation results found')
"
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
