# test_confidence.py
# quick test to see what confidence scores the model is giving on test tiles
# run from project root: python test_confidence.py
#
# this helps us figure out if the model is predicting waldo with confidence
# below the NMS threshold (0.5) which would explain why evaluate.py shows
# zero detections even though the model was trained successfully
# works with both our custom model and the pretrained SSD

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn.functional as F
from model import WaldoDetector, WaldoDetectorPretrained
from data_pipeline import get_dataloader
from config import CHECKPOINTS_DIR, USE_PRETRAINED_SSD

# supports nvidia gpu (cuda), apple silicon gpu (mps), or cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# load the model
checkpoint_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
if not os.path.exists(checkpoint_path):
    print("no checkpoint found, run train.py first")
    exit()

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model_type = checkpoint.get("model_type", "pretrained_ssd" if USE_PRETRAINED_SSD else "custom_ssd")

if model_type == "pretrained_ssd":
    model = WaldoDetectorPretrained().to(device)
else:
    model = WaldoDetector().to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.model_type = model_type
print(f"loaded checkpoint from epoch {checkpoint['epoch']} (model: {model_type})")

# load test tiles
loader = get_dataloader(split="test", tile_size=256, batch_size=16, shuffle=False)

print(f"\nchecking confidence scores on test tiles...")
print(f"NMS threshold is 0.5, anything below this gets thrown away\n")

total_tiles = 0
tiles_above_threshold = 0

with torch.no_grad():
    for images, labels, bboxes in loader:
        images = images.to(device)

        if model_type == "pretrained_ssd":
            # pretrained SSD returns detections with scores in eval mode
            detections = model(images)
            for i in range(len(labels)):
                label = labels[i].item()
                det = detections[i]
                # filter to waldo predictions only (label 1)
                waldo_mask = det["labels"] == 1
                waldo_scores = det["scores"][waldo_mask]
                conf = waldo_scores.max().item() if waldo_scores.shape[0] > 0 else 0.0

                marker = ""
                if label == 1:
                    marker = " <== WALDO IS HERE"
                if conf > 0.5:
                    tiles_above_threshold += 1
                    marker += " [ABOVE THRESHOLD]"

                print(f"  tile {total_tiles}: label={label}, max confidence={conf:.4f}{marker}")
                total_tiles += 1
        else:
            # custom model returns raw predictions
            locs, cls_scores, anchors = model(images)
            probs = F.softmax(cls_scores, dim=2)
            max_waldo_probs = probs[:, :, 1].max(dim=1).values

            for i in range(len(labels)):
                label = labels[i].item()
                conf = max_waldo_probs[i].item()
                marker = ""
                if label == 1:
                    marker = " <== WALDO IS HERE"
                if conf > 0.5:
                    tiles_above_threshold += 1
                    marker += " [ABOVE THRESHOLD]"

                print(f"  tile {total_tiles}: label={label}, max confidence={conf:.4f}{marker}")
                total_tiles += 1

print(f"\nsummary:")
print(f"  total test tiles: {total_tiles}")
print(f"  tiles above 0.5 threshold: {tiles_above_threshold}")
print(f"  tiles below 0.5 threshold: {total_tiles - tiles_above_threshold}")

if tiles_above_threshold == 0:
    print(f"\n  the model never reaches 0.5 confidence on any test tile")
    print(f"  this means NMS throws away all predictions")
    print(f"  we might need to lower NMS_CONFIDENCE_THRESHOLD in config.py")
    print(f"  or train for more epochs")
