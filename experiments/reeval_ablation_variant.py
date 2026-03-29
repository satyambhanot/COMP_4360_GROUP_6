'''
reeval_ablation_variant.py
re-evaluates a single ablation variant with correct preprocessing flags
called by reeval_all.py as a separate process

each variant has different input channels so the model must be created
with the right use_hsv/use_sobel flags before loading the checkpoint

usage: python experiments/reeval_ablation_variant.py <name> <use_hsv> <use_sobel>
'''

import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

os.environ["PYTHONUNBUFFERED"] = "1"

import torch

# set config for RetinaNet
import config
config.MODEL_TYPE = "retinanet"
config.BATCH_SIZE = 8
config.RETINANET_INPUT_SIZE = 512

from model import WaldoDetectorRetinaNet
from data_pipeline import get_dataloader
from utils import get_device
from config import BATCH_SIZE

device = get_device()


def evaluate_ablation_variant(variant_name, use_hsv, use_sobel):
    # creates the model with correct preprocessing flags, loads checkpoint, evaluates

    variant_dir = os.path.join(PROJECT_ROOT, "saved_results", "ablation", variant_name)
    checkpoint_path = os.path.join(variant_dir, "checkpoints", "best_model.pth")
    plots_dir = os.path.join(variant_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print(f"  {variant_name}: no checkpoint found")
        return None

    print(f"  loading {variant_name} (hsv={use_hsv}, sobel={use_sobel})", flush=True)

    # create model with the CORRECT preprocessing flags
    # this is why we need a separate script: the channel_conv shape depends on these flags
    model = WaldoDetectorRetinaNet(use_hsv=use_hsv, use_sobel=use_sobel)
    model = model.to(device)

    # load the trained weights
    cp = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    model.model_type = "retinanet"

    print(f"  loaded checkpoint from epoch {cp.get('epoch', '?')}", flush=True)

    # evaluate on test set
    test_loader = get_dataloader(split="test", tile_size=256, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  test set: {len(test_loader.dataset)} tiles", flush=True)

    all_preds = []
    all_labels = []
    all_confs = []

    with torch.no_grad():
        for images, labels, bboxes in test_loader:
            images = images.to(device, non_blocking=True)

            detections = model(images)
            for i in range(len(detections)):
                det = detections[i]
                waldo_mask = det["labels"] == 1
                scores = det["scores"][waldo_mask]
                if scores.shape[0] > 0:
                    all_preds.append(1)
                    all_confs.append(scores.max().item())
                else:
                    all_preds.append(0)
                    all_confs.append(0.0)
                all_labels.append(labels[i].item())

    # compute metrics
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)

    total = len(all_preds)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

    print(f"  {variant_name}: P={precision:.4f} R={recall:.4f} F1={f1:.4f} (TP={tp} FP={fp} FN={fn})", flush=True)

    # save metrics
    result = {
        "variant": variant_name,
        "use_hsv": use_hsv,
        "use_sobel": use_sobel,
        "metrics": metrics
    }
    with open(os.path.join(variant_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    return metrics


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python reeval_ablation_variant.py <name> <use_hsv> <use_sobel>")
        exit(1)

    variant_name = sys.argv[1]
    use_hsv = sys.argv[2].lower() == "true"
    use_sobel = sys.argv[3].lower() == "true"

    evaluate_ablation_variant(variant_name, use_hsv, use_sobel)
