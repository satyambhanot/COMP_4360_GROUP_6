# sanity_check.py
# quick check after training to see if the model learned anything useful
# loads the best checkpoint and runs a few training batches through it
# compares confidence on positive tiles (should be high) vs negative tiles (should be low)
#
# this was originally inline python in run.sh, moved here to keep things clean
# usage: python src/sanity_check.py

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from model import WaldoDetector, WaldoDetectorPretrained, WaldoDetectorRetinaNet
from data_pipeline import get_dataloader
from utils import get_device
from config import CHECKPOINTS_DIR, MODEL_TYPE


def run_sanity_check():
    device = get_device()

    checkpoint_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print("no checkpoint found, skipping sanity check")
        return

    cp = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = cp.get("model_type", MODEL_TYPE)

    if model_type == "retinanet":
        model = WaldoDetectorRetinaNet().to(device)
    elif model_type == "pretrained_ssd":
        model = WaldoDetectorPretrained().to(device)
    else:
        model = WaldoDetector().to(device)

    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    model.model_type = model_type

    # check training tiles (should have high confidence on positive tiles)
    loader = get_dataloader(split="train", tile_size=256, batch_size=32)
    pos_confs = []
    neg_confs = []

    with torch.no_grad():
        for batch_idx, (images, labels, bboxes) in enumerate(loader):
            if batch_idx >= 5:
                break
            images = images.to(device)

            if model_type in ("pretrained_ssd", "retinanet"):
                detections = model(images)
                for i in range(len(labels)):
                    det = detections[i]
                    waldo_mask = det["labels"] == 1
                    scores = det["scores"][waldo_mask]
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

    print(f"on training tiles (first 5 batches):")
    print(f"  positive tiles ({len(pos_confs)} tiles): avg confidence = {avg_pos:.4f}, max = {max_pos:.4f}")
    print(f"  negative tiles ({len(neg_confs)} tiles): avg confidence = {avg_neg:.4f}, max = {max_neg:.4f}")
    print(f"  separation ratio: {avg_pos / max(avg_neg, 0.0001):.1f}x")
    print()

    if avg_pos > avg_neg * 2:
        print(f"  the model shows higher confidence on positive tiles (good sign)")
    elif len(pos_confs) == 0:
        print(f"  no positive tiles seen in these batches (try more batches)")
    else:
        print(f"  WARNING: positive and negative tiles have similar confidence")
        print(f"  the model may not have learned to detect waldo properly")


if __name__ == "__main__":
    run_sanity_check()
