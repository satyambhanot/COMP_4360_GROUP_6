# utils.py
# shared utility functions used across multiple files in the project
# keeps common code in one place instead of duplicating it everywhere

import torch
from config import (
    MODEL_TYPE, INPUT_SIZE,
    PRETRAINED_SSD_INPUT_SIZE, RETINANET_INPUT_SIZE
)


def get_device():
    # figures out the best available hardware for training and inference
    # checks in order: nvidia gpu (cuda), apple silicon (mps), then falls back to cpu
    # we put this here because train.py, evaluate.py, and the experiment scripts
    # all need to do this exact same check
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def format_targets(labels, bboxes, device):
    # converts our dataloader output into the format torchvision detection models expect
    # both SSD and RetinaNet use the same dict format:
    #   "boxes": tensor of shape [num_objects, 4] in [xmin, ymin, xmax, ymax] format
    #   "labels": tensor of shape [num_objects] with class labels (1 for waldo)
    #
    # we scale the bboxes from our tile size (256) to whatever the model expects
    # SSD wants 300x300, RetinaNet wants 512x512
    #
    # this function is used in both train.py and ablation experiment scripts
    # so we keep it here instead of duplicating it

    if MODEL_TYPE == "retinanet":
        scale = RETINANET_INPUT_SIZE / INPUT_SIZE
    else:
        scale = PRETRAINED_SSD_INPUT_SIZE / INPUT_SIZE

    targets = []
    for i in range(len(labels)):
        if labels[i].item() == 1:
            # positive tile: waldo is here
            box = bboxes[i].unsqueeze(0).to(device) * scale
            label = torch.tensor([1], dtype=torch.long, device=device)
            targets.append({"boxes": box, "labels": label})
        else:
            # negative tile: no waldo
            targets.append({
                "boxes": torch.zeros(0, 4, device=device),
                "labels": torch.zeros(0, dtype=torch.long, device=device)
            })

    return targets
