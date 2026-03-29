'''
utils.py
shared utility functions used across multiple files in the project
keeps common code in one place instead of duplicating it everywhere
'''

import torch

from config import (
    MODEL_TYPE, INPUT_SIZE,
    PRETRAINED_SSD_INPUT_SIZE, RETINANET_INPUT_SIZE
)


def get_device():
    '''
    figures out the best available hardware for training and inference
    checks in order: nvidia gpu, apple silicon, then cpu

    returns:
        torch.device for cuda, mps, or cpu
    '''
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def format_targets(labels, bboxes, device):
    '''
    converts our dataloader output into the format torchvision detection models expect
    both SSD and RetinaNet use the same dict format so this works for either

    args:
        labels: tensor of tile labels (0 = no waldo, 1 = has waldo)
        bboxes: tensor of bounding boxes [N, 4] in xmin ymin xmax ymax format
        device: torch device to put the tensors on

    returns:
        list of dicts, one per image, each with "boxes" and "labels" keys
        positive tiles get the scaled bbox, negative tiles get empty tensors
    '''
    # scale bboxes from tile size (256) to whatever the model expects
    # SSD wants 300x300, RetinaNet wants 512x512
    if MODEL_TYPE == "retinanet":
        scale = RETINANET_INPUT_SIZE / INPUT_SIZE
    else:
        scale = PRETRAINED_SSD_INPUT_SIZE / INPUT_SIZE

    targets = []
    for i in range(len(labels)):
        if labels[i].item() == 1:
            box = bboxes[i].unsqueeze(0).to(device) * scale
            label = torch.tensor([1], dtype=torch.long, device=device)
            targets.append({"boxes": box, "labels": label})
        else:
            targets.append({
                "boxes": torch.zeros(0, 4, device=device),
                "labels": torch.zeros(0, dtype=torch.long, device=device)
            })

    return targets
