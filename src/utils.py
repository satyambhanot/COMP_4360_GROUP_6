'''
utils.py
shared utility functions used across multiple files in the project
keeps common code in one place instead of duplicating it everywhere
'''

import torch
from config import (
    MODEL_TYPE, INPUT_SIZE,
    PRETRAINED_SSD_INPUT_SIZE, RETINANET_INPUT_SIZE, NUM_SSD_HEADS
)
from model import(
    WaldoDetector, WaldoDetectorPretrained, WaldoDetectorRetinaNet
)

'''
helper function that creates the right model based on MODEL_TYPE config
this way train.py and evaluate.py dont need to know which model class to use
they just call create_model() and get the right one
'''
def create_model(use_hsv=True, use_sobel=True, use_pretrained_backbone=True, num_heads=NUM_SSD_HEADS):

    if MODEL_TYPE == "retinanet":
        model = WaldoDetectorRetinaNet(use_hsv=use_hsv, use_sobel=use_sobel)

    elif MODEL_TYPE == "pretrained_ssd":
        model = WaldoDetectorPretrained(use_hsv=use_hsv, use_sobel=use_sobel)
    else:
        model = WaldoDetector(use_hsv=use_hsv, use_sobel=use_sobel, use_pretrained=use_pretrained_backbone, num_heads=num_heads)

    return model

def get_device():
    '''
    # figures out the best available hardware for training and inference
    # checks in order: nvidia gpu (cuda), apple silicon (mps), then falls back to cpu
    # we put this here because train.py, evaluate.py, and the experiment scripts
    # all need to do this exact same check
    '''
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def format_targets(labels, bboxes, device):
    '''
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
    '''

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
