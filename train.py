# train.py
# this is the main training script, you run it from the project root
# like: python train.py
#
# what this file does:
#   loads tiles from the data pipeline, feeds them through the model,
#   computes two losses (classification + localization), uses those to
#   update the model weights, and saves the best version to checkpoints
#
# we are training a standard SSD detector which means:
#   every anchor gets a classification target (waldo or background)
#   matched anchors on positive tiles also get a localization target
#   the model learns at the anchor level, not the tile level
#
# we train on a GTX 1080 Ti with 11GB VRAM
# with batch size 16 and 256x256 tiles this should fit comfortably

import sys
import os
import time

# add the src folder to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import WaldoDetector
from data_pipeline import get_dataloader
from ConcatenateAnchors import generate_anchors
from config import (
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE,
    CHECKPOINTS_DIR, NUM_CLASSES, INPUT_SIZE
)

# check if we have a gpu available
# the GTX 1080 Ti has 11GB VRAM which is plenty for our model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

def bbox_to_center(bbox):
    # converts a bounding box from [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
    # our dataloader gives us bboxes in corner format
    # but anchors and SSD predictions use center format
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return torch.tensor([cx, cy, w, h])


def compute_iou(box, anchors):
    # computes intersection over union between one ground truth box and all anchors
    # box is [4] in [cx, cy, w, h] format
    # anchors is [num_anchors, 4] in [cx, cy, w, h] format
    # returns [num_anchors] with iou values
    #
    # we need to convert to corner format first to compute the overlap
    # iou = area of overlap / area of union
    # this tells us how much two boxes overlap on a scale of 0 to 1

    # convert box to corners
    b_x1 = box[0] - box[2] / 2
    b_y1 = box[1] - box[3] / 2
    b_x2 = box[0] + box[2] / 2
    b_y2 = box[1] + box[3] / 2

    # convert anchors to corners
    a_x1 = anchors[:, 0] - anchors[:, 2] / 2
    a_y1 = anchors[:, 1] - anchors[:, 3] / 2
    a_x2 = anchors[:, 0] + anchors[:, 2] / 2
    a_y2 = anchors[:, 1] + anchors[:, 3] / 2

    # compute intersection
    inter_x1 = torch.max(b_x1, a_x1)
    inter_y1 = torch.max(b_y1, a_y1)
    inter_x2 = torch.min(b_x2, a_x2)
    inter_y2 = torch.min(b_y2, a_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # compute union
    box_area = box[2] * box[3]
    anchor_areas = anchors[:, 2] * anchors[:, 3]
    union_area = box_area + anchor_areas - inter_area

    return inter_area / (union_area + 1e-6) # small epsilon to avoid division by zero


def encode_ground_truth(gt_box, anchor):
    # this is the reverse of decode_predictions in nms.py
    # given a ground truth box and an anchor, compute what offsets the model should predict
    # the SSD paper (Liu et al, ECCV 2016) defines this encoding formula
    #
    # gt_box is [cx, cy, w, h] the actual waldo position
    # anchor is [cx, cy, w, h] the anchor position
    # returns [dx, dy, dw, dh] the target offsets
    #
    # the formula:
    #   dx = (gt_cx minus anchor_cx) / anchor_w
    #   dy = (gt_cy minus anchor_cy) / anchor_h
    #   dw = log(gt_w / anchor_w)
    #   dh = log(gt_h / anchor_h)

    dx = (gt_box[0] - anchor[0]) / (anchor[2] + 1e-6)
    dy = (gt_box[1] - anchor[1]) / (anchor[3] + 1e-6)
    dw = torch.log((gt_box[2] + 1e-6) / (anchor[2] + 1e-6))
    dh = torch.log((gt_box[3] + 1e-6) / (anchor[3] + 1e-6))

    return torch.tensor([dx, dy, dw, dh])


# TODO: add compute_loss function
# TODO: add training loop
# TODO: add validation
# TODO: add checkpoint saving
