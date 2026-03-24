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


def compute_loss(locs, cls_scores, anchors, labels, bboxes):
    # computes the SSD loss for a batch of tiles
    # this follows the standard SSD training approach from the paper
    #
    # locs: [N, 1364, 4] predicted offsets from the model
    # cls_scores: [N, 1364, 2] raw logits (waldo vs background) from the model
    # anchors: [1364, 4] anchor positions in [cx, cy, w, h] format
    # labels: [N] tile labels (0 = no waldo, 1 = has waldo)
    # bboxes: [N, 4] ground truth bboxes in [xmin, ymin, xmax, ymax] format
    #
    # returns the combined loss (cls_loss + loc_loss)

    batch_size = locs.shape[0]
    num_anchors = locs.shape[1]

    total_cls_loss = 0.0
    total_loc_loss = 0.0
    num_positive = 0 # count how many positive anchors we found across the batch

    for i in range(batch_size):
        img_locs = locs[i] # [1364, 4]
        img_cls = cls_scores[i] # [1364, 2]
        label = labels[i].item() # 0 or 1

        # create classification targets for every anchor
        # on negative tiles: all anchors should predict background (class 0)
        # on positive tiles: matched anchors predict waldo (class 1), rest predict background
        cls_targets = torch.zeros(num_anchors, dtype=torch.long, device=locs.device)

        if label == 1:
            # this tile has waldo, convert the ground truth bbox to center format
            gt_box = bbox_to_center(bboxes[i]).to(locs.device)

            # find which anchor overlaps the most with the ground truth
            # we use IoU (intersection over union) to measure overlap
            ious = compute_iou(gt_box, anchors.to(locs.device))
            best_anchor_idx = ious.argmax().item()

            # also mark any anchor with IoU above 0.5 as positive
            # this gives the model more positive examples to learn from
            # instead of just the single best matching anchor
            positive_mask = ious > 0.5
            positive_mask[best_anchor_idx] = True # make sure the best one is always included

            # set classification targets for matched anchors to waldo (class 1)
            cls_targets[positive_mask] = 1

            # compute localization loss only for the matched anchors
            # we encode the ground truth as offsets from each matched anchor
            # then compare against what the model predicted
            positive_indices = positive_mask.nonzero(as_tuple=True)[0]
            num_positive += len(positive_indices)

            if len(positive_indices) > 0:
                for idx in positive_indices:
                    # encode the ground truth bbox as offsets from this anchor
                    target_offsets = encode_ground_truth(gt_box, anchors[idx].to(locs.device))
                    predicted_offsets = img_locs[idx]

                    # smooth L1 loss for bounding box regression
                    # this is the standard loss used in SSD and other object detectors
                    # its less sensitive to outliers than MSE loss
                    # torch.nn.functional.smooth_l1_loss is from pytorch
                    total_loc_loss += F.smooth_l1_loss(predicted_offsets, target_offsets)

        # classification loss using cross entropy
        # this is computed for all 1364 anchors on every tile
        # cross entropy handles the softmax internally so we pass raw logits
        # torch.nn.functional.cross_entropy is from pytorch
        total_cls_loss += F.cross_entropy(img_cls, cls_targets)

    # average the losses across the batch
    cls_loss = total_cls_loss / batch_size
    loc_loss = total_loc_loss / max(num_positive, 1) # avoid division by zero if no positives

    # combine both losses
    # the cls loss teaches "is waldo here?" and the loc loss teaches "where exactly?"
    total_loss = cls_loss + loc_loss

    return total_loss, cls_loss.item(), loc_loss.item() if isinstance(loc_loss, torch.Tensor) else loc_loss


def train_one_epoch(model, dataloader, optimizer, device):
    # runs one full pass through the training data
    # for each batch: forward pass, compute loss, backward pass, update weights
    # returns the average loss for this epoch

    model.train() # put the model in training mode

    running_loss = 0.0
    running_cls_loss = 0.0
    running_loc_loss = 0.0
    num_batches = 0

    for images, labels, bboxes in dataloader:
        # move everything to the gpu (or cpu if no gpu)
        images = images.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)

        # forward pass: run the tiles through the model
        locs, cls_scores, anchors = model(images)

        # compute the loss (both classification and localization)
        loss, cls_loss, loc_loss = compute_loss(locs, cls_scores, anchors, labels, bboxes)

        # backward pass: compute gradients
        optimizer.zero_grad() # reset gradients from the previous batch
        loss.backward() # compute new gradients based on the loss
        optimizer.step() # update the model weights using the gradients

        running_loss += loss.item()
        running_cls_loss += cls_loss
        running_loc_loss += loc_loss
        num_batches += 1

    # return average losses for the epoch
    avg_loss = running_loss / max(num_batches, 1)
    avg_cls = running_cls_loss / max(num_batches, 1)
    avg_loc = running_loc_loss / max(num_batches, 1)

    return avg_loss, avg_cls, avg_loc


# TODO: add validation
# TODO: add checkpoint saving
