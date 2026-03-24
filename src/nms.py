# nms.py
# non maximum suppression
# after the model makes predictions we end up with 1364 anchor predictions per image
# most of them will say "no waldo here" with low confidence
# but for the spots where waldo actually is, multiple nearby anchors
# will all say "waldo is here!" with high confidence
# we dont want 10 boxes around the same spot, we just want the best one
#
# this only runs during evaluation, not during training
# during training we need all 1364 raw predictions to compute the loss
#
# the steps:
#   1. decode the predicted offsets into actual box coordinates
#   2. convert box format from [cx, cy, w, h] to [x1, y1, x2, y2]
#   3. run softmax on class scores to get probabilities
#   4. throw away anything below the confidence threshold
#   5. run torchvision.ops.nms to remove overlapping boxes
#   6. return only the surviving boxes and their scores
#
# we do this for each image in the batch separately
# because you cant mix predictions from different images

import torch
import torch.nn.functional as F
from torchvision.ops import nms
from config import NMS_CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD


def center_to_corners(boxes):
    # converts boxes from [cx, cy, w, h] format to [x1, y1, x2, y2] format
    # we need this because torchvision.ops.nms expects corner format
    # our anchors and decoded predictions use center format
    #
    # cx, cy is the center of the box
    # w, h is the width and height
    # x1, y1 is the top left corner
    # x2, y2 is the bottom right corner
    #
    # the math is simple:
    #   x1 = cx minus half the width (left edge)
    #   y1 = cy minus half the height (top edge)
    #   x2 = cx plus half the width (right edge)
    #   y2 = cy plus half the height (bottom edge)

    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


def decode_predictions(loc_preds, anchors):
    # the SSD heads dont predict actual box positions
    # they predict offsets from the anchor positions
    # this function turns those offsets into actual box coordinates
    #
    # loc_preds is [num_anchors, 4] with predicted offsets (dx, dy, dw, dh)
    # anchors is [num_anchors, 4] with anchor positions (cx, cy, w, h)
    #
    # the decoding formula:
    #   predicted_cx = anchor_cx + (dx * anchor_w)
    #     the offset dx says how far to shift the center horizontally
    #     we multiply by anchor width so the offset is relative to the anchor size
    #   predicted_cy = anchor_cy + (dy * anchor_h)
    #     same thing but vertical
    #   predicted_w = anchor_w * exp(dw)
    #     the offset dw says how much to scale the width
    #     we use exp so the width is always positive
    #   predicted_h = anchor_h * exp(dh)
    #     same thing for height
    #
    # i learned this decoding formula from the SSD paper
    # (Liu et al, SSD: Single Shot MultiBox Detector, ECCV 2016)
    # its the standard way all SSD implementations decode predictions

    anchor_cx = anchors[:, 0]
    anchor_cy = anchors[:, 1]
    anchor_w = anchors[:, 2]
    anchor_h = anchors[:, 3]

    dx = loc_preds[:, 0]
    dy = loc_preds[:, 1]
    dw = loc_preds[:, 2]
    dh = loc_preds[:, 3]

    # apply the decoding formula
    pred_cx = anchor_cx + dx * anchor_w
    pred_cy = anchor_cy + dy * anchor_h
    pred_w = anchor_w * torch.exp(dw)
    pred_h = anchor_h * torch.exp(dh)

    return torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)


# TODO: add apply_nms function
# TODO: add test at the bottom
