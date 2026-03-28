"""
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
"""

import torch
import torch.nn.functional as F
from torchvision.ops import nms
from config import NMS_CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD


'''
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
'''
def center_to_corners(boxes):

    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)

'''

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
'''
def decode_predictions(loc_preds, anchors):

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


'''
# this is the main function that evaluate.py will call
# takes the raw output from the model and filters it down to
# just the best non overlapping predictions
#
# locs: [N, 1364, 4] predicted offsets from the model
# cls_scores: [N, 1364, 2] raw logits (waldo vs background) from the model
# anchors: [1364, 4] anchor positions in [cx, cy, w, h] format
#
# returns a list of tuples, one per image in the batch
# each tuple is (filtered_boxes, filtered_scores)
# filtered_boxes is [num_kept, 4] in [x1, y1, x2, y2] format
# filtered_scores is [num_kept] with confidence values
'''
def apply_nms(locs, cls_scores, anchors,
              confidence_threshold=NMS_CONFIDENCE_THRESHOLD,
              iou_threshold=NMS_IOU_THRESHOLD):

    batch_size = locs.shape[0]
    results = []

    for i in range(batch_size):
        # grab predictions for this one image
        img_locs = locs[i] # [1364, 4]
        img_cls = cls_scores[i] # [1364, 2]

        '''
        # step 1: turn raw logits into probabilities using softmax
        # the model outputs raw scores, softmax converts them to 0 to 1 range
        # column 0 is background probability, column 1 is waldo probability
        '''
        probs = F.softmax(img_cls, dim=1)
        waldo_probs = probs[:, 1] # we only care about the waldo probability

        '''
        # step 2: decode the predicted offsets into actual box coordinates
        '''
        decoded_boxes = decode_predictions(img_locs, anchors) # [1364, 4] in [cx, cy, w, h]

        '''
        # step 3: convert to corner format for torchvision nms
        '''
        corner_boxes = center_to_corners(decoded_boxes) # [1364, 4] in [x1, y1, x2, y2]

        '''
        # step 4: throw away predictions below the confidence threshold
        # most of the 1364 predictions will have very low confidence
        # this gets rid of the obvious noise before running nms
        '''
        confident_mask = waldo_probs > confidence_threshold
        confident_boxes = corner_boxes[confident_mask]
        confident_scores = waldo_probs[confident_mask]

        '''
        # if nothing passed the confidence threshold, return empty results for this image
        '''
        if confident_boxes.shape[0] == 0:
            results.append((torch.zeros(0, 4), torch.zeros(0)))
            continue

        '''
        # step 5: run torchvision nms to remove overlapping boxes
        # it sorts by score, keeps the best one, removes boxes that overlap too much
        # then moves to the next best and repeats
        # returns the indices of the boxes we should keep
        '''
        keep_indices = nms(confident_boxes, confident_scores, iou_threshold)

        # grab only the surviving boxes and scores
        final_boxes = confident_boxes[keep_indices]
        final_scores = confident_scores[keep_indices]

        results.append((final_boxes, final_scores))

    return results


if __name__ == "__main__":
    # test with fake predictions to make sure everything works

    # pretend we have 1 image with 10 anchor predictions
    num_anchors = 10

    # fake anchors scattered around the image
    anchors = torch.tensor([
        [50, 50, 30, 30],    # small box top left area
        [55, 55, 30, 30],    # overlapping with the first one (should get filtered)
        [200, 200, 40, 40],  # box in bottom right area
        [100, 100, 25, 25],  # box in the middle
        [105, 105, 25, 25],  # overlapping with middle box
        [30, 180, 20, 20],   # box bottom left
        [150, 50, 35, 35],   # box top right area
        [155, 55, 35, 35],   # overlapping with top right
        [128, 128, 50, 50],  # big box in center
        [10, 10, 15, 15],    # small box in corner
    ], dtype=torch.float32)

    # fake offsets (small random shifts)
    fake_locs = torch.randn(1, num_anchors, 4) * 0.1

    # fake class scores where a few anchors have high waldo confidence
    # column 0 is background, column 1 is waldo
    fake_cls = torch.tensor([[
        [5.0, -5.0],   # very confident background
        [4.0, -4.0],   # confident background
        [-3.0, 3.0],   # confident waldo
        [-2.0, 2.0],   # somewhat confident waldo
        [-1.5, 1.5],   # slightly confident waldo
        [3.0, -3.0],   # confident background
        [-4.0, 4.0],   # very confident waldo
        [-3.5, 3.5],   # confident waldo
        [1.0, -1.0],   # slightly background
        [6.0, -6.0],   # very confident background
    ]])

    print("testing apply_nms with fake predictions...")
    print(f"  input locs: {fake_locs.shape}")
    print(f"  input cls: {fake_cls.shape}")
    print(f"  anchors: {anchors.shape}")

    results = apply_nms(fake_locs, fake_cls, anchors, confidence_threshold=0.5, iou_threshold=0.5)

    for i, (boxes, scores) in enumerate(results):
        print(f"\n  image {i}:")
        print(f"    kept {boxes.shape[0]} predictions out of {num_anchors}")
        if boxes.shape[0] > 0:
            for j in range(boxes.shape[0]):
                print(f"    box {j}: {boxes[j].tolist()}, score: {scores[j]:.4f}")
