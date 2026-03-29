'''
nms.py
non maximum suppression for filtering duplicate detections
after the model predicts 1364 anchors per image, many of them overlap
in the same spot. NMS keeps only the best prediction in each region.

only used during evaluation, not training. the steps are:
  1. decode predicted offsets into actual box coordinates
  2. convert from center format [cx, cy, w, h] to corner format [x1, y1, x2, y2]
  3. run softmax to get confidence probabilities
  4. throw away predictions below the confidence threshold
  5. run torchvision.ops.nms to remove overlapping boxes
  6. return the surviving boxes and scores
'''
# because you cant mix predictions from different images

import torch
import torch.nn.functional as F
from torchvision.ops import nms
from config import NMS_CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD


def center_to_corners(boxes):
    '''
    converts boxes from center format [cx, cy, w, h] to corner format [x1, y1, x2, y2]
    torchvision.ops.nms expects corner format but our anchors use center format

    args:
        boxes: tensor [N, 4] in center format [cx, cy, w, h]

    returns:
        tensor [N, 4] in corner format [x1, y1, x2, y2]
    '''

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
    '''
    converts predicted offsets from SSD heads into actual box coordinates
    the model predicts (dx, dy, dw, dh) relative to each anchor position
    this decodes those into absolute (cx, cy, w, h) coordinates

    the decoding formula from the SSD paper (Liu et al, ECCV 2016):
      predicted_cx = anchor_cx + dx * anchor_w
      predicted_cy = anchor_cy + dy * anchor_h
      predicted_w = anchor_w * exp(dw)
      predicted_h = anchor_h * exp(dh)

    args:
        loc_preds: predicted offsets [num_anchors, 4] as (dx, dy, dw, dh)
        anchors: anchor positions [num_anchors, 4] as (cx, cy, w, h)

    returns:
        decoded boxes [num_anchors, 4] as (cx, cy, w, h) in pixel coordinates
    '''

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


def apply_nms(locs, cls_scores, anchors,
              confidence_threshold=NMS_CONFIDENCE_THRESHOLD,
              iou_threshold=NMS_IOU_THRESHOLD):
    '''
    main NMS function called by evaluate.py during inference
    takes raw model output and filters down to the best non overlapping predictions

    args:
        locs: predicted offsets [N, 1364, 4] from the model
        cls_scores: raw logits [N, 1364, 2] (waldo vs background)
        anchors: anchor positions [1364, 4] in center format
        confidence_threshold: minimum waldo probability to keep (default from config)
        iou_threshold: maximum overlap between two kept boxes (default 0.5)

    returns:
        list of (boxes, scores) tuples, one per image in the batch
        boxes are [num_kept, 4] in corner format [x1, y1, x2, y2]
        scores are [num_kept] confidence values
    '''

    batch_size = locs.shape[0]
    results = []

    for i in range(batch_size):
        # grab predictions for this one image
        img_locs = locs[i] # [1364, 4]
        img_cls = cls_scores[i] # [1364, 2]

        # step 1: turn raw logits into probabilities using softmax
        # the model outputs raw scores, softmax converts them to 0 to 1 range
        # column 0 is background probability, column 1 is waldo probability
        probs = F.softmax(img_cls, dim=1)
        waldo_probs = probs[:, 1] # we only care about the waldo probability

        # step 2: decode the predicted offsets into actual box coordinates
        decoded_boxes = decode_predictions(img_locs, anchors) # [1364, 4] in [cx, cy, w, h]

        # step 3: convert to corner format for torchvision nms
        corner_boxes = center_to_corners(decoded_boxes) # [1364, 4] in [x1, y1, x2, y2]

        # step 4: throw away predictions below the confidence threshold
        # most of the 1364 predictions will have very low confidence
        # this gets rid of the obvious noise before running nms
        confident_mask = waldo_probs > confidence_threshold
        confident_boxes = corner_boxes[confident_mask]
        confident_scores = waldo_probs[confident_mask]

        # if nothing passed the confidence threshold, return empty results for this image
        if confident_boxes.shape[0] == 0:
            results.append((torch.zeros(0, 4), torch.zeros(0)))
            continue

        # step 5: run torchvision nms to remove overlapping boxes
        # it sorts by score, keeps the best one, removes boxes that overlap too much
        # then moves to the next best and repeats
        # returns the indices of the boxes we should keep
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
