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
# with batch size 24 and 256x256 tiles this fits comfortably

import sys
import os
import time
import json

# add the src folder to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import WaldoDetector, create_model
from data_pipeline import get_dataloader
from ConcatenateAnchors import generate_anchors
from config import (
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE,
    CHECKPOINTS_DIR, NUM_CLASSES, INPUT_SIZE,
    MODEL_TYPE,
    PRETRAINED_SSD_INPUT_SIZE, RETINANET_INPUT_SIZE,
    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR,
    EARLY_STOPPING_PATIENCE,
    WEIGHT_DECAY,
    FOCAL_LOSS_GAMMA, FOCAL_LOSS_ALPHA
)

# check what hardware we have available
# cuda is for nvidia gpus (like the GTX 1080 Ti we train on)
# mps is for apple silicon (M1/M2 macs) which also has gpu acceleration
# if neither is available we fall back to cpu which is slower but works
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
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
    # use torch.stack instead of torch.tensor so it stays on the same device as the input
    return torch.stack([cx, cy, w, h])


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

    # make sure the returned tensor is on the same device as the inputs
    return torch.stack([dx, dy, dw, dh])


def focal_loss_weight(loss_tensor, gamma=FOCAL_LOSS_GAMMA):
    # applies focal loss weighting to an already computed loss
    # the idea: if the model is already confident about a prediction (low loss)
    # we reduce its contribution to the total loss
    # if the model is confused (high loss) we keep the full contribution
    #
    # this is a simplified version that works with torchvisions internal SSD loss
    # instead of rewriting their loss function we just reweight the output
    # the effect is similar: easy examples get down weighted, hard examples stay
    #
    # for the standard focal loss formula FL = -alpha * (1-pt)^gamma * log(pt)
    # we approximate (1-pt)^gamma using the loss itself since loss = -log(pt)
    # so pt = exp(-loss) and (1-pt)^gamma = (1 - exp(-loss))^gamma
    pt = torch.exp(-loss_tensor)
    focal_weight = (1.0 - pt) ** gamma
    return (focal_weight * loss_tensor).mean()


def format_targets(labels, bboxes, device):
    # converts our dataloader output into the format torchvision detection models expect
    # both SSD and RetinaNet use the same dict format:
    #   "boxes": tensor of shape [num_objects, 4] in [xmin, ymin, xmax, ymax] format
    #   "labels": tensor of shape [num_objects] with class labels (1 for waldo)
    #
    # for positive tiles: one box and one label
    # for negative tiles: empty tensors (no objects)
    #
    # we scale the bboxes from our tile size (256) to whatever the model expects
    # SSD wants 300x300, RetinaNet wants 800x800

    if MODEL_TYPE == "retinanet":
        scale = RETINANET_INPUT_SIZE / INPUT_SIZE
    else:
        scale = PRETRAINED_SSD_INPUT_SIZE / INPUT_SIZE
    targets = []

    for i in range(len(labels)):
        if labels[i].item() == 1:
            # positive tile: waldo is here
            # scale the bbox coordinates from 256 space to 300 space
            box = bboxes[i].unsqueeze(0).to(device) * scale
            label = torch.tensor([1], dtype=torch.long, device=device)
            targets.append({"boxes": box, "labels": label})
        else:
            # negative tile: no waldo
            # torchvision expects empty tensors not zeros
            targets.append({
                "boxes": torch.zeros(0, 4, device=device),
                "labels": torch.zeros(0, dtype=torch.long, device=device)
            })

    return targets


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
    # we initialize loc_loss as a tensor on the same device as the predictions
    # so the return type is always consistent whether or not we had positive tiles
    # if we used a plain float (0.0) it would cause issues when we call .item() later
    total_loc_loss = torch.tensor(0.0, device=locs.device)
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
            # bboxes are already on the right device because we moved them in the training loop
            gt_box = bbox_to_center(bboxes[i])

            # find which anchor overlaps the most with the ground truth
            # we use IoU (intersection over union) to measure overlap
            # anchors are already on the right device from concatenate_all_heads
            ious = compute_iou(gt_box, anchors)
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
                    target_offsets = encode_ground_truth(gt_box, anchors[idx])
                    predicted_offsets = img_locs[idx]

                    # smooth L1 loss for bounding box regression
                    # this is the standard loss used in SSD and other object detectors
                    # its less sensitive to outliers than MSE loss
                    # torch.nn.functional.smooth_l1_loss is from pytorch
                    total_loc_loss += F.smooth_l1_loss(predicted_offsets, target_offsets)

        # classification loss using focal cross entropy
        # standard cross entropy treats every anchor equally but most anchors are
        # easy background predictions. focal loss down weights those easy ones
        # so the model focuses on the confusing anchors near waldo like characters
        ce_loss = F.cross_entropy(img_cls, cls_targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = FOCAL_LOSS_ALPHA * (1.0 - pt) ** FOCAL_LOSS_GAMMA
        total_cls_loss += (focal_weight * ce_loss).mean()

    # average the losses across the batch
    cls_loss = total_cls_loss / batch_size
    loc_loss = total_loc_loss / max(num_positive, 1) # avoid division by zero if no positives

    # combine both losses
    # the cls loss teaches "is waldo here?" and the loc loss teaches "where exactly?"
    total_loss = cls_loss + loc_loss

    return total_loss, cls_loss.item(), loc_loss.item()


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

        if MODEL_TYPE in ("pretrained_ssd", "retinanet"):
            # torchvision models (SSD and RetinaNet) share the same training interface
            # they take images + targets and compute loss internally
            # returns a dict with "classification" and "bbox_regression" losses
            targets = format_targets(labels, bboxes, device)
            loss_dict = model(images, targets)

            # for pretrained SSD we manually apply focal loss weighting
            # RetinaNet already has focal loss built in so we use its loss directly
            if MODEL_TYPE == "pretrained_ssd":
                focal_cls_loss = focal_loss_weight(loss_dict["classification"])
                cls_loss = focal_cls_loss.item()
                loc_loss = loss_dict["bbox_regression"].item()
                loss = focal_cls_loss + loss_dict["bbox_regression"]
            else:
                # retinanet: focal loss is already applied internally
                cls_loss = loss_dict["classification"].item()
                loc_loss = loss_dict["bbox_regression"].item()
                loss = loss_dict["classification"] + loss_dict["bbox_regression"]
        else:
            # our custom model returns raw predictions and we compute loss ourselves
            locs, cls_scores, anchors = model(images)
            loss, cls_loss, loc_loss = compute_loss(locs, cls_scores, anchors, labels, bboxes)

        # backward pass: compute gradients
        optimizer.zero_grad() # reset gradients from the previous batch
        loss.backward() # compute new gradients based on the loss

        # clip gradients to prevent exploding updates when the loss is very high
        # without this a single bad batch can make the weights jump wildly
        # max_norm=1.0 is a standard value used in most fine tuning setups
        # torch.nn.utils.clip_grad_norm_ is from pytorch
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step() # update the model weights using the clipped gradients

        running_loss += loss.item()
        running_cls_loss += cls_loss
        running_loc_loss += loc_loss
        num_batches += 1

    # return average losses for the epoch
    avg_loss = running_loss / max(num_batches, 1)
    avg_cls = running_cls_loss / max(num_batches, 1)
    avg_loc = running_loc_loss / max(num_batches, 1)

    return avg_loss, avg_cls, avg_loc


def validate(model, dataloader, device):
    # runs one pass through the validation data without updating weights
    # we use this to check if the model is overfitting
    # if train loss goes down but val loss goes up, the model is memorizing
    # the training data instead of learning to generalize

    model.eval() # put the model in eval mode

    running_loss = 0.0
    running_cls_loss = 0.0
    running_loc_loss = 0.0
    num_batches = 0

    # torch.no_grad tells pytorch not to track gradients
    # this saves memory and speeds things up since we dont need gradients for validation
    #
    # for torchvision models (SSD and RetinaNet) we need train mode to get losses
    # because in eval mode they return detections instead of losses
    # we still use no_grad so no weights get updated
    if MODEL_TYPE in ("pretrained_ssd", "retinanet"):
        model.train()

    with torch.no_grad():
        for images, labels, bboxes in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            if MODEL_TYPE in ("pretrained_ssd", "retinanet"):
                targets = format_targets(labels, bboxes, device)
                loss_dict = model(images, targets)
                if MODEL_TYPE == "pretrained_ssd":
                    focal_cls = focal_loss_weight(loss_dict["classification"])
                    cls_loss = focal_cls.item()
                else:
                    cls_loss = loss_dict["classification"].item()
                loc_loss = loss_dict["bbox_regression"].item()
                loss_val = cls_loss + loc_loss
                running_loss += loss_val
            else:
                locs, cls_scores, anchors = model(images)
                loss, cls_loss, loc_loss = compute_loss(locs, cls_scores, anchors, labels, bboxes)
                running_loss += loss.item()

            running_cls_loss += cls_loss
            running_loc_loss += loc_loss
            num_batches += 1

    model.eval() # put back in eval mode

    avg_loss = running_loss / max(num_batches, 1)
    avg_cls = running_cls_loss / max(num_batches, 1)
    avg_loc = running_loc_loss / max(num_batches, 1)

    return avg_loss, avg_cls, avg_loc


if __name__ == "__main__":
    print("starting training...")
    start_time = time.time()

    # create the model and move it to the gpu
    # create_model checks the config flag and returns either our custom WaldoDetector
    # or the pretrained SSD300 VGG16 from torchvision
    model = create_model()
    model = model.to(device)
    model_type = model.model_type
    print(f"model type: {model_type}")
    print(f"model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    # create the optimizer
    # for pretrained models (SSD and RetinaNet) the backbone is frozen
    # so we only train the detection heads and the 1x1 preprocessing conv
    # SGD with momentum works well for fine tuning pretrained models
    # weight decay (L2 regularization) prevents the heads from overfitting
    if MODEL_TYPE in ("pretrained_ssd", "retinanet") and hasattr(model, 'get_param_groups'):
        head_params = model.get_param_groups()
        optimizer = torch.optim.SGD(
            head_params, lr=LEARNING_RATE,
            momentum=0.9, weight_decay=WEIGHT_DECAY
        )
        print(f"  backbone: frozen (COCO pretrained features)")
        print(f"  head lr: {LEARNING_RATE}, weight_decay: {WEIGHT_DECAY}")
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE
        )

    # load the train and validation dataloaders
    # the train loader uses weighted sampling to handle class imbalance
    # the val loader just loads tiles normally without any augmentation
    print("loading data...")
    train_loader = get_dataloader(split="train", tile_size=256, batch_size=BATCH_SIZE)
    val_loader = get_dataloader(split="val", tile_size=256, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  train: {len(train_loader.dataset)} tiles")
    print(f"  val: {len(val_loader.dataset)} tiles")

    # learning rate scheduler
    # if val loss stops improving for LR_SCHEDULER_PATIENCE epochs, cut the learning rate
    # this helps the model converge when its stuck at a plateau instead of bouncing around
    # torch.optim.lr_scheduler.ReduceLROnPlateau is a standard pytorch scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min", # we want to minimize val loss
        patience=LR_SCHEDULER_PATIENCE,
        factor=LR_SCHEDULER_FACTOR
    )

    # keep track of the best validation loss so we know when to save
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0 # for early stopping

    # save loss history so evaluate.py can plot training curves later
    # we store every epochs losses in a list and save to json at the end
    history = {"train_loss": [], "val_loss": [], "train_cls": [], "val_cls": [], "train_loc": [], "val_loc": []}

    # make sure the checkpoints directory exists
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # the main training loop
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # train for one epoch
        train_loss, train_cls, train_loc = train_one_epoch(model, train_loader, optimizer, device)

        # validate to check for overfitting
        val_loss, val_cls, val_loc = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        # record the losses for this epoch so we can plot curves later
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_cls"].append(train_cls)
        history["val_cls"].append(val_cls)
        history["train_loc"].append(train_loc)
        history["val_loc"].append(val_loc)

        # tell the scheduler about the val loss so it can decide whether to reduce LR
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # check if this is the best model so far
        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0 # reset early stopping counter

            # save the best model checkpoint
            torch.save({
                "epoch": epoch + 1,
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(CHECKPOINTS_DIR, "best_model.pth"))
            saved = " [saved best]"
        else:
            epochs_without_improvement += 1

        # always save the last model so we can resume from where we stopped
        torch.save({
            "epoch": epoch + 1,
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }, os.path.join(CHECKPOINTS_DIR, "last_model.pth"))

        # print a summary for this epoch including the current learning rate
        print(f"epoch {epoch + 1}/{NUM_EPOCHS} ({epoch_time:.1f}s)"
              f"  train loss: {train_loss:.4f} (cls: {train_cls:.4f}, loc: {train_loc:.4f})"
              f"  val loss: {val_loss:.4f} (cls: {val_cls:.4f}, loc: {val_loc:.4f})"
              f"  lr: {current_lr:.6f}"
              f"{saved}")

        # early stopping: if val loss hasnt improved for too long, stop training
        # no point continuing if the model has stopped learning
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nearly stopping: val loss hasnt improved for {EARLY_STOPPING_PATIENCE} epochs")
            print(f"best was {best_val_loss:.4f} at epoch {best_epoch}")
            break

    # save the training history to a json file so evaluate.py can plot curves
    history_path = os.path.join(CHECKPOINTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # print the final summary
    total_time = time.time() - start_time
    print(f"\ntraining complete in {total_time:.1f} seconds")
    print(f"best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"checkpoints saved to {CHECKPOINTS_DIR}")
    print(f"training history saved to {history_path}")
