'''
train.py
main training script for all model architectures
loads tiles from the data pipeline, feeds them through the model,
computes classification + localization losses, updates weights,
and saves the best checkpoint

supports three architectures via MODEL_TYPE in config:
  custom_ssd: trains SSD heads from scratch with Adam
  pretrained_ssd: fine tunes SSD300 VGG16 heads with SGD + focal loss
  retinanet: fine tunes RetinaNet FPN heads with SGD (focal loss built in)

the run_training() function can be called from experiment scripts
with custom checkpoint directories so each experiment saves separately

usage: python train.py (uses default config settings)
'''

import sys
import os
import time
import json

# add this directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

'''
standard pytorch performance flags
cudnn.benchmark auto tunes the best convolution algorithm for our fixed input size
this gives 10-20% speedup on all conv layers since the tuned algo gets reused every batch
only works when input dimensions dont change between batches (ours are always 256x256)
'''
torch.backends.cudnn.benchmark = True
# disable autograd anomaly detection since were not debugging NaN issues
# this removes per operation checking overhead
torch.autograd.set_detect_anomaly(False)

from models.model import create_model
from data.data_pipeline import get_dataloader
from utils import get_device, format_targets
from config import (
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE,
    CHECKPOINTS_DIR, MODEL_TYPE,
    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR,
    EARLY_STOPPING_PATIENCE,
    WEIGHT_DECAY,
    FOCAL_LOSS_GAMMA, FOCAL_LOSS_ALPHA
)

device = get_device()
print(f"using device: {device}")

def bbox_to_center(bbox):
    '''
    converts a bounding box from corner format to center format

    args:
        bbox: [xmin, ymin, xmax, ymax] tensor

    returns:
        [cx, cy, w, h] tensor
    '''
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    # use torch.stack instead of torch.tensor so it stays on the same device as the input
    return torch.stack([cx, cy, w, h])


def compute_iou(box, anchors):
    '''
    computes intersection over union between one ground truth box and all anchors
    iou measures how much two boxes overlap on a scale of 0 to 1

    args:
        box: single ground truth box [4] in center format [cx, cy, w, h]
        anchors: all anchor positions [num_anchors, 4] in center format

    returns:
        iou values [num_anchors] for each anchor vs the ground truth
    '''

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
    '''
    computes the target offsets the model should predict for a matched anchor
    this is the reverse of decode_predictions in nms.py
    formula from the SSD paper (Liu et al, ECCV 2016)

    args:
        gt_box: ground truth box [cx, cy, w, h]
        anchor: anchor position [cx, cy, w, h]

    returns:
        target offsets [dx, dy, dw, dh]
    '''

    dx = (gt_box[0] - anchor[0]) / (anchor[2] + 1e-6)
    dy = (gt_box[1] - anchor[1]) / (anchor[3] + 1e-6)
    dw = torch.log((gt_box[2] + 1e-6) / (anchor[2] + 1e-6))
    dh = torch.log((gt_box[3] + 1e-6) / (anchor[3] + 1e-6))

    # make sure the returned tensor is on the same device as the inputs
    return torch.stack([dx, dy, dw, dh])


def focal_loss_weight(loss_tensor, gamma=FOCAL_LOSS_GAMMA):
    '''
    applies focal loss weighting to an already computed loss tensor
    down weights easy examples (low loss) and keeps hard examples (high loss)

    args:
        loss_tensor: precomputed loss from torchvision SSD
        gamma: focusing parameter (default 2.0, higher = more focus on hard examples)

    returns:
        weighted mean loss
    '''
    # so pt = exp(-loss) and (1-pt)^gamma = (1 - exp(-loss))^gamma
    pt = torch.exp(-loss_tensor)
    focal_weight = (1.0 - pt) ** gamma
    return (focal_weight * loss_tensor).mean()



def compute_loss(locs, cls_scores, anchors, labels, bboxes):
    '''
    computes the SSD loss for a batch of tiles (custom SSD model only)
    combines focal cross entropy for classification with smooth L1 for localization

    args:
        locs: predicted offsets [N, 1364, 4]
        cls_scores: raw logits [N, 1364, 2] (waldo vs background)
        anchors: anchor positions [1364, 4] in center format
        labels: tile labels [N] (0 or 1)
        bboxes: ground truth boxes [N, 4] in corner format

    returns:
        total_loss, cls_loss (float), loc_loss (float)
    '''

    batch_size = locs.shape[0]
    num_anchors = locs.shape[1]

    total_cls_loss = 0.0
    '''
    we initialize loc_loss as a tensor on the same device as the predictions
    so the return type is always consistent whether or not we had positive tiles
    if we used a plain float (0.0) it would cause issues when we call .item() later
    '''
    total_loc_loss = torch.tensor(0.0, device=locs.device)
    num_positive = 0 # count how many positive anchors we found across the batch

    for i in range(batch_size):
        img_locs = locs[i] # [1364, 4]
        img_cls = cls_scores[i] # [1364, 2]
        label = labels[i].item() # 0 or 1

        '''
        create classification targets for every anchor
        on negative tiles: all anchors should predict background (class 0)
        on positive tiles: matched anchors predict waldo (class 1), rest predict background
        '''
        cls_targets = torch.zeros(num_anchors, dtype=torch.long, device=locs.device)

        if label == 1:
            # this tile has waldo, convert the ground truth bbox to center format
            # bboxes are already on the right device because we moved them in the training loop
            gt_box = bbox_to_center(bboxes[i])

            '''
            find which anchor overlaps the most with the ground truth
            we use IoU (intersection over union) to measure overlap
            anchors are already on the right device from concatenate_all_heads
            '''
            ious = compute_iou(gt_box, anchors)
            best_anchor_idx = ious.argmax().item()

            '''
            also mark any anchor with IoU above 0.5 as positive
            this gives the model more positive examples to learn from
            instead of just the single best matching anchor
            '''
            positive_mask = ious > 0.5
            positive_mask[best_anchor_idx] = True # make sure the best one is always included

            # set classification targets for matched anchors to waldo (class 1)
            cls_targets[positive_mask] = 1

            '''
            compute localization loss only for the matched anchors
            we encode the ground truth as offsets from each matched anchor
            then compare against what the model predicted
            '''
            positive_indices = positive_mask.nonzero(as_tuple=True)[0]
            num_positive += len(positive_indices)

            if len(positive_indices) > 0:
                for idx in positive_indices:
                    # encode the ground truth bbox as offsets from this anchor
                    target_offsets = encode_ground_truth(gt_box, anchors[idx])
                    predicted_offsets = img_locs[idx]

                    '''
                    smooth L1 loss for bounding box regression
                    this is the standard loss used in SSD and other object detectors
                    its less sensitive to outliers than MSE loss
                    torch.nn.functional.smooth_l1_loss is from pytorch
                    '''
                    total_loc_loss += F.smooth_l1_loss(predicted_offsets, target_offsets)

        '''
        classification loss using focal cross entropy
        standard cross entropy treats every anchor equally but most anchors are
        easy background predictions. focal loss down weights those easy ones
        so the model focuses on the confusing anchors near waldo like characters
        '''
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


def train_one_epoch(model, dataloader, optimizer, device, scaler=None):
    '''
    runs one full pass through the training data
    handles both torchvision models (SSD, RetinaNet) and our custom SSD
    uses mixed precision (AMP) if a scaler is provided for ~2x speedup

    args:
        model: the detection model
        dataloader: training data loader
        optimizer: the optimizer
        device: torch device (cuda/mps/cpu)
        scaler: GradScaler for AMP (None to disable)

    returns:
        avg_loss, avg_cls_loss, avg_loc_loss for the epoch
    '''

    model.train() # put the model in training mode

    running_loss = 0.0
    running_cls_loss = 0.0
    running_loc_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None

    total_batches = len(dataloader)
    for batch_idx, (images, labels, bboxes) in enumerate(dataloader):
        # non_blocking=True overlaps CPU to GPU transfer with GPU computation
        # the GPU can keep processing the current batch while the next batch transfers
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bboxes = bboxes.to(device, non_blocking=True)

        # print progress so gabe can see its still running
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            print(f"  batch {batch_idx + 1}/{total_batches}", flush=True)

        '''
        mixed precision: run the forward pass in float16 for speed
        the autocast context manager handles the dtype conversions automatically
        we only use AMP on CUDA (not MPS or CPU) because thats where it helps
        '''
        with torch.amp.autocast("cuda", enabled=use_amp):
            if MODEL_TYPE in ("pretrained_ssd", "retinanet"):
                targets = format_targets(labels, bboxes, device)
                loss_dict = model(images, targets)

                if MODEL_TYPE == "pretrained_ssd":
                    focal_cls_loss = focal_loss_weight(loss_dict["classification"])
                    cls_loss = focal_cls_loss.item()
                    loc_loss = loss_dict["bbox_regression"].item()
                    loss = focal_cls_loss + loss_dict["bbox_regression"]
                else:
                    cls_loss = loss_dict["classification"].item()
                    loc_loss = loss_dict["bbox_regression"].item()
                    loss = loss_dict["classification"] + loss_dict["bbox_regression"]
            else:
                locs, cls_scores, anchors = model(images)
                loss, cls_loss, loc_loss = compute_loss(locs, cls_scores, anchors, labels, bboxes)

        # backward pass with gradient scaling for mixed precision
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # clip gradients to prevent exploding updates when the loss is very high
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # update the model weights
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

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
    '''
    runs one pass through validation data without updating weights
    used to check for overfitting (train loss down but val loss up)

    args:
        model: the detection model
        dataloader: validation data loader
        device: torch device

    returns:
        avg_loss, avg_cls_loss, avg_loc_loss for the validation set
    '''

    model.eval() # put the model in eval mode

    running_loss = 0.0
    running_cls_loss = 0.0
    running_loc_loss = 0.0
    num_batches = 0

    # torch.no_grad tells pytorch not to track gradients
    # this saves memory and speeds things up since we dont need gradients for validation
    #
    '''
    for torchvision models (SSD and RetinaNet) we need train mode to get losses
    because in eval mode they return detections instead of losses
    we still use no_grad so no weights get updated
    '''
    if MODEL_TYPE in ("pretrained_ssd", "retinanet"):
        model.train()

    use_amp = torch.cuda.is_available() # only use AMP on CUDA

    with torch.no_grad():
        for images, labels, bboxes in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
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


def run_training(checkpoint_dir=None, model_override=None):
    '''
    main training function, can be called directly or from experiment scripts
    handles model creation, optimizer setup, training loop, and checkpointing

    args:
        checkpoint_dir: where to save weights (defaults to CHECKPOINTS_DIR)
        model_override: pre-created model to use instead of create_model()

    returns:
        model: the trained model
        best_val_loss: best validation loss achieved
        history: dict with train/val loss lists for each epoch
    '''

    save_dir = checkpoint_dir or CHECKPOINTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    print("starting training...")
    start_time = time.time()

    # create the model and move it to the gpu
    if model_override is not None:
        model = model_override.to(device)
    else:
        model = create_model()
        model = model.to(device)
    model_type = model.model_type
    print(f"model type: {model_type}")
    print(f"model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    '''
    create the optimizer
    for pretrained models (SSD and RetinaNet) the backbone is frozen
    so we only train the detection heads and the 1x1 preprocessing conv
    SGD with momentum works well for fine tuning pretrained models
    weight decay (L2 regularization) prevents the heads from overfitting
    '''
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

    '''
    load the train and validation dataloaders
    the train loader uses weighted sampling to handle class imbalance
    the val loader just loads tiles normally without any augmentation
    '''
    print("loading data...")
    train_loader = get_dataloader(split="train", tile_size=256, batch_size=BATCH_SIZE)
    val_loader = get_dataloader(split="val", tile_size=256, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  train: {len(train_loader.dataset)} tiles")
    print(f"  val: {len(val_loader.dataset)} tiles")

    '''
    learning rate scheduler
    if val loss stops improving for LR_SCHEDULER_PATIENCE epochs, cut the learning rate
    this helps the model converge when its stuck at a plateau instead of bouncing around
    torch.optim.lr_scheduler.ReduceLROnPlateau is a standard pytorch scheduler
    '''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min", # we want to minimize val loss
        patience=LR_SCHEDULER_PATIENCE,
        factor=LR_SCHEDULER_FACTOR
    )

    '''
    mixed precision training with automatic gradient scaling
    AMP (automatic mixed precision) runs the forward pass in float16
    which is roughly 2x faster and uses half the GPU memory
    the GradScaler adjusts gradient magnitudes to prevent underflow in float16
    we only use this on CUDA since MPS and CPU dont benefit from AMP
    '''
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
    if scaler:
        print(f"  using mixed precision (AMP) for faster training")

    # keep track of the best validation loss so we know when to save
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0 # for early stopping

    # save loss history so evaluate.py can plot training curves later
    # we store every epochs losses in a list and save to json at the end
    history = {"train_loss": [], "val_loss": [], "train_cls": [], "val_cls": [], "train_loc": [], "val_loc": []}

    # make sure the checkpoints directory exists
    os.makedirs(save_dir, exist_ok=True)

    # the main training loop
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # train for one epoch
        train_loss, train_cls, train_loc = train_one_epoch(model, train_loader, optimizer, device, scaler)

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
                "use_hsv": getattr(model, "use_hsv", True),
                "use_sobel": getattr(model, "use_sobel", True),
            }, os.path.join(save_dir, "best_model.pth"))
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
            "use_hsv": getattr(model, "use_hsv", True),
            "use_sobel": getattr(model, "use_sobel", True),
        }, os.path.join(save_dir, "last_model.pth"))

        # print a summary for this epoch including the current learning rate
        # flush=True forces the output to show up immediately even when piped through tee
        print(f"epoch {epoch + 1}/{NUM_EPOCHS} ({epoch_time:.1f}s)"
              f"  train loss: {train_loss:.4f} (cls: {train_cls:.4f}, loc: {train_loc:.4f})"
              f"  val loss: {val_loss:.4f} (cls: {val_cls:.4f}, loc: {val_loc:.4f})"
              f"  lr: {current_lr:.6f}"
              f"{saved}", flush=True)

        # early stopping: if val loss hasnt improved for too long, stop training
        # no point continuing if the model has stopped learning
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nearly stopping: val loss hasnt improved for {EARLY_STOPPING_PATIENCE} epochs")
            print(f"best was {best_val_loss:.4f} at epoch {best_epoch}")
            break

    # save the training history to a json file so evaluate.py can plot curves
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # print the final summary
    total_time = time.time() - start_time
    print(f"\ntraining complete in {total_time:.1f} seconds")
    print(f"best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"checkpoints saved to {save_dir}")
    print(f"training history saved to {history_path}")

    return model, best_val_loss, history


# when run directly (python train.py), use the default config paths
if __name__ == "__main__":
    run_training()
