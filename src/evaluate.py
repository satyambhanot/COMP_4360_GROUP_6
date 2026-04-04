'''
evaluate.py
loads the best trained model and evaluates it on test tiles the model
has never seen before. measures how well it finds waldo using precision,
recall, F1, and generates 10+ visualization plots for the report.

includes HSV pre-filter (skip tiles with no red/white) and test-time
augmentation (horizontal flip). the run_evaluation() function can be
called from experiment scripts with custom paths.

usage: python evaluate.py (uses default config settings)
'''

import sys
import os
import json
import time

# add this directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib
matplotlib.use("Agg") # use non interactive backend so it works without a display
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms

from models.model import WaldoDetector, WaldoDetectorPretrained, WaldoDetectorRetinaNet
from data.data_pipeline import get_dataloader
from models.nms import apply_nms
from utils import get_device
from config import (
    BATCH_SIZE, INPUT_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, RAW_IMAGES,
    MODEL_TYPE, NMS_CONFIDENCE_THRESHOLD,
    HSV_FILTER_THRESHOLD, USE_TTA,
    RETINANET_INPUT_SIZE
)
from data.pre_processing import apply_hsv_mask

# where we save all the output images and results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

device = get_device()
print(f"using device: {device}")


def load_model():
    '''
    loads the trained model from the best checkpoint
    reads model_type from the checkpoint to create the right architecture

    returns:
        loaded model in eval mode, or None if no checkpoint exists
    '''

    # read CHECKPOINTS_DIR from config at call time, not import time
    # this way experiment scripts can override it before calling load_model
    import config as _cfg
    ckpt_dir = _cfg.CHECKPOINTS_DIR
    checkpoint_path = os.path.join(ckpt_dir, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"no checkpoint found at {checkpoint_path}")
        print("you need to train the model first by running: python train.py")
        return None

    '''
    load the checkpoint first to check which model type was used during training
    map_location makes sure it works even if the checkpoint was saved on GPU
    but we are loading on CPU or vice versa
    '''
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # check which model was used for training
    # older checkpoints might not have model_type so we default to the config
    model_type = checkpoint.get("model_type", MODEL_TYPE)

    # read preprocessing flags saved during training
    # older checkpoints might not have these so default to both enabled
    use_hsv = checkpoint.get("use_hsv", True)
    use_sobel = checkpoint.get("use_sobel", True)

    # create the right model based on what was trained
    if model_type == "retinanet":
        model = WaldoDetectorRetinaNet(use_hsv=use_hsv, use_sobel=use_sobel)
    elif model_type == "pretrained_ssd":
        model = WaldoDetectorPretrained(use_hsv=use_hsv, use_sobel=use_sobel)
    else:
        model = WaldoDetector(use_hsv=use_hsv, use_sobel=use_sobel)

    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # put the model in eval mode
    # this is important because it changes how some layers behave
    model.eval()
    model.model_type = model_type

    print(f"loaded checkpoint from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")
    return model


def draw_boxes(image, boxes, labels=None, scores=None, color="red", width=2):
    '''
    takes a PIL image and draws bounding boxes on it
    used for visualizing what the model predicted
    '''
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box.tolist()
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)

        text = ""
        if labels is not None:
            text += f"{labels[i]}"
        if scores is not None:
            text += f": {scores[i]:.2f}"

        if text:
            draw.text((x_min, y_min), text, fill=color)

    return image


def create_tiles_from_image(image, tile_size, overlap):
    '''
    takes a full wheres waldo image and chops it into tiles
    we keep track of each tiles position so we can map predictions
    back to the original image later
    '''
    tiles = []
    tile_coords = []

    width, height = image.size

    for h in range(0, height, tile_size - overlap):
        for w in range(0, width, tile_size - overlap):
            tile = image.crop((w, h, w + tile_size, h + tile_size))
            tiles.append(tile)
            tile_coords.append((w, h))

    return tiles, tile_coords


def unnormalize(tensor):
    '''
    the dataloader normalizes images with imagenet mean and std
    to display them as normal images we need to undo that
    otherwise the colors look completely wrong
    '''
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def run_predictions(model, dataloader, device):
    '''
    runs all test tiles through the model with HSV pre-filter and optional TTA
    collects predictions, labels, confidences, and NMS results for each tile

    args:
        model: trained detection model
        dataloader: test data loader
        device: torch device

    returns:
        predictions: list of 0/1 for each tile
        labels: list of ground truth 0/1
        confidences: list of max waldo confidence per tile
        nms_results: list of (boxes, scores) tuples per tile
    '''
    # we use our HSV mask instead since we already built it for preprocessing

    all_predictions = [] # 1 if model thinks waldo is in this tile, 0 if not
    all_labels = [] # the actual ground truth label (0 or 1)
    all_confidences = [] # the highest waldo confidence score for each tile
    all_nms_results = [] # the filtered boxes and scores from NMS for visualization
    hsv_filtered = 0 # count how many tiles we skipped with the HSV pre filter

    use_amp = torch.cuda.is_available()

    with torch.no_grad():
        for images, labels, bboxes in dataloader:
            images = images.to(device, non_blocking=True)

            '''
            HSV pre filter: check each tile for red/white pixels before running the model
            compute the HSV mask and see what fraction of pixels light up
            tiles with no waldo colors get marked as negative without model inference
            '''
            hsv_mask = apply_hsv_mask(images)
            # hsv_mask is [N, 1, H, W], compute the mean per tile
            hsv_scores = hsv_mask.view(hsv_mask.shape[0], -1).mean(dim=1) # [N]

            # figure out which tiles pass the HSV filter
            passes_filter = hsv_scores >= HSV_FILTER_THRESHOLD

            '''
            prepare default results for every tile in this batch
            we fill in the defaults first (assume negative) then overwrite
            with actual model predictions for tiles that passed the filter
            this keeps results in the correct order matching the dataset
            '''
            batch_preds = [0] * len(labels)
            batch_confs = [0.0] * len(labels)
            batch_nms = [(torch.zeros(0, 4), torch.zeros(0))] * len(labels)

            # count how many got filtered out
            for i in range(len(labels)):
                if not passes_filter[i]:
                    hsv_filtered += 1

            # if all tiles in this batch were filtered out, skip model inference
            if not passes_filter.any():
                for i in range(len(labels)):
                    all_predictions.append(batch_preds[i])
                    all_labels.append(labels[i].item())
                    all_confidences.append(batch_confs[i])
                    all_nms_results.append(batch_nms[i])
                continue

            # only run the model on tiles that passed the filter
            filtered_images = images[passes_filter]
            filtered_indices = passes_filter.nonzero(as_tuple=True)[0]

            '''
            also run on flipped versions if TTA is enabled
            waldo looks the same facing left or right so flipping is safe
            we take the max confidence between original and flipped
            '''
            flipped_confs = {}
            if USE_TTA:
                flipped_images = torch.flip(filtered_images, dims=[3]) # flip horizontally
                if hasattr(model, "model_type") and model.model_type in ("pretrained_ssd", "retinanet"):
                    flip_dets = model(flipped_images)
                    for j in range(len(flip_dets)):
                        det = flip_dets[j]
                        waldo_mask = det["labels"] == 1
                        scores = det["scores"][waldo_mask]
                        flipped_confs[j] = scores.max().item() if scores.shape[0] > 0 else 0.0
                else:
                    flip_locs, flip_cls, flip_anchors = model(flipped_images)
                    flip_nms = apply_nms(flip_locs, flip_cls, flip_anchors)
                    for j in range(filtered_images.shape[0]):
                        _, scores = flip_nms[j]
                        flipped_confs[j] = scores.max().item() if scores.shape[0] > 0 else 0.0

            if hasattr(model, "model_type") and model.model_type in ("pretrained_ssd", "retinanet"):
                # torchvision models return detections directly in eval mode
                detections = model(filtered_images)

                for j in range(len(detections)):
                    det = detections[j]
                    orig_idx = filtered_indices[j].item()

                    waldo_mask = det["labels"] == 1
                    boxes = det["boxes"][waldo_mask]
                    scores = det["scores"][waldo_mask]

                    if boxes.shape[0] > 0:
                        prediction = 1
                        confidence = scores.max().item()
                    else:
                        prediction = 0
                        confidence = 0.0

                    if USE_TTA and j in flipped_confs:
                        if flipped_confs[j] > confidence:
                            confidence = flipped_confs[j]
                            prediction = 1

                    # write into the correct position in the batch arrays
                    batch_preds[orig_idx] = prediction
                    batch_confs[orig_idx] = confidence
                    batch_nms[orig_idx] = (boxes.cpu(), scores.cpu())

            else:
                # custom model returns raw predictions, we run our own NMS
                locs, cls_scores, anchors = model(filtered_images)
                nms_results = apply_nms(locs, cls_scores, anchors)

                for j in range(filtered_images.shape[0]):
                    boxes, scores = nms_results[j]
                    orig_idx = filtered_indices[j].item()

                    if boxes.shape[0] > 0:
                        prediction = 1
                        confidence = scores.max().item()
                    else:
                        prediction = 0
                        confidence = 0.0

                    if USE_TTA and j in flipped_confs:
                        if flipped_confs[j] > confidence:
                            confidence = flipped_confs[j]
                            prediction = 1

                    batch_preds[orig_idx] = prediction
                    batch_confs[orig_idx] = confidence
                    batch_nms[orig_idx] = (boxes, scores)

            # append all results for this batch in the correct order
            # this ensures results line up with dataset.tiles for image level metrics
            for i in range(len(labels)):
                all_predictions.append(batch_preds[i])
                all_labels.append(labels[i].item())
                all_confidences.append(batch_confs[i])
                all_nms_results.append(batch_nms[i])

    print(f"  HSV pre-filter skipped {hsv_filtered} tiles with no red/white pixels")
    return all_predictions, all_labels, all_confidences, all_nms_results


def compute_metrics(predictions, labels):
    '''
    computes accuracy, precision, recall, F1 and prints confusion matrix

    args:
        predictions: list of 0/1 predictions
        labels: list of 0/1 ground truth

    returns:
        dict with accuracy, precision, recall, f1, tp, fp, tn, fn
    '''

    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1) # correctly found waldo
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0) # false alarm
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0) # correctly said no waldo
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1) # missed waldo

    total = len(predictions)
    accuracy = (tp + tn) / total if total > 0 else 0

    # precision: when we say waldo, how often are we right
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # recall: of all tiles with waldo, how many did we find
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # f1: combines precision and recall into one number
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # print the confusion matrix
    print("\nconfusion matrix:")
    print(f"                  predicted waldo    predicted no waldo")
    print(f"  actual waldo         {tp:>5}               {fn:>5}")
    print(f"  actual no waldo      {fp:>5}               {tn:>5}")

    # print the metrics
    print(f"\nmetrics:")
    print(f"  total tiles: {total}")
    print(f"  accuracy:  {accuracy:.4f} ({tp + tn}/{total})")
    print(f"  precision: {precision:.4f} ({tp}/{tp + fp})")
    print(f"  recall:    {recall:.4f} ({tp}/{tp + fn})")
    print(f"  f1 score:  {f1:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


def compute_image_level_metrics(predictions, labels, confidences, dataset):
    '''
    the tile level metrics above tell us stuff like "precision is 0.24 across all tiles"
    but thats not really what we care about in practice
    what we actually want to know is: if i give you a full wheres waldo image
    can you tell me where waldo is? thats an image level question not a tile level one
    
    so we group all the test tiles back by their source image and ask:
      for each image that has waldo, did the model detect him in at least one tile?
      for each image, how many tiles gave false alarms?
    this gives us a much better picture of how useful the model actually is
    
    example: if we have 500 tiles from one image and 1 tile correctly finds waldo
    but 3 tiles have false positives, the tile level precision looks bad (1/4 = 25%)
    but at the image level we found waldo and only had 3 false alarms which is fine
    '''

    # the dataset tiles list is in the same order as the dataloader output
    # so we can just loop through both at the same time
    image_results = {} # image_id -> {"has_waldo": bool, "found_waldo": bool, "false_positives": int}

    for i, tile_info in enumerate(dataset.tiles):
        if i >= len(predictions):
            break

        img_id = tile_info["image_id"]
        if img_id not in image_results:
            image_results[img_id] = {
                "has_waldo": False,
                "found_waldo": False,
                "false_positives": 0,
                "total_tiles": 0
            }

        image_results[img_id]["total_tiles"] += 1

        if labels[i] == 1:
            image_results[img_id]["has_waldo"] = True
            if predictions[i] == 1:
                image_results[img_id]["found_waldo"] = True

        if predictions[i] == 1 and labels[i] == 0:
            image_results[img_id]["false_positives"] += 1

    # compute image level stats
    images_with_waldo = [img_id for img_id, r in image_results.items() if r["has_waldo"]]
    images_waldo_found = [img_id for img_id in images_with_waldo if image_results[img_id]["found_waldo"]]
    images_clean = [img_id for img_id, r in image_results.items() if r["false_positives"] == 0]
    total_fp = sum(r["false_positives"] for r in image_results.values())

    num_images = len(image_results)
    num_with_waldo = len(images_with_waldo)
    num_found = len(images_waldo_found)
    num_clean = len(images_clean)

    detection_rate = num_found / num_with_waldo if num_with_waldo > 0 else 0
    clean_rate = num_clean / num_images if num_images > 0 else 0
    avg_fp = total_fp / num_images if num_images > 0 else 0

    print(f"\nimage level metrics ({num_images} test images):")
    print(f"  images containing waldo: {num_with_waldo}")
    print(f"  waldo found in: {num_found}/{num_with_waldo} ({detection_rate:.1%})")
    print(f"  images with zero false positives: {num_clean}/{num_images} ({clean_rate:.1%})")
    print(f"  average false positives per image: {avg_fp:.1f}")
    print(f"  total false positives across all images: {total_fp}")

    return {
        "num_test_images": num_images,
        "images_with_waldo": num_with_waldo,
        "waldo_found_in": num_found,
        "image_detection_rate": detection_rate,
        "images_no_false_positives": num_clean,
        "clean_image_rate": clean_rate,
        "avg_false_positives_per_image": avg_fp,
        "total_false_positives": total_fp
    }


def analyze_thresholds(confidences, labels):
    '''
    tests different NMS confidence thresholds to find the best one
    a higher threshold means fewer false positives but might miss waldos
    a lower threshold catches more waldos but also more false alarms
    this helps us pick the right threshold and shows the tradeoff in the report
    '''

    print("\nthreshold analysis (what happens at different confidence cutoffs):")
    print(f"  {'threshold':>10}  {'precision':>10}  {'recall':>10}  {'f1':>10}  {'detected':>10}  {'false pos':>10}")

    best_f1 = 0
    best_threshold = 0.5

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = [1 if c >= threshold else 0 for c in confidences]
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  {threshold:>10.1f}  {precision:>10.4f}  {recall:>10.4f}  {f1:>10.4f}  {tp:>10}  {fp:>10}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\n  best f1 score: {best_f1:.4f} at threshold {best_threshold}")
    return best_threshold, best_f1


def plot_precision_recall_curve(confidences, labels):
    '''
    plots a precision vs recall curve showing the tradeoff at different thresholds
    this is a standard chart in object detection papers
    each point represents a different confidence threshold
    the curve shows: if we want higher recall we sacrifice precision and vice versa
    '''

    precisions = []
    recalls = []
    f1_scores = []
    thresholds = [t / 100 for t in range(5, 96, 5)] # 0.05 to 0.95 in steps of 0.05

    best_f1 = 0
    best_p = 0
    best_r = 0

    for threshold in thresholds:
        preds = [1 if c >= threshold else 0 for c in confidences]
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_p = p
            best_r = r

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker="o", markersize=4, linewidth=2)
    # mark the best F1 point
    if best_r > 0 or best_p > 0:
        plt.scatter([best_r], [best_p], color="red", s=100, zorder=5, label=f"best F1={best_f1:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision vs Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close()
    print(f"saved precision recall curve to results/precision_recall_curve.png")


def save_results(metrics, confidences, labels, best_threshold=0.5, best_f1=0, image_metrics=None, inference_time=0, num_tiles=0):
    '''
    saves all the evaluation results to a json file
    this makes it easy to load the numbers into the report
    or compare against ablation study results later
    '''

    results = {
        "metrics": metrics,
        "best_threshold": best_threshold,
        "best_f1_at_optimal_threshold": best_f1,
        "total_positive_tiles": sum(1 for l in labels if l == 1),
        "total_negative_tiles": sum(1 for l in labels if l == 0),
        "avg_confidence_positive": sum(c for c, l in zip(confidences, labels) if l == 1) / max(sum(1 for l in labels if l == 1), 1),
        "avg_confidence_negative": sum(c for c, l in zip(confidences, labels) if l == 0) / max(sum(1 for l in labels if l == 0), 1),
        "max_confidence_positive": max((c for c, l in zip(confidences, labels) if l == 1), default=0),
        "max_confidence_negative": max((c for c, l in zip(confidences, labels) if l == 0), default=0),
        "inference_time_seconds": round(inference_time, 2),
        "ms_per_tile": round(inference_time / max(num_tiles, 1) * 1000, 1),
    }

    # include image level metrics if we computed them
    if image_metrics is not None:
        results["image_level"] = image_metrics

    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved evaluation results to {results_path}")


def plot_training_curves():
    '''
    loads the training history json and plots the loss curves
    this shows whether the model was learning and when it started overfitting
    we plot train and val loss on the same chart for easy comparison
    '''

    # read from config at call time so experiment scripts can override the path
    import config as _cfg
    history_path = os.path.join(_cfg.CHECKPOINTS_DIR, "training_history.json")

    if not os.path.exists(history_path):
        print("no training history found, skipping training curves")
        print("(train.py needs to be run first to generate training_history.json)")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    # plot total loss (train vs val)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("total loss")
    plt.title("Total Loss")
    plt.legend()

    # plot classification loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_cls"], label="train")
    plt.plot(epochs, history["val_cls"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("cls loss")
    plt.title("Classification Loss")
    plt.legend()

    # plot localization loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_loc"], label="train")
    plt.plot(epochs, history["val_loc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loc loss")
    plt.title("Localization Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print(f"saved training curves to results/training_curves.png")


def plot_confidence_distribution(confidences, labels):
    '''
    shows the distribution of confidence scores for positive vs negative tiles
    if the model is working well, positive tiles should have high confidence
    and negative tiles should have low confidence
    if the two distributions overlap a lot the model is confused
    '''

    pos_conf = [c for c, l in zip(confidences, labels) if l == 1]
    neg_conf = [c for c, l in zip(confidences, labels) if l == 0]

    plt.figure(figsize=(8, 5))
    plt.hist(neg_conf, bins=50, alpha=0.6, label=f"no waldo ({len(neg_conf)} tiles)", color="blue")
    plt.hist(pos_conf, bins=50, alpha=0.6, label=f"has waldo ({len(pos_conf)} tiles)", color="red")
    plt.axvline(x=NMS_CONFIDENCE_THRESHOLD, color="black", linestyle=":", label=f"confidence threshold ({NMS_CONFIDENCE_THRESHOLD})")
    plt.xlabel("waldo confidence score")
    plt.ylabel("number of tiles")
    plt.title("Confidence Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confidence_distribution.png"), dpi=150)
    plt.close()
    print(f"saved confidence distribution to results/confidence_distribution.png")


def plot_confusion_matrix(predictions, labels):
    '''
    draws the confusion matrix as a colored heatmap image
    way more useful in a report than just printing numbers to the console
    green cells for correct predictions, red cells for errors
    the professor will expect to see this in any ML project
    '''

    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

    matrix = [[tp, fn], [fp, tn]]
    labels_text = [["True Pos", "False Neg"], ["False Pos", "True Neg"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    # green for correct (TP, TN), red for errors (FP, FN)
    colors = [["#4CAF50", "#F44336"], ["#F44336", "#4CAF50"]]

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, fill=True,
                         color=colors[i][j], alpha=0.6))
            ax.text(j + 0.5, 1.5 - i, f"{labels_text[i][j]}\n{matrix[i][j]}",
                    ha="center", va="center", fontsize=14, fontweight="bold")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted Waldo", "Predicted No Waldo"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual No Waldo", "Actual Waldo"])
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"saved confusion matrix to results/confusion_matrix.png")


def plot_f1_vs_threshold(confidences, labels):
    '''
    shows how the F1 score changes as we adjust the confidence threshold
    this is helpful because it answers "what threshold should we use?"
    a curve that peaks sharply means theres a clear best threshold
    a flat curve means the model isnt very sensitive to the threshold choice
    we also plot precision and recall alongside so you can see the tradeoff
    '''

    thresholds = [t / 100 for t in range(5, 96, 5)]
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        preds = [1 if c >= threshold else 0 for c in confidences]
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    # find the peak F1
    best_idx = f1_scores.index(max(f1_scores)) if f1_scores else 0
    best_t = thresholds[best_idx] if thresholds else 0
    best_f1 = f1_scores[best_idx] if f1_scores else 0

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores, marker="o", markersize=4, linewidth=2, label="F1 Score", color="blue")
    plt.plot(thresholds, precisions, marker="s", markersize=3, linewidth=1.5, label="Precision", color="green", alpha=0.7)
    plt.plot(thresholds, recalls, marker="^", markersize=3, linewidth=1.5, label="Recall", color="orange", alpha=0.7)

    # mark the best F1 threshold
    plt.scatter([best_t], [best_f1], color="red", s=100, zorder=5, label=f"Best F1={best_f1:.3f} at t={best_t:.2f}")
    plt.axvline(x=best_t, color="red", linestyle=":", alpha=0.5)

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.title("F1, Precision, and Recall vs Confidence Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "f1_vs_threshold.png"), dpi=150)
    plt.close()
    print(f"saved f1 vs threshold curve to results/f1_vs_threshold.png")


def plot_roc_curve(confidences, labels):
    '''
    ROC curve shows false positive rate vs true positive rate
    this is one of the standard evaluation plots in machine learning
    the area under the curve (AUC) tells you overall how good the model is
    AUC of 1.0 means perfect, 0.5 means random guessing
    the professor will definitely expect to see this
    '''

    # compute ROC points at many thresholds
    thresholds = [t / 200 for t in range(0, 201)]
    fpr_list = []
    tpr_list = []

    total_positive = sum(1 for l in labels if l == 1)
    total_negative = sum(1 for l in labels if l == 0)

    for threshold in thresholds:
        preds = [1 if c >= threshold else 0 for c in confidences]
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)

        tpr = tp / total_positive if total_positive > 0 else 0
        fpr = fp / total_negative if total_negative > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # compute AUC using the trapezoidal rule
    # sort by fpr so the curve goes left to right
    sorted_points = sorted(zip(fpr_list, tpr_list))
    auc = 0
    for i in range(1, len(sorted_points)):
        dx = sorted_points[i][0] - sorted_points[i - 1][0]
        avg_y = (sorted_points[i][1] + sorted_points[i - 1][1]) / 2
        auc += dx * avg_y

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_list, tpr_list, linewidth=2, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Random Classifier (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print(f"saved ROC curve to results/roc_curve.png (AUC = {auc:.3f})")


def plot_per_image_summary(predictions, labels, dataset):
    '''
    creates a visual summary showing how the model performed on each test image
    for each image we show: image id, total tiles, waldo found?, false positives
    this is useful for the report because it shows which specific images are
    easy vs hard for the model and helps explain where the model struggles
    '''

    # group results by image id (same logic as compute_image_level_metrics)
    image_results = {}
    for i, tile_info in enumerate(dataset.tiles):
        if i >= len(predictions):
            break
        img_id = tile_info["image_id"]
        if img_id not in image_results:
            image_results[img_id] = {
                "has_waldo": False, "found_waldo": False,
                "false_positives": 0, "total_tiles": 0
            }
        image_results[img_id]["total_tiles"] += 1
        if labels[i] == 1:
            image_results[img_id]["has_waldo"] = True
            if predictions[i] == 1:
                image_results[img_id]["found_waldo"] = True
        if predictions[i] == 1 and labels[i] == 0:
            image_results[img_id]["false_positives"] += 1

    # sort by image id for consistent display
    sorted_ids = sorted(image_results.keys(), key=lambda x: (len(x), x))

    # build table data
    rows = []
    for img_id in sorted_ids:
        r = image_results[img_id]
        found = "yes" if r["found_waldo"] else ("MISSED" if r["has_waldo"] else "n/a")
        rows.append([img_id, str(r["total_tiles"]), found, str(r["false_positives"])])

    # draw as a table image
    # figure out how tall the table needs to be
    num_rows = len(rows)
    fig_height = max(4, 0.4 * num_rows + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")

    col_labels = ["Image ID", "Tiles", "Waldo Found?", "False Positives"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    # color code the rows
    for i, row_data in enumerate(rows):
        row_idx = i + 1 # +1 because row 0 is the header
        found_text = row_data[2]
        fp_count = int(row_data[3])

        if found_text == "yes" and fp_count == 0:
            # perfect: found waldo, no false positives
            for j in range(4):
                table[row_idx, j].set_facecolor("#C8E6C9")
        elif found_text == "MISSED":
            # bad: missed waldo
            for j in range(4):
                table[row_idx, j].set_facecolor("#FFCDD2")
        elif fp_count > 5:
            # lots of false positives
            for j in range(4):
                table[row_idx, j].set_facecolor("#FFF9C4")

    # style the header
    for j in range(4):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Per Image Detection Summary", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_image_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved per image summary to results/per_image_summary.png")


def visualize_samples(model, dataloader, device, num_samples=5):
    '''
    grabs some test tiles and draws the model predictions on them
    saves the images so we can put them in the report
    we try to get a mix of true positives, false positives, and false negatives
    '''

    tp_images = [] # correctly found waldo
    fp_images = [] # falsely said waldo
    fn_images = [] # missed waldo

    with torch.no_grad():
        for images, labels, bboxes in dataloader:
            images = images.to(device, non_blocking=True)

            # get detections based on model type
            if hasattr(model, "model_type") and model.model_type in ("pretrained_ssd", "retinanet"):
                detections = model(images)
                tile_results = []
                for det in detections:
                    waldo_mask = det["labels"] == 1
                    tile_results.append((det["boxes"][waldo_mask].cpu(), det["scores"][waldo_mask].cpu()))
            else:
                locs, cls_scores, anchors = model(images)
                nms_results = apply_nms(locs, cls_scores, anchors)
                tile_results = nms_results

            for i in range(images.shape[0]):
                boxes, scores = tile_results[i]
                label = labels[i].item()
                predicted = 1 if boxes.shape[0] > 0 else 0

                # unnormalize the image so it looks like a real photo
                img_tensor = unnormalize(images[i].cpu())
                # convert tensor to PIL image for drawing
                img_pil = transforms.ToPILImage()(img_tensor)

                if predicted == 1 and label == 1 and len(tp_images) < num_samples:
                    img_pil = draw_boxes(img_pil, boxes.cpu(), scores=scores.cpu(), color="green")
                    tp_images.append(img_pil)

                elif predicted == 1 and label == 0 and len(fp_images) < num_samples:
                    img_pil = draw_boxes(img_pil, boxes.cpu(), scores=scores.cpu(), color="red")
                    fp_images.append(img_pil)

                elif predicted == 0 and label == 1 and len(fn_images) < num_samples:
                    # draw the ground truth box so we can see where waldo actually was
                    gt_box = bboxes[i].unsqueeze(0)
                    img_pil = draw_boxes(img_pil, gt_box, color="yellow")
                    fn_images.append(img_pil)

            # stop if we have enough samples of each type
            if len(tp_images) >= num_samples and len(fp_images) >= num_samples and len(fn_images) >= num_samples:
                break

    # save the sample images
    for j, img in enumerate(tp_images):
        img.save(os.path.join(RESULTS_DIR, f"tp_{j}.png"))
    for j, img in enumerate(fp_images):
        img.save(os.path.join(RESULTS_DIR, f"fp_{j}.png"))
    for j, img in enumerate(fn_images):
        img.save(os.path.join(RESULTS_DIR, f"fn_{j}.png"))

    print(f"saved {len(tp_images)} true positive, {len(fp_images)} false positive, {len(fn_images)} false negative samples to results/")


def run_demo(model, device, image_path=None):
    '''
    takes a full wheres waldo image, tiles it up, runs every tile
    through the model, and draws boxes where it thinks waldo is
    this is the final "look it found waldo" image for the presentation
    '''

    # pick the first test image if none specified
    if image_path is None:
        raw_images = sorted(os.listdir(RAW_IMAGES))
        if len(raw_images) == 0:
            print("no images found in data/raw/images")
            return
        image_path = os.path.join(RAW_IMAGES, raw_images[0])

    print(f"running demo on {image_path}...")
    image = Image.open(image_path).convert("RGB")

    # tile the image the same way data_pipeline does
    tiles, tile_coords = create_tiles_from_image(image, INPUT_SIZE, overlap=0)

    # apply the same transforms the dataloader uses
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # process tiles in batches to keep it fast
    all_detections = [] # (x_offset, y_offset, boxes, scores) for each tile with detections

    for batch_start in range(0, len(tiles), BATCH_SIZE):
        batch_tiles = tiles[batch_start:batch_start + BATCH_SIZE]
        batch_coords = tile_coords[batch_start:batch_start + BATCH_SIZE]

        # transform tiles to tensors and stack into a batch
        tile_tensors = torch.stack([transform(t) for t in batch_tiles])
        tile_tensors = tile_tensors.to(device, non_blocking=True)

        with torch.no_grad():
            if hasattr(model, "model_type") and model.model_type in ("pretrained_ssd", "retinanet"):
                # pretrained SSD returns detections directly
                detections = model(tile_tensors)
                for i in range(len(batch_tiles)):
                    det = detections[i]
                    waldo_mask = det["labels"] == 1
                    boxes = det["boxes"][waldo_mask]
                    scores = det["scores"][waldo_mask]
                    if boxes.shape[0] > 0:
                        w_off, h_off = batch_coords[i]
                        # boxes are in model input space, scale back to tile size
                        # SSD uses 300x300, RetinaNet uses 512x512
                        model_input = RETINANET_INPUT_SIZE if model.model_type == "retinanet" else 300.0
                        scale = INPUT_SIZE / model_input
                        adjusted_boxes = boxes.cpu().clone() * scale
                        adjusted_boxes[:, 0] += w_off
                        adjusted_boxes[:, 1] += h_off
                        adjusted_boxes[:, 2] += w_off
                        adjusted_boxes[:, 3] += h_off
                        all_detections.append((adjusted_boxes, scores.cpu()))
            else:
                locs, cls_scores, anchors = model(tile_tensors)
                nms_results = apply_nms(locs, cls_scores, anchors)
                for i in range(len(batch_tiles)):
                    boxes, scores = nms_results[i]
                    if boxes.shape[0] > 0:
                        w_off, h_off = batch_coords[i]
                        adjusted_boxes = boxes.cpu().clone()
                        adjusted_boxes[:, 0] += w_off
                        adjusted_boxes[:, 1] += h_off
                        adjusted_boxes[:, 2] += w_off
                        adjusted_boxes[:, 3] += h_off
                        all_detections.append((adjusted_boxes, scores.cpu()))

    # draw all detections on the original full image
    for boxes, scores in all_detections:
        image = draw_boxes(image, boxes, scores=scores, color="green", width=3)

    # save with a unique name based on the source image
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    demo_path = os.path.join(RESULTS_DIR, f"demo_{img_name}.png")
    image.save(demo_path)
    print(f"found detections in {len(all_detections)} tiles")
    print(f"saved demo image to {demo_path}")


def run_evaluation(output_dir=None, checkpoint_dir=None):
    '''
    main evaluation function, can be called directly or from experiment scripts
    loads model, runs predictions, computes metrics, generates all plots

    args:
        output_dir: where to save plots and json (defaults to results/)
        checkpoint_dir: where to load model from (defaults to checkpoints/)

    returns:
        metrics dict with precision, recall, f1, etc
    '''

    global RESULTS_DIR
    original_results_dir = RESULTS_DIR
    if output_dir is not None:
        RESULTS_DIR = output_dir

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # load the trained model
    if checkpoint_dir is not None:
        # temporarily override CHECKPOINTS_DIR for loading
        import config as cfg
        original_ckpt_dir = cfg.CHECKPOINTS_DIR
        cfg.CHECKPOINTS_DIR = checkpoint_dir

    model = load_model()

    if checkpoint_dir is not None:
        cfg.CHECKPOINTS_DIR = original_ckpt_dir

    if model is None:
        RESULTS_DIR = original_results_dir
        return None

    # load the test data
    print("\nloading test data...")
    test_loader = get_dataloader(split="test", tile_size=256, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  test set: {len(test_loader.dataset)} tiles")

    # run predictions on every test tile and measure how long it takes
    print("\nrunning predictions on test set...")
    inference_start = time.time()
    predictions, labels, confidences, nms_results = run_predictions(model, test_loader, device)
    inference_time = time.time() - inference_start
    num_tiles = len(predictions)
    time_per_tile = inference_time / max(num_tiles, 1)
    print(f"  inference time: {inference_time:.1f}s for {num_tiles} tiles ({time_per_tile * 1000:.1f}ms per tile)")

    # compute and print tile level metrics
    metrics = compute_metrics(predictions, labels)

    # compute image level metrics (did we find waldo in each test image?)
    image_metrics = compute_image_level_metrics(predictions, labels, confidences, test_loader.dataset)

    # analyze different confidence thresholds to find the best one
    print("\nanalyzing confidence thresholds...")
    best_threshold, best_f1 = analyze_thresholds(confidences, labels)

    # save all results to json for the report
    print("\nsaving results to json...")
    save_results(metrics, confidences, labels, best_threshold, best_f1, image_metrics, inference_time, num_tiles)

    # generate all plots
    print("\nplotting confidence distribution...")
    plot_confidence_distribution(confidences, labels)

    print("\nplotting precision recall curve...")
    plot_precision_recall_curve(confidences, labels)

    print("\nplotting confusion matrix...")
    plot_confusion_matrix(predictions, labels)

    print("\nplotting f1 vs threshold curve...")
    plot_f1_vs_threshold(confidences, labels)

    print("\nplotting ROC curve...")
    plot_roc_curve(confidences, labels)

    print("\nplotting per image summary...")
    plot_per_image_summary(predictions, labels, test_loader.dataset)

    print("\nplotting training curves...")
    plot_training_curves()

    print("\nvisualizing sample tiles...")
    visualize_samples(model, test_loader, device)

    print("\nrunning full image demo...")
    raw_images = sorted(os.listdir(RAW_IMAGES))
    demo_count = min(3, len(raw_images))
    for i in range(demo_count):
        img_path = os.path.join(RAW_IMAGES, raw_images[i])
        run_demo(model, device, image_path=img_path)

    print(f"\nevaluation complete! check {RESULTS_DIR}/ for all outputs")

    # restore the original RESULTS_DIR
    RESULTS_DIR = original_results_dir

    return metrics


# when run directly (python evaluate.py), use the default paths
if __name__ == "__main__":
    run_evaluation()
