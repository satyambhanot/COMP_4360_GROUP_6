# evaluate.py
# this is where we find out if our model actually works
# run from project root: python evaluate.py
#
# what this file does:
#   loads the best trained model from checkpoints, runs it on test tiles
#   the model has never seen before, measures how well it finds waldo
#   using real metrics, and saves visualizations for the report

import sys
import os
import json

# add the src folder to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg") # use non interactive backend so it works without a display
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms

from model import WaldoDetector
from data_pipeline import get_dataloader
from nms import apply_nms
from config import (
    CHECKPOINTS_DIR, BATCH_SIZE, INPUT_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, RAW_IMAGES
)

# where we save all the output images and results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# check if we have a gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")


def load_model():
    # loads the trained model from the best checkpoint
    # if the checkpoint doesnt exist it means nobody has trained yet
    # so we print a helpful message instead of crashing

    checkpoint_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"no checkpoint found at {checkpoint_path}")
        print("you need to train the model first by running: python train.py")
        return None

    # create the model with the same architecture it was trained with
    model = WaldoDetector()
    model = model.to(device)

    # load the saved weights into the model
    # map_location makes sure it works even if the checkpoint was saved on GPU
    # but we are loading on CPU or vice versa
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # put the model in eval mode
    # this is important because it changes how some layers behave
    model.eval()

    print(f"loaded checkpoint from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")
    return model


def draw_boxes(image, boxes, labels=None, scores=None, color="red", width=2):
    # takes a PIL image and draws bounding boxes on it
    # used for visualizing what the model predicted
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
    # takes a full wheres waldo image and chops it into tiles
    # we keep track of each tiles position so we can map predictions
    # back to the original image later
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
    # the dataloader normalizes images with imagenet mean and std
    # to display them as normal images we need to undo that
    # otherwise the colors look completely wrong
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def run_predictions(model, dataloader, device):
    # runs every test tile through the model and collects the results
    # for each tile we record: did the model predict waldo? what was the confidence?
    # and what was the actual ground truth label?
    # we use these to compute metrics and draw visualizations

    all_predictions = [] # 1 if model thinks waldo is in this tile, 0 if not
    all_labels = [] # the actual ground truth label (0 or 1)
    all_confidences = [] # the highest waldo confidence score for each tile
    all_nms_results = [] # the filtered boxes and scores from NMS for visualization

    with torch.no_grad():
        for images, labels, bboxes in dataloader:
            images = images.to(device)

            # run the tiles through the model
            locs, cls_scores, anchors = model(images)

            # run NMS to filter overlapping predictions
            nms_results = apply_nms(locs, cls_scores, anchors)

            # for each tile in the batch, check if NMS found any waldo boxes
            for i in range(images.shape[0]):
                boxes, scores = nms_results[i]
                label = labels[i].item()

                # if NMS returned any boxes, the model thinks this tile has waldo
                if boxes.shape[0] > 0:
                    prediction = 1
                    confidence = scores.max().item()
                else:
                    prediction = 0
                    confidence = 0.0

                all_predictions.append(prediction)
                all_labels.append(label)
                all_confidences.append(confidence)
                all_nms_results.append((boxes, scores))

    return all_predictions, all_labels, all_confidences, all_nms_results


def compute_metrics(predictions, labels):
    # computes accuracy, precision, recall, and f1 from the predictions
    # also prints a confusion matrix showing TP, FP, TN, FN counts
    # returns a dictionary of all metrics so ablation studies can use them

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


def plot_training_curves():
    # loads the training history json and plots the loss curves
    # this shows whether the model was learning and when it started overfitting
    # we plot train and val loss on the same chart for easy comparison

    history_path = os.path.join(CHECKPOINTS_DIR, "training_history.json")

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
    # shows the distribution of confidence scores for positive vs negative tiles
    # if the model is working well, positive tiles should have high confidence
    # and negative tiles should have low confidence
    # if the two distributions overlap a lot the model is confused

    pos_conf = [c for c, l in zip(confidences, labels) if l == 1]
    neg_conf = [c for c, l in zip(confidences, labels) if l == 0]

    plt.figure(figsize=(8, 5))
    plt.hist(neg_conf, bins=50, alpha=0.6, label=f"no waldo ({len(neg_conf)} tiles)", color="blue")
    plt.hist(pos_conf, bins=50, alpha=0.6, label=f"has waldo ({len(pos_conf)} tiles)", color="red")
    plt.axvline(x=0.5, color="black", linestyle=":", label="confidence threshold (0.5)")
    plt.xlabel("waldo confidence score")
    plt.ylabel("number of tiles")
    plt.title("Confidence Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confidence_distribution.png"), dpi=150)
    plt.close()
    print(f"saved confidence distribution to results/confidence_distribution.png")


def visualize_samples(model, dataloader, device, num_samples=5):
    # grabs some test tiles and draws the model predictions on them
    # saves the images so we can put them in the report
    # we try to get a mix of true positives, false positives, and false negatives

    tp_images = [] # correctly found waldo
    fp_images = [] # falsely said waldo
    fn_images = [] # missed waldo

    with torch.no_grad():
        for images, labels, bboxes in dataloader:
            images = images.to(device)
            locs, cls_scores, anchors = model(images)
            nms_results = apply_nms(locs, cls_scores, anchors)

            for i in range(images.shape[0]):
                boxes, scores = nms_results[i]
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
    # takes a full wheres waldo image, tiles it up, runs every tile
    # through the model, and draws boxes where it thinks waldo is
    # this is the final "look it found waldo" image for the presentation

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
        tile_tensors = tile_tensors.to(device)

        with torch.no_grad():
            locs, cls_scores, anchors = model(tile_tensors)
            nms_results = apply_nms(locs, cls_scores, anchors)

        for i in range(len(batch_tiles)):
            boxes, scores = nms_results[i]
            if boxes.shape[0] > 0:
                w_off, h_off = batch_coords[i]
                # map tile coordinates back to the original image
                # the boxes from NMS are relative to the tile
                # we add the tiles position offset to get image level coordinates
                adjusted_boxes = boxes.cpu().clone()
                adjusted_boxes[:, 0] += w_off # x1
                adjusted_boxes[:, 1] += h_off # y1
                adjusted_boxes[:, 2] += w_off # x2
                adjusted_boxes[:, 3] += h_off # y2
                all_detections.append((adjusted_boxes, scores.cpu()))

    # draw all detections on the original full image
    for boxes, scores in all_detections:
        image = draw_boxes(image, boxes, scores=scores, color="green", width=3)

    demo_path = os.path.join(RESULTS_DIR, "demo_full_image.png")
    image.save(demo_path)
    print(f"found detections in {len(all_detections)} tiles")
    print(f"saved demo image to {demo_path}")


if __name__ == "__main__":
    # make sure the results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # load the trained model
    model = load_model()
    if model is None:
        exit()

    # load the test data
    print("\nloading test data...")
    test_loader = get_dataloader(split="test", tile_size=256, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  test set: {len(test_loader.dataset)} tiles")

    # run predictions on every test tile
    print("\nrunning predictions on test set...")
    predictions, labels, confidences, nms_results = run_predictions(model, test_loader, device)

    # compute and print metrics
    metrics = compute_metrics(predictions, labels)

    # plot the confidence distribution
    print("\nplotting confidence distribution...")
    plot_confidence_distribution(confidences, labels)

    # plot training curves if we have the history
    print("\nplotting training curves...")
    plot_training_curves()

    # save sample visualization images
    print("\nvisualizing sample tiles...")
    visualize_samples(model, test_loader, device)

    # run the full image demo
    print("\nrunning full image demo...")
    run_demo(model, device)

    print("\nevaluation complete! check the results/ folder for all outputs")
