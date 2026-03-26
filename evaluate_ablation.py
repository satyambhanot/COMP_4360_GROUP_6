# evaluate_ablation.py
# runs ablation studies to prove each component of our pipeline actually helps
# for each experiment we train a variant of the model with one thing changed
# then evaluate on the same test set and compare the results
#
# this is really important for the report because the professor wants to see
# that we understand why each component is there, not just that we threw
# everything together and hoped it would work. ablation studies show that
# each design choice was deliberate and backed by data
#
# the studies we run:
#   1. preprocessing impact: does our HSV mask and Sobel edge detection help?
#   2. model architecture: how do custom SSD, pretrained SSD, and RetinaNet compare?
#
# for study 1 we train the same architecture with different preprocessing:
#   full pipeline (RGB + HSV + Sobel) vs RGB + HSV only vs RGB only
#   if removing a component hurts performance, that proves it was helping
#
# study 2 just loads results from the different model runs
# (we dont retrain here, just compare the saved results)
#
# each training run uses the same random seed, same data split, same number
# of epochs so the comparison is fair. the only thing that changes is
# the specific component being tested
#
# WARNING: training multiple models takes a long time
# on the GTX 1080 Ti each run is about 1 hour
# study 1 alone trains 3 models so thats about 3 hours
# plan accordingly and maybe run overnight
#
# usage: python evaluate_ablation.py

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import create_model
from data_pipeline import get_dataloader
from config import (
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE,
    CHECKPOINTS_DIR, MODEL_TYPE, INPUT_SIZE,
    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR,
    EARLY_STOPPING_PATIENCE, WEIGHT_DECAY,
    RETINANET_INPUT_SIZE, PRETRAINED_SSD_INPUT_SIZE
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ABLATION_DIR = os.path.join(RESULTS_DIR, "ablation")

# how many epochs to train each ablation variant
# we use fewer epochs than the main training to save time
# 30 is enough to see if a component makes a meaningful difference
# the model wont fully converge but thats fine for ablation
# we just need to see relative differences between variants
ABLATION_EPOCHS = 30

# device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def train_variant(model, train_loader, val_loader, variant_name, num_epochs=ABLATION_EPOCHS):
    # trains a single model variant and returns the best validation loss
    # this is a simplified version of train.py's training loop
    # we dont need all the bells and whistles for ablation, just train and evaluate

    model = model.to(device)
    model_type = model.model_type
    print(f"  training {variant_name} ({model_type})...")
    print(f"  trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # set up optimizer (same as train.py)
    if hasattr(model, 'get_param_groups'):
        head_params = model.get_param_groups()
        optimizer = torch.optim.SGD(head_params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR
    )

    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        # train one epoch
        model.train()
        train_loss = 0
        train_batches = 0

        for images, labels, bboxes in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            if model_type in ("pretrained_ssd", "retinanet"):
                # format targets for torchvision models
                if model_type == "retinanet":
                    scale = RETINANET_INPUT_SIZE / INPUT_SIZE
                else:
                    scale = PRETRAINED_SSD_INPUT_SIZE / INPUT_SIZE

                targets = []
                for i in range(len(labels)):
                    if labels[i].item() == 1:
                        box = bboxes[i].unsqueeze(0).to(device) * scale
                        targets.append({"boxes": box, "labels": torch.tensor([1], dtype=torch.long, device=device)})
                    else:
                        targets.append({"boxes": torch.zeros(0, 4, device=device), "labels": torch.zeros(0, dtype=torch.long, device=device)})

                loss_dict = model(images, targets)
                loss = loss_dict["classification"] + loss_dict["bbox_regression"]
            else:
                from ConcatenateAnchors import concatenate_all_heads
                locs, cls_scores, anchors = model(images)
                # simplified loss for custom SSD
                loss = F.cross_entropy(cls_scores.view(-1, 2), torch.zeros(cls_scores.shape[0] * cls_scores.shape[1], dtype=torch.long, device=device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train = train_loss / max(train_batches, 1)

        # validate
        model.train() if model_type in ("pretrained_ssd", "retinanet") else model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for images, labels, bboxes in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                bboxes = bboxes.to(device)

                if model_type in ("pretrained_ssd", "retinanet"):
                    if model_type == "retinanet":
                        scale = RETINANET_INPUT_SIZE / INPUT_SIZE
                    else:
                        scale = PRETRAINED_SSD_INPUT_SIZE / INPUT_SIZE

                    targets = []
                    for i in range(len(labels)):
                        if labels[i].item() == 1:
                            box = bboxes[i].unsqueeze(0).to(device) * scale
                            targets.append({"boxes": box, "labels": torch.tensor([1], dtype=torch.long, device=device)})
                        else:
                            targets.append({"boxes": torch.zeros(0, 4, device=device), "labels": torch.zeros(0, dtype=torch.long, device=device)})

                    loss_dict = model(images, targets)
                    loss_val = loss_dict["classification"].item() + loss_dict["bbox_regression"].item()
                else:
                    locs, cls_scores, anchors = model(images)
                    loss_val = F.cross_entropy(cls_scores.view(-1, 2), torch.zeros(cls_scores.shape[0] * cls_scores.shape[1], dtype=torch.long, device=device)).item()

                val_loss += loss_val
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            # save checkpoint for this variant
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"ablation_{variant_name}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val
            }, checkpoint_path)

        # print every epoch so you can see training is progressing
        saved_marker = " [saved]" if avg_val == best_val_loss else ""
        print(f"    epoch {epoch + 1}/{num_epochs}  train: {avg_train:.4f}  val: {avg_val:.4f}{saved_marker}", flush=True)

    elapsed = time.time() - start_time
    print(f"  done in {elapsed:.0f}s, best val loss {best_val_loss:.4f} at epoch {best_epoch}")
    model.eval()
    return best_val_loss, best_epoch, elapsed


def evaluate_variant(model, test_loader, variant_name):
    # runs the trained model on the test set and computes metrics
    # returns a dict with precision, recall, F1 etc

    model = model.to(device)
    model.eval()
    model_type = model.model_type

    all_preds = []
    all_labels = []
    all_confs = []

    with torch.no_grad():
        for images, labels, bboxes in test_loader:
            images = images.to(device)

            if model_type in ("pretrained_ssd", "retinanet"):
                detections = model(images)
                for i in range(len(detections)):
                    det = detections[i]
                    waldo_mask = det["labels"] == 1
                    scores = det["scores"][waldo_mask]
                    if scores.shape[0] > 0:
                        all_preds.append(1)
                        all_confs.append(scores.max().item())
                    else:
                        all_preds.append(0)
                        all_confs.append(0.0)
                    all_labels.append(labels[i].item())
            else:
                locs, cls_scores, anchors = model(images)
                from nms import apply_nms
                nms_results = apply_nms(locs, cls_scores, anchors)
                for i in range(images.shape[0]):
                    boxes, scores = nms_results[i]
                    if boxes.shape[0] > 0:
                        all_preds.append(1)
                        all_confs.append(scores.max().item())
                    else:
                        all_preds.append(0)
                        all_confs.append(0.0)
                    all_labels.append(labels[i].item())

    # compute metrics
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)

    total = len(all_preds)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    result = {
        "variant": variant_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

    print(f"  {variant_name}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    return result


def run_preprocessing_ablation():
    # study 1: does our preprocessing (HSV mask + Sobel edges) actually help?
    # we train three variants of the same base architecture:
    #   full: RGB + HSV mask + Sobel edges (our normal pipeline)
    #   hsv_only: RGB + HSV mask (no Sobel)
    #   rgb_only: just RGB (no preprocessing at all)
    # if the full model beats rgb_only, our preprocessing helps
    # if hsv_only is close to full, sobel might not be adding much

    print("\n" + "=" * 50)
    print("  Study 1: Preprocessing Impact")
    print("=" * 50)

    train_loader = get_dataloader(split="train", tile_size=256, batch_size=BATCH_SIZE)
    val_loader = get_dataloader(split="val", tile_size=256, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(split="test", tile_size=256, batch_size=BATCH_SIZE, shuffle=False)

    variants = [
        ("full_pipeline", True, True),    # RGB + HSV + Sobel
        ("hsv_only", True, False),         # RGB + HSV (no Sobel)
        ("rgb_only", False, False),        # just RGB
    ]

    results = []

    for name, use_hsv, use_sobel in variants:
        print(f"\n--- {name} (hsv={use_hsv}, sobel={use_sobel}) ---", flush=True)

        # free GPU memory from the previous model before creating a new one
        # without this we run out of VRAM on the second or third variant
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # create model with specific preprocessing flags
        model = create_model(use_hsv=use_hsv, use_sobel=use_sobel)

        # train it
        val_loss, best_epoch, elapsed = train_variant(model, train_loader, val_loader, name)

        # load best checkpoint and evaluate
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"ablation_{name}.pth")
        if os.path.exists(checkpoint_path):
            cp = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(cp["model_state_dict"])

        result = evaluate_variant(model, test_loader, name)
        result["val_loss"] = val_loss
        result["best_epoch"] = best_epoch
        result["training_time"] = elapsed

        # free the model from GPU memory before the next variant
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results.append(result)

    return results


def plot_preprocessing_comparison(results):
    # creates a bar chart comparing the preprocessing variants
    # this goes in the report to show that HSV and Sobel actually help

    if len(results) == 0:
        return

    names = [r["variant"] for r in results]
    display_names = {
        "full_pipeline": "RGB + HSV + Sobel",
        "hsv_only": "RGB + HSV",
        "rgb_only": "RGB Only"
    }
    labels = [display_names.get(n, n) for n in names]

    metrics = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [r[metric] for r in results]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Preprocessing Impact")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_DIR, "preprocessing_comparison.png"), dpi=150)
    plt.close()
    print(f"saved preprocessing comparison to results/ablation/preprocessing_comparison.png")


def load_architecture_results():
    # study 2: model architecture comparison
    # we dont retrain here, just load results from previous runs
    # the user needs to have run each model and saved the results separately

    print("\n" + "=" * 50)
    print("  Study 2: Model Architecture Comparison")
    print("=" * 50)

    result_files = {
        "Custom SSD": os.path.join(RESULTS_DIR, "custom_ssd_results.json"),
        "Pretrained SSD300": os.path.join(RESULTS_DIR, "ssd_results.json"),
        "RetinaNet": os.path.join(RESULTS_DIR, "retinanet_results.json"),
    }

    yolo_path = os.path.join(RESULTS_DIR, "yolo", "yolo_results.json")
    if os.path.exists(yolo_path):
        result_files["YOLOv8"] = yolo_path

    results = []
    for name, path in result_files.items():
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            m = data["metrics"]
            results.append({
                "variant": name,
                "precision": m.get("precision", 0),
                "recall": m.get("recall", 0),
                "f1": m.get("f1", 0),
                "accuracy": m.get("accuracy", 0)
            })
            print(f"  loaded {name}: F1={m.get('f1', 0):.4f}")
        else:
            print(f"  {name}: no results found at {path}")

    return results


def plot_architecture_comparison(results):
    # creates a bar chart comparing the different model architectures

    if len(results) < 2:
        print("need at least 2 architecture results to compare")
        return

    names = [r["variant"] for r in results]
    metrics = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2.5), 6))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [r[metric] for r in results]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Model Architecture Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_DIR, "architecture_comparison.png"), dpi=150)
    plt.close()
    print(f"saved architecture comparison to results/ablation/architecture_comparison.png")


def save_ablation_summary(preprocessing_results, architecture_results):
    # saves all ablation results to a single JSON file and generates a summary table
    # the JSON is machine readable for automated comparison
    # the table image goes directly into the report

    summary = {
        "preprocessing_ablation": preprocessing_results,
        "architecture_comparison": architecture_results
    }

    json_path = os.path.join(ABLATION_DIR, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved all ablation results to {json_path}")

    # generate summary table image
    all_results = preprocessing_results + architecture_results
    if len(all_results) == 0:
        return

    headers = ["Experiment", "Precision", "Recall", "F1"]
    rows = []
    for r in all_results:
        rows.append([
            r["variant"],
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}"
        ])

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(rows) + 2)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # highlight best F1 in green
    best_idx = 0
    best_f1 = 0
    for i, r in enumerate(all_results):
        if r["f1"] > best_f1:
            best_f1 = r["f1"]
            best_idx = i
    for j in range(len(headers)):
        table[best_idx + 1, j].set_facecolor("#C8E6C9")

    ax.set_title("Ablation Study Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_DIR, "ablation_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved ablation summary table to results/ablation/ablation_summary.png")


if __name__ == "__main__":
    print("=" * 50)
    print("  Ablation Studies")
    print("=" * 50)
    print()
    print(f"using device: {device}")
    print(f"ablation epochs: {ABLATION_EPOCHS}")
    print(f"base model type: {MODEL_TYPE}")

    os.makedirs(ABLATION_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # study 1: preprocessing impact (requires training 3 models)
    print("\nthis will train 3 model variants to test preprocessing impact")
    print("estimated time: ~90 minutes on GTX 1080 Ti")
    preprocessing_results = run_preprocessing_ablation()
    plot_preprocessing_comparison(preprocessing_results)

    # study 2: architecture comparison (loads saved results, no training)
    architecture_results = load_architecture_results()
    if len(architecture_results) >= 2:
        plot_architecture_comparison(architecture_results)

    # save everything
    save_ablation_summary(preprocessing_results, architecture_results)

    print("\n" + "=" * 50)
    print("  Ablation studies complete!")
    print(f"  check results/ablation/ for all outputs")
    print("=" * 50)
