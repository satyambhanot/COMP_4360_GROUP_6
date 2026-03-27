# experiments/run_ablation.py
# runs the preprocessing ablation study: tests whether HSV mask and Sobel
# edge detection actually improve the model by training three variants
# saves everything to saved_results/ablation/
#
# variants:
#   full_pipeline: RGB + HSV mask + Sobel edges (our normal approach)
#   hsv_only: RGB + HSV mask (no Sobel)
#   rgb_only: just RGB (no preprocessing at all)
#
# each variant trains for 30 epochs on the same data with the same seed
# then evaluates on the same test set for a fair comparison
#
# usage: python experiments/run_ablation.py

import sys
import os
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# use RetinaNet as the base model for ablation since its our best architecture
import config
config.MODEL_TYPE = "retinanet"
config.BATCH_SIZE = 8
config.NUM_EPOCHS = 30  # enough to see relative differences
config.EARLY_STOPPING_PATIENCE = 15

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import create_model
from train import run_training
from evaluate import run_evaluation

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "saved_results", "ablation")

# device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def run_variant(name, use_hsv, use_sobel):
    # trains and evaluates one preprocessing variant
    # returns the metrics dict

    variant_dir = os.path.join(OUTPUT_DIR, name)
    checkpoint_dir = os.path.join(variant_dir, "checkpoints")
    plots_dir = os.path.join(variant_dir, "plots")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n{'=' * 40}")
    print(f"  Variant: {name}")
    print(f"  HSV: {use_hsv}, Sobel: {use_sobel}")
    print(f"  Output: {variant_dir}")
    print(f"{'=' * 40}", flush=True)

    # create model with specific preprocessing flags
    model = create_model(use_hsv=use_hsv, use_sobel=use_sobel)

    # train it
    print("\n--- Training ---", flush=True)
    model, val_loss, history = run_training(
        checkpoint_dir=checkpoint_dir,
        model_override=model
    )

    # free GPU before evaluation
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # evaluate
    print("\n--- Evaluation ---", flush=True)
    metrics = run_evaluation(output_dir=plots_dir, checkpoint_dir=checkpoint_dir)

    # save metrics to variant dir
    if metrics:
        result = {
            "variant": name,
            "use_hsv": use_hsv,
            "use_sobel": use_sobel,
            "metrics": metrics,
            "val_loss": val_loss
        }
        with open(os.path.join(variant_dir, "metrics.json"), "w") as f:
            json.dump(result, f, indent=2)

    # free GPU for next variant
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def generate_comparison(all_results):
    # creates comparison charts across the preprocessing variants

    comparison_dir = os.path.join(OUTPUT_DIR, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    names = [r["name"] for r in all_results]
    display_names = {
        "full_pipeline": "RGB + HSV + Sobel",
        "hsv_only": "RGB + HSV",
        "rgb_only": "RGB Only"
    }
    labels = [display_names.get(n, n) for n in names]

    metric_keys = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        values = [r["metrics"].get(key, 0) for r in all_results]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Preprocessing Impact on RetinaNet")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "preprocessing_comparison.png"), dpi=150)
    plt.close()
    print(f"saved comparison chart to {comparison_dir}/preprocessing_comparison.png")

    # save combined results
    with open(os.path.join(comparison_dir, "ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"saved combined results to {comparison_dir}/ablation_results.json")


if __name__ == "__main__":
    print("=" * 50)
    print("  Experiment: Preprocessing Ablation Study")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Device: {device}")
    print(f"  Epochs per variant: {config.NUM_EPOCHS}")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    variants = [
        ("full_pipeline", True, True),
        ("hsv_only", True, False),
        ("rgb_only", False, False),
    ]

    all_results = []
    total_start = time.time()

    for name, use_hsv, use_sobel in variants:
        metrics = run_variant(name, use_hsv, use_sobel)
        if metrics:
            all_results.append({
                "name": name,
                "use_hsv": use_hsv,
                "use_sobel": use_sobel,
                "metrics": metrics
            })

    # generate comparison charts
    if len(all_results) >= 2:
        print("\n--- Generating Comparisons ---")
        generate_comparison(all_results)

    total_time = (time.time() - total_start) / 60
    print(f"\n{'=' * 50}")
    print(f"  Ablation study complete! ({total_time:.0f} minutes)")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 50}")
