'''
visualize_data.py
shows sample tiles with the preprocessing channels so you can see
what the model actually receives as input

generates a grid showing:
  - original RGB tile
  - HSV color mask (red/white pixel detection)
  - sobel edges of the HSV mask (stripe outlines)
  - all 5 channels combined

saves output to results/preprocessing_visualization.png

usage: python src/scripts/visualize_data.py
'''

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.data_pipeline import get_dataloader
from data.pre_processing import apply_hsv_mask, preprocess_batch
from config import IMAGENET_MEAN, IMAGENET_STD


def denormalize(images):
    '''undo ImageNet normalization to get displayable RGB images'''
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images.cpu() * std + mean).clamp(0, 1)


def visualize_preprocessing(num_positive=3, num_negative=3):
    '''
    loads sample tiles and shows what each preprocessing stage looks like
    picks both positive (waldo) and negative (background) tiles
    '''

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)

    # load test tiles
    loader = get_dataloader(split="test", batch_size=32, shuffle=False)

    positives = []
    negatives = []

    for images, labels, _ in loader:
        for i in range(len(labels)):
            if labels[i] == 1 and len(positives) < num_positive:
                positives.append(images[i])
            elif labels[i] == 0 and len(negatives) < num_negative:
                negatives.append(images[i])
            if len(positives) >= num_positive and len(negatives) >= num_negative:
                break
        if len(positives) >= num_positive and len(negatives) >= num_negative:
            break

    samples = positives + negatives
    sample_labels = ["WALDO"] * len(positives) + ["background"] * len(negatives)

    if len(samples) == 0:
        print("no tiles found. run data pipeline first: python src/data/data_pipeline.py")
        return

    batch = torch.stack(samples)

    # compute preprocessing channels
    hsv_mask = apply_hsv_mask(batch)
    full_preprocessed = preprocess_batch(batch)
    sobel_channel = full_preprocessed[:, 4:5, :, :]

    rgb = denormalize(batch)

    # create the visualization grid
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    column_titles = ["Original RGB", "HSV Mask (red/white)", "Sobel of HSV (stripe edges)", "All 5 Channels (model input)"]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for i in range(n):
        # original RGB
        axes[i, 0].imshow(rgb[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_ylabel(sample_labels[i], fontsize=11, fontweight="bold",
                               color="green" if sample_labels[i] == "WALDO" else "gray")

        # HSV mask
        axes[i, 1].imshow(hsv_mask[i, 0].cpu().numpy(), cmap="Reds")

        # sobel edges
        axes[i, 2].imshow(sobel_channel[i, 0].cpu().numpy(), cmap="hot")

        # combined: show RGB with HSV mask overlaid in red
        combined = rgb[i].permute(1, 2, 0).numpy().copy()
        mask = hsv_mask[i, 0].cpu().numpy()
        overlay = combined.copy()
        overlay[:, :, 0] = 1.0  # red channel
        overlay[:, :, 1] = 0.0
        overlay[:, :, 2] = 0.0
        alpha = 0.3
        blended = combined * (1 - alpha * mask[:, :, None]) + overlay * (alpha * mask[:, :, None])
        axes[i, 3].imshow(blended.clip(0, 1))

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Preprocessing Pipeline Visualization\n"
                 "HSV mask highlights Waldo's red/white stripes, Sobel detects stripe edges",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = os.path.join(results_dir, "preprocessing_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved visualization to {output_path}")


if __name__ == "__main__":
    visualize_preprocessing()
