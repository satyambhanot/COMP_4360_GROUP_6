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


# TODO: add run_predictions
# TODO: add compute_metrics
# TODO: add plot functions
# TODO: add visualize_samples
# TODO: add run_demo
# TODO: add main block
