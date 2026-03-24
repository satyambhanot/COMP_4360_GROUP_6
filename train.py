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
# with batch size 16 and 256x256 tiles this should fit comfortably

import sys
import os
import time

# add the src folder to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import WaldoDetector
from data_pipeline import get_dataloader
from ConcatenateAnchors import generate_anchors
from config import (
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE,
    CHECKPOINTS_DIR, NUM_CLASSES, INPUT_SIZE
)

# check if we have a gpu available
# the GTX 1080 Ti has 11GB VRAM which is plenty for our model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# TODO: add loss functions
# TODO: add training loop
# TODO: add validation
# TODO: add checkpoint saving
