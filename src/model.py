# model.py
# this is the main model file where we connect all the pieces together
# a tile image goes in and waldo predictions come out
#
# i was confused about how many SSD heads to use because config originally said 6
# but when i traced through the sizes it only works with 5
# the backbone gives us 16x16 and each head cuts it in half
# 16 -> 8 -> 4 -> 2 -> 1, thats 5 heads
# after that its 1x1 and theres nothing left to halve
#
# the full flow through the model:
#
#   1. images come in as [N, 3, 256, 256] from the dataloader
#
#   2. preprocess_batch adds the HSV mask and sobel edges of the HSV mask
#      now we have [N, 5, 256, 256] (3 RGB + 1 HSV + 1 Sobel of HSV)
#      but this is configurable, we can turn off HSV or sobel for ablation studies
#
#   3. mobilenet expects 3 channels not 5
#      so we use a 1x1 conv to squish the extra channels back down to 3
#      this is a learnable layer so the model figures out
#      the best way to combine those channels
#      if preprocessing is off we skip this step entirely
#
#   4. run it through the mobilenet backbone (first 11 layers)
#      this extracts features and shrinks the spatial size
#      now we have [N, 64, 16, 16]
#
#   5. pass the features through 5 SSD heads one after another
#      each head predicts bounding box offsets and class scores
#      then pools the feature map to half the size for the next head
#
#      head 1: 16x16 -> 1024 anchors -> pools to 8x8
#      head 2: 8x8   -> 256 anchors  -> pools to 4x4
#      head 3: 4x4   -> 64 anchors   -> pools to 2x2
#      head 4: 2x2   -> 16 anchors   -> pools to 1x1
#      head 5: 1x1   -> 4 anchors    -> pools to 0x0 (we ignore this)
#
#   6. concatenate_all_heads combines everything into final tensors
#      total anchors: 1024 + 256 + 64 + 16 + 4 = 1364
#      loc output: [N, 1364, 4]
#      cls output: [N, 1364, 2]
#
# whats trainable and whats frozen:
#   frozen (we dont update these weights):
#     the mobilenet backbone, it already knows how to extract features from imagenet
#   trainable (these learn during training):
#     the 1x1 conv that merges the extra channels down to 3
#     all 5 SSD heads (the loc and cls conv layers inside each one)
#   not a layer at all (just a function with no weights):
#     preprocess_batch, it just does math on the pixels
#     concatenate_all_heads, it just reshapes tensors
#
# for ablation studies we added flags to turn parts on and off:
#   use_hsv: whether to include the HSV red/white mask channel
#   use_sobel: whether to include the sobel edge channel
#   use_pretrained: whether to load imagenet weights or start from random
#   num_heads: how many SSD heads to use (default 5, can try 3 for model size ablation)

import torch
import torch.nn as nn
from torchvision import models

from pre_processing import preprocess_batch, ApplyHSVMask, ApplySobelEdgeDetection
from ssd import ssd_block
from ConcatenateAnchors import concatenate_all_heads
from config import (
    BACKBONE_LAYERS, BACKBONE_CHANNELS,
    NUM_CLASSES, NUM_ANCHORS, SSD_KERNEL_SIZE,
    NUM_SSD_HEADS, BATCH_SIZE
)
from data_pipeline import get_dataloader
import torch.nn.functional as F


class WaldoDetector(nn.Module):

    def __init__(self, use_hsv=True, use_sobel=True, use_pretrained=True, num_heads=NUM_SSD_HEADS):
        super().__init__()

        # save these flags so forward knows what preprocessing to do
        self.use_hsv = use_hsv
        self.use_sobel = use_sobel

        # figure out how many input channels we have after preprocessing
        # starts at 3 for RGB, then we add 1 for HSV mask and 1 for sobel if enabled
        self.input_channels = 3
        if use_hsv:
            self.input_channels += 1 # HSV mask adds 1 channel
        if use_sobel:
            self.input_channels += 1 # sobel edges adds 1 channel

        # if we have more than 3 channels we need a 1x1 conv to squish them back to 3
        # because mobilenet was trained on 3 channel images and expects 3 channels
        # this is a learnable layer so the model figures out the best way to combine them
        # if preprocessing is completely off (just raw RGB) we skip this
        if self.input_channels > 3:
            self.channel_conv = nn.Conv2d(self.input_channels, 3, kernel_size=1)
        else:
            self.channel_conv = None # no need for this layer if we already have 3 channels

        # load mobilenet and grab only the first BACKBONE_LAYERS layers
        # we load it once here and reuse it for every batch instead of loading it fresh each time
        # use_pretrained controls whether we start with imagenet weights or random weights
        # this is useful for our ablation study where we test if transfer learning helps
        if use_pretrained:
            mobilenet = models.mobilenet_v2(weights='DEFAULT')
        else:
            mobilenet = models.mobilenet_v2(weights=None)

        self.backbone = mobilenet.features[:BACKBONE_LAYERS]

        # freeze the backbone weights so training doesnt mess up what mobilenet already learned
        # the backbone just extracts features, we dont want to change how it does that
        # gradients still flow through the backbone to reach the 1x1 conv
        # but the backbones own weights dont get updated
        for param in self.backbone.parameters():
            param.requires_grad = False

        # create the SSD detection heads using nn.ModuleList
        # we use ModuleList instead of a regular python list so pytorch knows about them
        # this way their weights get saved in checkpoints and moved to GPU properly
        # each head takes 64 channels from the backbone and predicts loc and cls
        self.ssd_heads = nn.ModuleList([
            ssd_block(
                in_channels=BACKBONE_CHANNELS,
                num_classes=NUM_CLASSES,
                kernel_size=SSD_KERNEL_SIZE,
                num_anchors=NUM_ANCHORS
            )
            for _ in range(num_heads)
        ])

        # the feature sizes that each head will see
        # backbone outputs 16x16, each head halves it: 16, 8, 4, 2, 1
        # we need this list for concatenate_all_heads so it knows the grid sizes
        # if someone uses fewer heads for ablation we only take the first N sizes
        all_sizes = [16, 8, 4, 2, 1]
        self.feature_sizes = all_sizes[:num_heads]

    # TODO: add forward method
    # TODO: add test at the bottom
