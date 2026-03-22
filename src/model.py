

# model.py
# this is where we wire everything together into one full model
# the idea is: image tile goes in, waldo predictions come out
#
# the flow goes like this:
#   1. tile image [N, 3, 256, 256] comes from data_pipeline
#   2. pre_processing runs HSV mask + sobel edge detection on it
#   3. MobileV2Backbone extracts features [N, 64, 16, 16]
#   4. 6 SSD heads run on the features at different scales, each one predicts bounding boxes + class scores
#   5. ConcatenateAnchors collects all predictions from all 6 heads
#   6. NMS (non-maximum suppression) filters out overlapping boxes, keeps the best ones
#
# right now we have the individual pieces built:
#   data_pipeline.py (done): loads tiles and labels
#   pre_processing.py (done): HSV mask and sobel edges
#   MobileV2Backbone.py (done): feature extraction
#   ssd.py (done): single SSD block
#   ConcatenateAnchors.py (needs work): just a stub right now
#
# TODO: wire backbone -> ssd heads -> anchor concat -> nms into one nn.Module
# TODO: figure out how pre_processing fits in (extra input channels? or separate scoring?)
# TODO: handle the channel mismatch (backbone outputs 64 channels, need to decide SSD input channels)

import pre_processing
import MobileV2Backbone
import ConcatenateAnchors
import config

import torch
import torch.nn as nn

from ssd import ssd_block
from ConcatenateAnchors import concat_ssd_heads
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection._utils import BoxCoder
import torchvision.ops as ops

class WaldoSSD(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        num_heads: int = 4,
        num_anchors: int = 4,
        in_channels: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.backbone = backbone

        # Configurate the anchors
        aspect_ratios = [[1.0, 2.0, 0.5]] * num_heads

        self.anchor_generator = DefaultBoxGenerator(
            aspect_ratios=aspect_ratios,
            min_ratio=0.07,
            max_ratio=0.9
        )

        # SSD heads
        self.blocks = nn.ModuleList([
            ssd_block(
                in_channels=in_channels,
                num_classes=num_classes,
                kernel_size=kernel_size,
                num_anchors=num_anchors
            )
            for _ in range(num_heads)
        ])

    def forward(self, x):
        locs = []
        clss = []
        feature_maps = []

        # The original input to the model
        images = x

        # Backbone
        x = self.backbone(x)

        # SSD heads
        for block in self.blocks:
            loc, cls, next_x = block(x)

            locs.append(loc)
            clss.append(cls)
            feature_maps.append(x)

            x = next_x

        # Concatenate predictions
        locs, clss = concat_ssd_heads(locs, clss)

        # Generate anchors
        anchors = self.anchor_generator(images, feature_maps)
        
        #return locs, clss, anchors
        
        pred_boxes = box_coder.decode(locs, anchors)
        return pred_boxes

if __name__ == "__main__":

    backbone = CreateWalsoModel()
    pred_boxed = WaldoSSD(backbone)
    print(pred_boxes)
