# MobileV2Backbone.py
# this is our feature extractor - it takes raw tile images and pulls out
# useful features that the SSD heads can use to find waldo
# we use a pretrained MobileNetV2 because its already learned to recognize
# shapes, edges, textures etc from being trained on imagenet
# we only take the first few layers (set by BACKBONE_LAYERS in config)
# because we dont need the full classification network, just the features

import torch
from torchvision import models
from config import BATCH_SIZE, BACKBONE_LAYERS
from data_pipeline import get_dataloader


class MobileV2Backbone:

    def PassBatchThroughMobileV2Backbone(self, batch):

        mobileNet = models.mobilenet_v2(weights='DEFAULT') # pretrained on imagenet

        # grab only the first BACKBONE_LAYERS layers for feature extraction
        # with 11 layers this gives us 64 channels at 16x16 spatial resolution
        featureExtractor = mobileNet.features[:BACKBONE_LAYERS]
        featureExtractor.eval() # we dont want to update these weights, just use them as-is

        with(torch.no_grad()): # no need to track gradients here since were not training the backbone
            output = featureExtractor(batch)

        # output shape is [Batch size N, Channels C, Height H, Width W]
        # so for a batch of 10 images at 256x256, we get [10, 64, 16, 16]
        return output


def CreateWaldoModel():
    # quick test to make sure the backbone works with our data pipeline
    loader = get_dataloader(split="train", tile_size=256, batch_size=BATCH_SIZE)
    images, labels, bboxes = next(iter(loader))

    backbone = MobileV2Backbone()
    output = backbone.PassBatchThroughMobileV2Backbone(images)
    print(f"input shape: {images.shape}") # should be [BATCH_SIZE, 3, 256, 256]
    print(f"output shape: {output.shape}") # should be [BATCH_SIZE, 64, 16, 16]


if __name__ == "__main__":
    CreateWaldoModel()
