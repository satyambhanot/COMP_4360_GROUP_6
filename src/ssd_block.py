'''
ssd_block.py
a single SSD detection block that takes a feature map and makes predictions
each block predicts bounding box offsets and class scores at one scale
then downsamples the feature map for the next block to work on

we chain 5 of these together in model.py to detect waldo at different sizes
the feature map starts at 16x16 from the backbone and each block halves it
16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1

this follows the SSD architecture from Liu et al (ECCV 2016)
'''

import torch.nn as nn


class SSDBlock(nn.Module):
    '''
    one detection head in our custom SSD pipeline
    takes a feature map, predicts bounding boxes and class scores,
    then pools the feature map to half size for the next head

    args (constructor):
        in_channels: number of channels in the input feature map
        num_classes: number of classes to predict (2 for waldo vs background)
        kernel_size: size of the convolution kernel (usually 3)
        num_anchors: number of anchor boxes per spatial position (usually 4)
    '''

    def __init__(self, in_channels, num_classes, kernel_size, num_anchors):
        super().__init__()

        # predicts bounding box offsets (x, y, w, h) for each anchor at each position
        # num_anchors * 4 because each anchor needs 4 values to describe a box
        self.loc = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            kernel_size,
            padding=1
        )

        # predicts class probabilities (waldo or background) for each anchor
        self.cls = nn.Conv2d(
            in_channels,
            num_anchors * num_classes,
            kernel_size,
            padding=1
        )

        # downsample the feature map so the next block can detect at a coarser scale
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        '''
        runs one detection head on the input feature map

        args:
            x: feature map tensor [N, C, H, W]

        returns:
            loc: bounding box predictions [N, num_anchors*4, H, W]
            cls: class score predictions [N, num_anchors*num_classes, H, W]
            x: pooled feature map for the next head [N, C, H/2, W/2]
        '''
        loc = self.loc(x)
        cls = self.cls(x)

        # only pool if the feature map is bigger than 1x1
        # the last head gets a 1x1 input and pooling would give 0x0 which crashes
        if x.shape[2] > 1 and x.shape[3] > 1:
            x = self.pool(x)

        return loc, cls, x
