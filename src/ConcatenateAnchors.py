# ConcatenateAnchors.py
# this file handles anchor boxes for our SSD detector
# each SSD head makes predictions at different grid sizes (16x16, 8x8, 4x4, etc)
# at every grid position we place anchor boxes of different sizes
# the SSD heads predict offsets from these anchors, not absolute positions
# so we need to know where the anchors actually are on the image
#
# this file does three things:
#   1. generate anchor box positions for a given grid size
#   2. reshape the raw SSD output into a usable format
#   3. combine predictions from all heads into one big tensor

import torch
from config import NUM_ANCHORS, NUM_CLASSES, INPUT_SIZE


def generate_anchors(feature_size, image_size=INPUT_SIZE):
    # creates anchor box positions for one SSD head
    # feature_size is the spatial size of the feature map (like 16 for a 16x16 grid)
    # image_size is the original tile size (256)
    #
    # for each grid cell we place NUM_ANCHORS boxes of different sizes
    # we return the center x, center y, width, height of each anchor
    # all values are in pixels relative to the original image

    anchors = []

    # how big is each grid cell in pixels
    step = image_size / feature_size

    # we want anchors of different sizes at each position
    # smaller scale for early heads (they see fine detail)
    # bigger scale for later heads (they see the big picture)
    # these are fractions of the step size
    scales = [0.5, 1.0, 1.5, 2.0]

    for row in range(feature_size):
        for col in range(feature_size):
            # center of this grid cell in pixels
            cx = (col + 0.5) * step
            cy = (row + 0.5) * step

            # place one anchor box for each scale
            for scale in scales:
                w = step * scale
                h = step * scale
                anchors.append([cx, cy, w, h])

    # convert to tensor, shape is [feature_size * feature_size * NUM_ANCHORS, 4]
    return torch.tensor(anchors, dtype=torch.float32)


if __name__ == "__main__":
    # test generate_anchors with the feature sizes we expect from our SSD heads
    # backbone outputs 16x16, then each head halves it: 8x8, 4x4, 2x2, 1x1
    feature_sizes = [16, 8, 4, 2, 1]

    for fs in feature_sizes:
        anchors = generate_anchors(fs)
        print(f"feature size {fs}x{fs}: {anchors.shape[0]} anchors, shape {anchors.shape}")
        # show the first anchor as a sanity check
        print(f"  first anchor: cx={anchors[0][0]:.1f}, cy={anchors[0][1]:.1f}, w={anchors[0][2]:.1f}, h={anchors[0][3]:.1f}")
