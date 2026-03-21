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


def reshape_predictions(loc, cls):
    # the SSD head outputs loc and cls in convolution format
    # loc comes out as [N, NUM_ANCHORS * 4, H, W] which is hard to work with
    # cls comes out as [N, NUM_ANCHORS * NUM_CLASSES, H, W]
    #
    # we want to rearrange them so each row is one prediction:
    # loc becomes [N, H * W * NUM_ANCHORS, 4] (one row per anchor, 4 box values each)
    # cls becomes [N, H * W * NUM_ANCHORS, NUM_CLASSES] (one row per anchor, 2 class scores each)
    #
    # we're not changing any values here, just rearranging them into a flat list

    batch_size = loc.shape[0]

    # loc: [N, anchors*4, H, W] -> [N, H, W, anchors*4] -> [N, H*W*anchors, 4]
    loc = loc.permute(0, 2, 3, 1).contiguous() # move channels to the end
    loc = loc.view(batch_size, -1, 4) # flatten into rows of 4

    # cls: [N, anchors*classes, H, W] -> [N, H, W, anchors*classes] -> [N, H*W*anchors, classes]
    cls = cls.permute(0, 2, 3, 1).contiguous()
    cls = cls.view(batch_size, -1, NUM_CLASSES)

    return loc, cls


def concatenate_all_heads(all_loc_preds, all_cls_preds, feature_sizes):
    # takes the raw loc and cls outputs from every SSD head
    # reshapes each one into a flat list of predictions
    # then stacks them all together so we have one big tensor
    #
    # all_loc_preds is a list of loc tensors, one per head
    # all_cls_preds is a list of cls tensors, one per head
    # feature_sizes is a list like [16, 8, 4, 2, 1] telling us each heads grid size
    #
    # returns:
    #   all_locs: [N, total_anchors, 4] every box prediction from every head
    #   all_cls: [N, total_anchors, 2] every class score from every head
    #   all_anchors: [total_anchors, 4] the actual anchor positions on the image

    reshaped_locs = []
    reshaped_cls = []
    all_anchors = []

    for i in range(len(all_loc_preds)):
        # reshape this heads predictions into flat lists
        loc, cls = reshape_predictions(all_loc_preds[i], all_cls_preds[i])
        reshaped_locs.append(loc)
        reshaped_cls.append(cls)

        # generate the anchor positions for this heads grid size
        anchors = generate_anchors(feature_sizes[i])
        all_anchors.append(anchors)

    # stack everything together along the anchor dimension
    # so instead of 5 separate lists we get one big tensor
    all_locs = torch.cat(reshaped_locs, dim=1) # [N, total_anchors, 4]
    all_cls = torch.cat(reshaped_cls, dim=1) # [N, total_anchors, 2]
    all_anchors = torch.cat(all_anchors, dim=0) # [total_anchors, 4]

    return all_locs, all_cls, all_anchors


if __name__ == "__main__":
    # test generate_anchors with the feature sizes we expect from our SSD heads
    # backbone outputs 16x16, then each head halves it: 8x8, 4x4, 2x2, 1x1
    feature_sizes = [16, 8, 4, 2, 1]

    for fs in feature_sizes:
        anchors = generate_anchors(fs)
        print(f"feature size {fs}x{fs}: {anchors.shape[0]} anchors, shape {anchors.shape}")
        # show the first anchor as a sanity check
        print(f"  first anchor: cx={anchors[0][0]:.1f}, cy={anchors[0][1]:.1f}, w={anchors[0][2]:.1f}, h={anchors[0][3]:.1f}")

    # test reshape_predictions with fake SSD output
    # pretend we have a batch of 2 images, 16x16 feature map, 4 anchors, 2 classes
    print("\ntesting reshape_predictions")
    fake_loc = torch.randn(2, NUM_ANCHORS * 4, 16, 16) # [2, 16, 16, 16]
    fake_cls = torch.randn(2, NUM_ANCHORS * NUM_CLASSES, 16, 16) # [2, 8, 16, 16]
    print(f"  raw loc shape: {fake_loc.shape}")
    print(f"  raw cls shape: {fake_cls.shape}")

    loc_reshaped, cls_reshaped = reshape_predictions(fake_loc, fake_cls)
    print(f"  reshaped loc: {loc_reshaped.shape}") # should be [2, 1024, 4]
    print(f"  reshaped cls: {cls_reshaped.shape}") # should be [2, 1024, 2]

    # test concatenate_all_heads with fake outputs from all 5 heads
    print("\ntesting concatenate_all_heads")
    batch = 2
    all_locs = []
    all_cls = []
    for fs in feature_sizes:
        # fake what each SSD head would output
        all_locs.append(torch.randn(batch, NUM_ANCHORS * 4, fs, fs))
        all_cls.append(torch.randn(batch, NUM_ANCHORS * NUM_CLASSES, fs, fs))

    locs, cls, anchors = concatenate_all_heads(all_locs, all_cls, feature_sizes)
    print(f"  combined loc: {locs.shape}") # should be [2, 1364, 4]
    print(f"  combined cls: {cls.shape}") # should be [2, 1364, 2]
    print(f"  all anchors: {anchors.shape}") # should be [1364, 4]
    # 1364 = 1024 + 256 + 64 + 16 + 4
    print(f"  total anchors: 16*16*4 + 8*8*4 + 4*4*4 + 2*2*4 + 1*1*4 = {16*16*4 + 8*8*4 + 4*4*4 + 2*2*4 + 1*1*4}")
