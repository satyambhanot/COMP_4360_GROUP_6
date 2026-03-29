'''
anchors.py
handles anchor boxes for our custom SSD detector
each SSD head makes predictions at different grid sizes (16x16, 8x8, 4x4, etc)
at every grid position we place anchor boxes of different sizes
the SSD heads predict offsets from these anchors, not absolute positions
so we need to know where the anchors actually are on the image

this file does three things:
  1. generate anchor box positions for a given grid size
  2. reshape the raw SSD output into a usable format
  3. combine predictions from all heads into one big tensor
'''

import torch
from config import NUM_ANCHORS, NUM_CLASSES, INPUT_SIZE


def generate_anchors(feature_size, image_size=INPUT_SIZE):
    '''
    creates anchor box positions for one SSD head
    for each grid cell we place NUM_ANCHORS boxes of different sizes

    args:
        feature_size: spatial size of the feature map (16 for a 16x16 grid)
        image_size: original tile size in pixels (default 256)

    returns:
        tensor of anchor positions [feature_size^2 * NUM_ANCHORS, 4]
        each row is [center_x, center_y, width, height] in pixels
    '''

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
    '''
    reshapes raw SSD head output from convolution format into flat prediction lists
    no values are changed, just rearranged so each row is one anchor prediction

    args:
        loc: bounding box predictions [N, NUM_ANCHORS*4, H, W]
        cls: class score predictions [N, NUM_ANCHORS*NUM_CLASSES, H, W]

    returns:
        loc: reshaped to [N, H*W*NUM_ANCHORS, 4] (one row per anchor)
        cls: reshaped to [N, H*W*NUM_ANCHORS, NUM_CLASSES] (one row per anchor)
    '''

    batch_size = loc.shape[0]

    # loc: [N, anchors*4, H, W] -> [N, H, W, anchors*4] -> [N, H*W*anchors, 4]
    loc = loc.permute(0, 2, 3, 1).contiguous() # move channels to the end
    loc = loc.view(batch_size, -1, 4) # flatten into rows of 4

    # cls: [N, anchors*classes, H, W] -> [N, H, W, anchors*classes] -> [N, H*W*anchors, classes]
    cls = cls.permute(0, 2, 3, 1).contiguous()
    cls = cls.view(batch_size, -1, NUM_CLASSES)

    return loc, cls


def concatenate_all_heads(all_loc_preds, all_cls_preds, feature_sizes):
    '''
    combines predictions from all SSD heads into single tensors
    reshapes each heads output into flat lists then stacks them together

    args:
        all_loc_preds: list of loc tensors from each head
        all_cls_preds: list of cls tensors from each head
        feature_sizes: list like [16, 8, 4, 2, 1] for each heads grid size

    returns:
        all_locs: [N, total_anchors, 4] every box prediction from every head
        all_cls: [N, total_anchors, 2] every class score from every head
        all_anchors: [total_anchors, 4] the actual anchor positions on the image
    '''

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

    # make sure anchors are on the same device as the predictions
    # generate_anchors creates them on CPU but if the model is on GPU
    # everything needs to be on the same device or pytorch will crash
    all_anchors = all_anchors.to(all_locs.device)

    return all_locs, all_cls, all_anchors
