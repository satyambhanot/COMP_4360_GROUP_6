# pre_processing.py
# takes image tiles and adds extra channels that help the model spot waldo
# we concatenate the original RGB with a HSV color mask and sobel edges of the HSV mask
# so instead of just 3 channels (RGB) the model gets 5 channels (RGB + HSV mask + edges of HSV)
#
# originally we ran sobel on the full grayscale image which found edges of everything
# buildings, people, trees, signs, all creating edges that have nothing to do with waldo
# we realized it makes more sense to run sobel on the HSV mask instead
# that way the edge channel only shows edges around red and white regions
# which is basically the outline of waldos stripes
# much more focused and useful than edges of random stuff in the background
#
# we kept the old ApplySobelEdgeDetection function around in case we want to
# compare the two approaches in our ablation studies later

import torch
import torch.nn.functional as F
from config import (
    IMAGENET_MEAN, IMAGENET_STD,
    RED_HUE_MAX, RED_HUE_MIN, RED_SAT_MIN, RED_VAL_MIN,
    WHITE_SAT_MAX, WHITE_VAL_MIN
)


def RGBtoHSV(images): # converts normalized RGB images to HSV color space
    # undo the ImageNet normalization first
    # we use .to(images.device) so these tensors end up on the same device as the input
    # without this it crashes on GPU because mean/std would be on CPU but images on GPU
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(images.device)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(images.device)
    rgb  = (images * std + mean).clamp(0, 1)

    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    c_max, _ = rgb.max(dim=1)
    c_min, _ = rgb.min(dim=1)
    diff = c_max - c_min

    # compute hue based on which channel is dominant
    hue = torch.zeros_like(c_max)

    mask_r = (c_max == r) & (diff > 0)
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r])) % 360

    mask_g = (c_max == g) & (diff > 0)
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360

    mask_b = (c_max == b) & (diff > 0)
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360

    # saturation
    sat = torch.zeros_like(c_max)
    sat[c_max > 0] = diff[c_max > 0] / c_max[c_max > 0]

    return torch.stack([hue, sat, c_max], dim=1)  # [N, 3, H, W]


def ApplyHSVMask(images): # binary mask that highlights red and white pixels (waldos stripes)
    hsv = RGBtoHSV(images)
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]

    # red sits at both ends of the hue wheel (near 0 and 360)
    red_mask   = ((h <= RED_HUE_MAX) | (h >= RED_HUE_MIN)) & (s > RED_SAT_MIN) & (v > RED_VAL_MIN)
    white_mask = (s < WHITE_SAT_MAX) & (v > WHITE_VAL_MIN)

    combined = (red_mask | white_mask).float().unsqueeze(1)  # [N, 1, H, W]
    return combined


def ApplySobelEdgeDetection(images): # edge detection, waldos stripes make strong horizontal edges
    # undo normalization
    # same device fix as RGBtoHSV, tensors need to be on the same device as input
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(images.device)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(images.device)
    rgb  = (images * std + mean).clamp(0, 1)

    # convert to grayscale
    gray = (0.2989 * rgb[:, 0] + 0.5870 * rgb[:, 1] + 0.1140 * rgb[:, 2]).unsqueeze(1)

    # sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)

    sobel_y = torch.tensor([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(gx ** 2 + gy ** 2)

    # normalize each image to [0, 1]
    for i in range(magnitude.shape[0]):
        mag_max = magnitude[i].max()
        if mag_max > 0:
            magnitude[i] /= mag_max

    return magnitude  # [N, 1, H, W]


def preprocess_batch(images): # takes [N, 3, 256, 256] and returns [N, 5, 256, 256]
    hsv_mask = ApplyHSVMask(images) # [N, 1, 256, 256] binary mask of red/white pixels

    # instead of running sobel on the full grayscale image (which finds edges of everything)
    # we run it on the HSV mask so it only finds edges around red and white regions
    # the HSV mask is binary (0 or 1) so the edges show up at the boundaries
    # of red/white areas, which is basically the outline of waldos stripes
    # the sobel kernels are the standard ones defined by Irwin Sobel in 1968
    # sobel_x finds vertical edges, sobel_y finds horizontal edges
    # make sure sobel kernels are on the same device as the input images
    sobel_x = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)

    sobel_y = torch.tensor([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)

    # run the sobel kernels on the hsv mask, not the raw image
    gx = F.conv2d(hsv_mask, sobel_x, padding=1)
    gy = F.conv2d(hsv_mask, sobel_y, padding=1)
    sobel_of_hsv = torch.sqrt(gx ** 2 + gy ** 2) # [N, 1, 256, 256] edges of red/white regions

    # stack RGB + HSV mask + sobel of HSV = 5 channels total
    # channel 1,2,3: the original RGB image
    # channel 4: where the red and white pixels are
    # channel 5: the edges around those red and white regions
    return torch.cat([images, hsv_mask, sobel_of_hsv], dim=1)  # [N, 5, 256, 256]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_pipeline import get_dataloader

    # test with real tiles
    loader = get_dataloader(split="train", tile_size=256, batch_size=4)
    images, labels, bboxes = next(iter(loader))

    print(f"input shape: {images.shape}") # [4, 3, 256, 256]

    hsv_mask = ApplyHSVMask(images)
    print(f"hsv mask shape: {hsv_mask.shape}") # [4, 1, 256, 256]

    # test the old sobel on raw image (kept for ablation studies)
    sobel_raw = ApplySobelEdgeDetection(images)
    print(f"sobel on raw image shape: {sobel_raw.shape}") # [4, 1, 256, 256]

    # test the new preprocess_batch which uses sobel on hsv mask
    combined = preprocess_batch(images)
    print(f"combined shape: {combined.shape}") # [4, 5, 256, 256]
    print(f"  channel 1,2,3: RGB")
    print(f"  channel 4: HSV mask (red/white pixels)")
    print(f"  channel 5: sobel edges of HSV mask (outlines of red/white regions)")
