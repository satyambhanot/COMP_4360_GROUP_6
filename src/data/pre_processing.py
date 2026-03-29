'''
pre_processing.py
takes image tiles and adds extra channels that help the model spot waldo
we concatenate the original RGB with a HSV color mask and sobel edges
so instead of just 3 channels (RGB) the model gets 5 channels

originally we ran sobel on the full grayscale image which found edges of
everything: buildings, people, trees, signs. we realized it makes more sense
to run sobel on the HSV mask instead. that way the edge channel only shows
edges around red and white regions which is basically waldos stripe outline.

we kept the old apply_sobel_edge_detection function around for the ablation
studies where we compare different preprocessing approaches
'''

import torch
import torch.nn.functional as F

from config import (
    IMAGENET_MEAN, IMAGENET_STD,
    RED_HUE_MAX, RED_HUE_MIN, RED_SAT_MIN, RED_VAL_MIN,
    WHITE_SAT_MAX, WHITE_VAL_MIN
)


def rgb_to_hsv(images):
    '''
    converts a batch of normalized RGB images to HSV color space
    undoes the ImageNet normalization first since HSV needs raw 0-1 RGB values

    args:
        images: batch of ImageNet normalized images [N, 3, H, W]

    returns:
        HSV tensor [N, 3, H, W] where channels are hue (0-360), saturation (0-1), value (0-1)
    '''
    # undo ImageNet normalization to get back to 0-1 range
    # .to(images.device) so tensors end up on GPU if input is on GPU
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(images.device)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(images.device)
    rgb  = (images * std + mean).clamp(0, 1)

    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    c_max, _ = rgb.max(dim=1)
    c_min, _ = rgb.min(dim=1)
    diff = c_max - c_min

    # compute hue based on which RGB channel is dominant
    hue = torch.zeros_like(c_max)

    mask_r = (c_max == r) & (diff > 0)
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r])) % 360

    mask_g = (c_max == g) & (diff > 0)
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360

    mask_b = (c_max == b) & (diff > 0)
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360

    # saturation = color difference / max brightness
    sat = torch.zeros_like(c_max)
    sat[c_max > 0] = diff[c_max > 0] / c_max[c_max > 0]

    return torch.stack([hue, sat, c_max], dim=1)  # [N, 3, H, W]


def apply_hsv_mask(images):
    '''
    creates a binary mask highlighting red and white pixels in the image
    waldos shirt has red and white stripes so this gives the model a hint
    about where to look

    args:
        images: batch of ImageNet normalized images [N, 3, H, W]

    returns:
        binary mask [N, 1, H, W] where 1 means the pixel is red or white
    '''
    hsv = rgb_to_hsv(images)
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]

    # red sits at both ends of the hue wheel (near 0 and near 360)
    red_mask   = ((h <= RED_HUE_MAX) | (h >= RED_HUE_MIN)) & (s > RED_SAT_MIN) & (v > RED_VAL_MIN)
    white_mask = (s < WHITE_SAT_MAX) & (v > WHITE_VAL_MIN)

    combined = (red_mask | white_mask).float().unsqueeze(1)  # [N, 1, H, W]
    return combined


def apply_sobel_edge_detection(images):
    '''
    runs sobel edge detection on the raw grayscale image
    finds edges of everything in the scene (not just waldo)
    kept around for ablation studies to compare with sobel on HSV mask

    args:
        images: batch of ImageNet normalized images [N, 3, H, W]

    returns:
        edge magnitude [N, 1, H, W] normalized to 0-1 per image
    '''
    # undo normalization to get raw RGB
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(images.device)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(images.device)
    rgb  = (images * std + mean).clamp(0, 1)

    # convert to grayscale using standard luminance weights
    gray = (0.2989 * rgb[:, 0] + 0.5870 * rgb[:, 1] + 0.1140 * rgb[:, 2]).unsqueeze(1)

    # standard 3x3 sobel kernels for horizontal and vertical edges
    sobel_x = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)

    sobel_y = torch.tensor([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(gx ** 2 + gy ** 2)

    # normalize each image independently to [0, 1]
    for i in range(magnitude.shape[0]):
        mag_max = magnitude[i].max()
        if mag_max > 0:
            magnitude[i] /= mag_max

    return magnitude  # [N, 1, H, W]


def preprocess_batch(images):
    '''
    the main preprocessing function that adds HSV mask and sobel channels
    takes standard 3 channel RGB tiles and returns 5 channel tensors

    args:
        images: batch of ImageNet normalized images [N, 3, H, W]

    returns:
        5 channel tensor [N, 5, H, W] with:
          channels 0-2: original RGB
          channel 3: HSV mask (red/white pixel locations)
          channel 4: sobel edges of the HSV mask (stripe outlines)
    '''
    hsv_mask = apply_hsv_mask(images)  # [N, 1, H, W]

    # run sobel on the HSV mask (not the raw image) so edges only appear
    # around red and white regions, which is waldos stripe pattern
    sobel_x = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)

    sobel_y = torch.tensor([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)

    gx = F.conv2d(hsv_mask, sobel_x, padding=1)
    gy = F.conv2d(hsv_mask, sobel_y, padding=1)
    sobel_of_hsv = torch.sqrt(gx ** 2 + gy ** 2)  # [N, 1, H, W]

    return torch.cat([images, hsv_mask, sobel_of_hsv], dim=1)  # [N, 5, H, W]
