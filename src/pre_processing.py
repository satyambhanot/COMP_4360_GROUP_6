# pre_processing.py
# takes image tiles and adds extra channels that help the model spot waldo
# we concatenate the original RGB with a HSV color mask and sobel edges
# so instead of just 3 channels (RGB) the model gets 5 channels (RGB + HSV mask + edges)

import torch
import torch.nn.functional as F


def RGBtoHSV(images): # converts normalized RGB images to HSV color space
    # undo the ImageNet normalization first
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
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
    red_mask   = ((h <= 15) | (h >= 345)) & (s > 0.4) & (v > 0.3)
    white_mask = (s < 0.15) & (v > 0.7)

    combined = (red_mask | white_mask).float().unsqueeze(1)  # [N, 1, H, W]
    return combined


def ApplySobelEdgeDetection(images): # edge detection - waldos stripes make strong horizontal edges
    # undo normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb  = (images * std + mean).clamp(0, 1)

    # convert to grayscale
    gray = (0.2989 * rgb[:, 0] + 0.5870 * rgb[:, 1] + 0.1140 * rgb[:, 2]).unsqueeze(1)

    # sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    sobel_y = torch.tensor([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)

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
    hsv_mask = ApplyHSVMask(images)       # [N, 1, 256, 256] - red/white pixel mask
    sobel = ApplySobelEdgeDetection(images) # [N, 1, 256, 256] - edge magnitude

    # stack RGB + HSV mask + sobel edges = 5 channels total
    # this gives the model extra hints about where waldo might be
    return torch.cat([images, hsv_mask, sobel], dim=1)  # [N, 5, 256, 256]


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

    sobel = ApplySobelEdgeDetection(images)
    print(f"sobel shape: {sobel.shape}") # [4, 1, 256, 256]

    combined = preprocess_batch(images)
    print(f"combined shape: {combined.shape}") # [4, 5, 256, 256]
