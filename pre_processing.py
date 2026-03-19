# pre_processing.py
# Handles HSV masking, Sobel edge detection, and ROI slicing for the Waldo pipeline

import torch
import torch.nn.functional as F

GRID_SIZE = 8  # 8x8 grid on 256x256 = 64 cells of 32x32 each


# --- Step 2: HSV Masking ---

def RGBtoHSV(images):
    """Converts normalized RGB images back to HSV so we can filter by color."""

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


def ApplyHSVMask(images):
    """Creates a binary mask highlighting red and white pixels (Waldo's stripes)."""

    hsv = RGBtoHSV(images)
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]

    # red sits at both ends of the hue wheel (near 0 and 360)
    red_mask   = ((h <= 15) | (h >= 345)) & (s > 0.4) & (v > 0.3)
    white_mask = (s < 0.15) & (v > 0.7)

    combined = (red_mask | white_mask).float().unsqueeze(1)  # [N, 1, H, W]
    return combined


# --- Step 3: Sobel Edge Detection ---

def ApplySobelEdgeDetection(images):
    """Runs Sobel filters to get an edge magnitude map. Useful because
    Waldo's stripes produce strong horizontal edges."""

    # undo normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb  = (images * std + mean).clamp(0, 1)

    # convert to grayscale using standard weights
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

    # normalize each image to [0, 1] range
    for i in range(magnitude.shape[0]):
        mag_max = magnitude[i].max()
        if mag_max > 0:
            magnitude[i] /= mag_max

    return magnitude  # [N, 1, H, W]


# --- Step 4: Slice and Order ROI ---

def SliceAndOrderROI(hsv_mask, sobel_edges, grid_size=GRID_SIZE):
    """Splits the image into a grid, scores each cell using the HSV mask
    and edge map, then returns cells ranked by how likely they contain Waldo."""

    N, _, H, W = hsv_mask.shape
    cell_h = H // grid_size
    cell_w = W // grid_size

    roi_rankings = []
    roi_regions  = []

    for img_idx in range(N):
        scores  = []
        regions = []

        for r in range(grid_size):
            for c in range(grid_size):
                y1, y2 = r * cell_h, (r + 1) * cell_h
                x1, x2 = c * cell_w, (c + 1) * cell_w

                hsv_score  = hsv_mask[img_idx, 0, y1:y2, x1:x2].mean().item()
                edge_score = sobel_edges[img_idx, 0, y1:y2, x1:x2].mean().item()

                # color is weighted higher since red-white is pretty unique to Waldo
                combined = 0.6 * hsv_score + 0.4 * edge_score

                scores.append([r, c, hsv_score, edge_score, combined])
                regions.append((r, c, y1, y2, x1, x2))

        scores_tensor  = torch.tensor(scores)
        sorted_indices = scores_tensor[:, 4].argsort(descending=True)

        roi_rankings.append(scores_tensor[sorted_indices])
        roi_regions.append([regions[i] for i in sorted_indices])

    return roi_rankings, roi_regions
