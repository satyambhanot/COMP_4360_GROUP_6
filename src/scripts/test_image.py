'''
test_image.py
run waldo detection on any image using a saved checkpoint
tiles the image, runs inference on each tile, and draws bounding boxes
where the model thinks waldo is

usage:
  python src/scripts/test_image.py path/to/image.jpg
  python src/scripts/test_image.py path/to/image.jpg --output results/my_result.png
  python src/scripts/test_image.py path/to/image.jpg --checkpoint checkpoints/best_model.pth

if no image is given, picks a random image from data/raw/images/
'''

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from models.model import WaldoDetector, WaldoDetectorPretrained, WaldoDetectorRetinaNet
from models.nms import apply_nms
from data.pre_processing import apply_hsv_mask
from utils import get_device
from config import (
    INPUT_SIZE, BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    MODEL_TYPE, NMS_CONFIDENCE_THRESHOLD, RETINANET_INPUT_SIZE,
    HSV_FILTER_THRESHOLD, RAW_IMAGES, CHECKPOINTS_DIR
)


def load_model(checkpoint_path, device):
    '''load model from checkpoint'''
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: no checkpoint found at {checkpoint_path}")
        print("train a model first with: bash run.sh")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", MODEL_TYPE)

    if model_type == "retinanet":
        model = WaldoDetectorRetinaNet()
    elif model_type == "pretrained_ssd":
        model = WaldoDetectorPretrained()
    else:
        model = WaldoDetector()

    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.model_type = model_type

    print(f"loaded {model_type} model from epoch {checkpoint['epoch']}")
    return model


def detect_waldo(image_path, checkpoint_path, output_path, device):
    '''run detection on a single image and save the result'''

    model = load_model(checkpoint_path, device)
    if model is None:
        return

    image = Image.open(image_path).convert("RGB")
    print(f"image: {image_path} ({image.size[0]}x{image.size[1]})")

    # tile the image
    tile_size = INPUT_SIZE
    tiles = []
    tile_coords = []

    width, height = image.size
    for h in range(0, height, tile_size):
        for w in range(0, width, tile_size):
            tile = image.crop((w, h, w + tile_size, h + tile_size))
            tiles.append(tile)
            tile_coords.append((w, h))

    print(f"created {len(tiles)} tiles at {tile_size}x{tile_size}")

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    all_detections = []
    tiles_with_detections = 0

    for batch_start in range(0, len(tiles), BATCH_SIZE):
        batch_tiles = tiles[batch_start:batch_start + BATCH_SIZE]
        batch_coords = tile_coords[batch_start:batch_start + BATCH_SIZE]

        tile_tensors = torch.stack([transform(t) for t in batch_tiles])
        tile_tensors = tile_tensors.to(device, non_blocking=True)

        # hsv pre-filter: skip tiles with no red/white pixels
        hsv_masks = apply_hsv_mask(tile_tensors)
        hsv_fractions = hsv_masks.mean(dim=(1, 2, 3))

        with torch.no_grad():
            if hasattr(model, "model_type") and model.model_type in ("pretrained_ssd", "retinanet"):
                # only run inference on tiles that pass the HSV filter
                filtered_indices = (hsv_fractions >= HSV_FILTER_THRESHOLD).nonzero(as_tuple=True)[0]
                if len(filtered_indices) == 0:
                    continue

                filtered_tensors = tile_tensors[filtered_indices]
                detections = model(filtered_tensors)

                for j, fi in enumerate(filtered_indices):
                    det = detections[j]
                    waldo_mask = det["labels"] == 1
                    boxes = det["boxes"][waldo_mask]
                    scores = det["scores"][waldo_mask]

                    # filter by confidence
                    conf_mask = scores >= NMS_CONFIDENCE_THRESHOLD
                    boxes = boxes[conf_mask]
                    scores = scores[conf_mask]

                    if boxes.shape[0] > 0:
                        w_off, h_off = batch_coords[fi.item()]
                        model_input = RETINANET_INPUT_SIZE if model.model_type == "retinanet" else 300.0
                        scale = INPUT_SIZE / model_input
                        adjusted_boxes = boxes.cpu().clone() * scale
                        adjusted_boxes[:, 0] += w_off
                        adjusted_boxes[:, 1] += h_off
                        adjusted_boxes[:, 2] += w_off
                        adjusted_boxes[:, 3] += h_off
                        all_detections.append((adjusted_boxes, scores.cpu()))
                        tiles_with_detections += 1
            else:
                locs, cls_scores, anchors = model(tile_tensors)
                nms_results = apply_nms(locs, cls_scores, anchors)
                for i in range(len(batch_tiles)):
                    boxes, scores = nms_results[i]
                    if boxes.shape[0] > 0:
                        w_off, h_off = batch_coords[i]
                        adjusted_boxes = boxes.cpu().clone()
                        adjusted_boxes[:, 0] += w_off
                        adjusted_boxes[:, 1] += h_off
                        adjusted_boxes[:, 2] += w_off
                        adjusted_boxes[:, 3] += h_off
                        all_detections.append((adjusted_boxes, scores.cpu()))
                        tiles_with_detections += 1

    # draw detections on the image
    draw = ImageDraw.Draw(image)
    total_boxes = 0
    for boxes, scores in all_detections:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            score = scores[i].item()
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            draw.text((x1, y1 - 12), f"{score:.2f}", fill="lime")
            total_boxes += 1

    image.save(output_path)
    print(f"found {total_boxes} detection(s) in {tiles_with_detections} tile(s)")
    print(f"saved result to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect Waldo in an image")
    parser.add_argument("image", nargs="?", help="path to input image (default: random from data/raw/images/)")
    parser.add_argument("--output", "-o", help="path to save result image (default: results/test_result.png)")
    parser.add_argument("--checkpoint", "-c", help="path to model checkpoint",
                        default=os.path.join(CHECKPOINTS_DIR, "best_model.pth"))
    args = parser.parse_args()

    device = get_device()
    print(f"using device: {device}")

    # pick an image
    if args.image:
        image_path = args.image
    else:
        images = sorted(os.listdir(RAW_IMAGES))
        if not images:
            print("ERROR: no images found in data/raw/images/")
            sys.exit(1)
        import random
        image_path = os.path.join(RAW_IMAGES, random.choice(images))
        print(f"no image specified, picked: {os.path.basename(image_path)}")

    if not os.path.exists(image_path):
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)

    # output path
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)

    if args.output:
        output_path = args.output
    else:
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(results_dir, f"test_{img_name}.png")

    detect_waldo(image_path, args.checkpoint, output_path, device)


if __name__ == "__main__":
    main()
