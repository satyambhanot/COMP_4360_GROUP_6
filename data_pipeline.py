import os
import json
import random
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# where our stuff lives
RAW_IMAGES = "data/raw/images"
RAW_ANNOTATIONS = "data/raw/annotations"
PROCESSED = "data/processed"

# tile sizes we want to generate (multi-resolution like the paper says)
TILE_SIZES = [64, 128, 256]

# if at least this fraction of waldo is inside a tile, count it as positive
OVERLAP_THRESHOLD = 0.5


def parse_annotation(xml_path): # reads one xml file and pulls out the bounding box
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    obj = root.find("object") # theres only one waldo per image
    bbox = obj.find("bndbox")
    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)

    return img_w, img_h, [xmin, ymin, xmax, ymax]


def waldo_overlap(waldo_box, tile_box): # what fraction of waldo is inside this tile?
    xA = max(waldo_box[0], tile_box[0])
    yA = max(waldo_box[1], tile_box[1])
    xB = min(waldo_box[2], tile_box[2])
    yB = min(waldo_box[3], tile_box[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    waldo_area = (waldo_box[2] - waldo_box[0]) * (waldo_box[3] - waldo_box[1])
    return inter / waldo_area # fraction of waldo thats inside the tile


def tile_image(image_id): # chop one image into tiles at all resolutions
    img_path = os.path.join(RAW_IMAGES, f"{image_id}.jpg")
    xml_path = os.path.join(RAW_ANNOTATIONS, f"{image_id}.xml")

    img = Image.open(img_path)
    img_w, img_h, waldo_bbox = parse_annotation(xml_path)

    tiles_info = [] # will hold metadata for every tile we create

    for tile_size in TILE_SIZES:
        tile_dir = os.path.join(PROCESSED, f"{tile_size}")
        os.makedirs(tile_dir, exist_ok=True)

        # slide across the image with no overlap for now
        for y in range(0, img_h - tile_size + 1, tile_size):
            for x in range(0, img_w - tile_size + 1, tile_size):
                tile_box = [x, y, x + tile_size, y + tile_size]
                overlap = waldo_overlap(waldo_bbox, tile_box)

                has_waldo = 1 if overlap >= OVERLAP_THRESHOLD else 0

                # figure out waldo's position relative to this tile (if hes in it)
                local_bbox = None
                if has_waldo:
                    local_bbox = [
                        max(waldo_bbox[0] - x, 0), # clip to tile boundaries
                        max(waldo_bbox[1] - y, 0),
                        min(waldo_bbox[2] - x, tile_size),
                        min(waldo_bbox[3] - y, tile_size)
                    ]

                # crop and save the tile
                tile_img = img.crop(tile_box)
                tile_name = f"img{image_id}_{tile_size}_{x}_{y}.jpg"
                tile_img.save(os.path.join(tile_dir, tile_name))

                tiles_info.append({
                    "image_id": image_id,
                    "tile_size": tile_size,
                    "file": tile_name,
                    "x": x,
                    "y": y,
                    "has_waldo": has_waldo,
                    "local_bbox": local_bbox # None if no waldo
                })

    return tiles_info


def make_splits(all_tiles, train_ratio=0.7, val_ratio=0.15): # split by image id so no leaking
    image_ids = list(set(t["image_id"] for t in all_tiles))
    random.shuffle(image_ids)

    n = len(image_ids)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_ids = set(image_ids[:train_end])
    val_ids = set(image_ids[train_end:val_end])
    test_ids = set(image_ids[val_end:])

    splits = {"train": [], "val": [], "test": []}
    for tile in all_tiles:
        if tile["image_id"] in train_ids:
            splits["train"].append(tile)
        elif tile["image_id"] in val_ids:
            splits["val"].append(tile)
        else:
            splits["test"].append(tile)

    return splits


def run_pipeline():
    print("starting tiling pipeline...")

    # get all image ids from the annotation folder
    image_ids = []
    for f in os.listdir(RAW_ANNOTATIONS):
        if f.endswith(".xml"):
            image_ids.append(f.replace(".xml", ""))

    all_tiles = []
    for img_id in sorted(image_ids):
        print(f"  tiling image {img_id}...")
        tiles = tile_image(img_id)
        all_tiles.append(tiles)
        print(f"    got {len(tiles)} tiles")

    all_tiles = [t for sublist in all_tiles for t in sublist] # flatten

    # print some stats
    positive = sum(1 for t in all_tiles if t["has_waldo"])
    negative = len(all_tiles) - positive
    print(f"\ntotal tiles: {len(all_tiles)}")
    print(f"  positive (has waldo): {positive}")
    print(f"  negative: {negative}")

    for size in TILE_SIZES:
        count = sum(1 for t in all_tiles if t["tile_size"] == size)
        pos = sum(1 for t in all_tiles if t["tile_size"] == size and t["has_waldo"])
        print(f"  {size}x{size}: {count} tiles ({pos} positive)")

    # split into train/val/test
    random.seed(42) # reproducible splits
    splits = make_splits(all_tiles)

    for split_name, split_tiles in splits.items():
        pos = sum(1 for t in split_tiles if t["has_waldo"])
        print(f"  {split_name}: {len(split_tiles)} tiles ({pos} positive)")

    # save splits to json
    splits_path = os.path.join(PROCESSED, "splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nsaved splits to {splits_path}")


class WaldoTileDataset(Dataset): # pytorch dataset that loads tiles from splits.json

    def __init__(self, split="train", tile_size=256):
        splits_path = os.path.join(PROCESSED, "splits.json")
        with open(splits_path, "r") as f:
            all_splits = json.load(f)

        # only grab tiles of the requested size
        self.tiles = [t for t in all_splits[split] if t["tile_size"] == tile_size]
        self.tile_size = tile_size

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), # everything goes to 256x256 for the backbone
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet normalization
        ])

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile_info = self.tiles[idx]

        img_path = os.path.join(PROCESSED, str(self.tile_size), tile_info["file"])
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        label = tile_info["has_waldo"] # 0 or 1

        # bbox is [xmin, ymin, xmax, ymax] relative to tile, or [0,0,0,0] if no waldo
        if tile_info["local_bbox"] is not None:
            bbox = torch.tensor(tile_info["local_bbox"], dtype=torch.float32)
        else:
            bbox = torch.zeros(4, dtype=torch.float32)

        return img_tensor, label, bbox


def get_dataloader(split="train", tile_size=256, batch_size=16, shuffle=True):
    dataset = WaldoTileDataset(split=split, tile_size=tile_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


if __name__ == "__main__":
    run_pipeline()

    # quick test to make sure the dataset and dataloader work
    print("\n--- testing dataset ---")
    for size in TILE_SIZES:
        ds = WaldoTileDataset(split="train", tile_size=size)
        print(f"  {size}x{size} train set: {len(ds)} tiles")

    print("\n--- testing dataloader ---")
    loader = get_dataloader(split="train", tile_size=256, batch_size=4)
    images, labels, bboxes = next(iter(loader))
    print(f"  batch images shape: {images.shape}") # should be [4, 3, 256, 256]
    print(f"  batch labels: {labels}")
    print(f"  batch bboxes shape: {bboxes.shape}") # should be [4, 4]
