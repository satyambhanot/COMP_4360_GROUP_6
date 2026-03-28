'''
# data_pipeline.py
# this file handles everything related to getting our data ready for the model
# it takes our raw wheres waldo images and their xml annotations,
# chops them into 256x256 tiles using multiple grid offsets,
# runs the random crop augmentation for extra positive tiles,
# figures out which tiles actually contain waldo,
# splits everything into train/val/test sets,
# and provides a pytorch dataset and dataloader so other files can easily grab batches
'''

import os
import json
import random
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import functional as TF
from config import *

'''
reads one xml file and pulls out the bounding box
'''
def parse_annotation(xml_path): 

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

'''
# what fraction of waldo is inside this tile?
'''
def waldo_overlap(waldo_box, tile_box): 
    xA = max(waldo_box[0], tile_box[0])
    yA = max(waldo_box[1], tile_box[1])
    xB = min(waldo_box[2], tile_box[2])
    yB = min(waldo_box[3], tile_box[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    waldo_area = (waldo_box[2] - waldo_box[0]) * (waldo_box[3] - waldo_box[1])
    return inter / waldo_area # fraction of waldo thats inside the tile


'''
# chop one image into tiles at all resolutions and grid offsets
'''
def tile_image(image_id): 
    img_path = os.path.join(RAW_IMAGES, f"{image_id}.jpg")
    xml_path = os.path.join(RAW_ANNOTATIONS, f"{image_id}.xml")

    img = Image.open(img_path)
    img_w, img_h, waldo_bbox = parse_annotation(xml_path)

    tiles_info = [] # will hold metadata for every tile we create

    for tile_size in TILE_SIZES:
        tile_dir = os.path.join(PROCESSED, f"{tile_size}")
        os.makedirs(tile_dir, exist_ok=True)

        for ox, oy in TILE_OFFSETS:
        
            if ox >= img_w or oy >= img_h:
                continue

            for y in range(oy, img_h - tile_size + 1, tile_size):
                for x in range(ox, img_w - tile_size + 1, tile_size):
                    tile_box = [x, y, x + tile_size, y + tile_size]
                    overlap = waldo_overlap(waldo_bbox, tile_box)

                    has_waldo = 1 if overlap >= OVERLAP_THRESHOLD else 0

                    local_bbox = None
                    if has_waldo:
                        local_bbox = [
                            max(waldo_bbox[0] - x, 0), # clip to tile boundaries
                            max(waldo_bbox[1] - y, 0),
                            min(waldo_bbox[2] - x, tile_size),
                            min(waldo_bbox[3] - y, tile_size)
                        ]

                
                    tile_img = img.crop(tile_box)
                    tile_name = f"img{image_id}_{tile_size}_o{ox}_{oy}_{x}_{y}.jpg"
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


def make_splits(all_tiles, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO): # split by image id so no leaking
    image_ids = list(set(t["image_id"] for t in all_tiles))
    random.shuffle(image_ids)

    n = len(image_ids)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_ids = set(image_ids[:train_end])
    val_ids = set(image_ids[train_end:val_end])

    splits = {"train": [], "val": [], "test": []}
    for tile in all_tiles:
        if tile["image_id"] in train_ids:
            splits["train"].append(tile)
        elif tile["image_id"] in val_ids:
            splits["val"].append(tile)
        else:
            splits["test"].append(tile)

    return splits

'''
# the main function that does everything, tiles all images and creates the splits
'''
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

    '''
    # run the random crop augmentation to generate extra positive tiles
    # this takes random 256x256 patches centered around waldo from the larger images
    # each patch has waldo at a different position with different surrounding context
    # we import it here so the augmented tiles get included in the same splits
    '''
    from augment import run_augmentation
    augmented_tiles = run_augmentation()
    all_tiles.extend(augmented_tiles)

    # print some stats
    positive = sum(1 for t in all_tiles if t["has_waldo"])
    negative = len(all_tiles) - positive
    print(f"\ntotal tiles (grid + augmented): {len(all_tiles)}")
    print(f"  positive (has waldo): {positive}")
    print(f"  negative: {negative}")

    for size in TILE_SIZES:
        count = sum(1 for t in all_tiles if t["tile_size"] == size)
        pos = sum(1 for t in all_tiles if t["tile_size"] == size and t["has_waldo"])
        print(f"  {size}x{size}: {count} tiles ({pos} positive)")

    # split into train/val/test
    random.seed(RANDOM_SEED) # reproducible splits
    splits = make_splits(all_tiles)

    for split_name, split_tiles in splits.items():
        pos = sum(1 for t in split_tiles if t["has_waldo"])
        print(f"  {split_name}: {len(split_tiles)} tiles ({pos} positive)")

    # save splits to json
    splits_path = os.path.join(PROCESSED, "splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nsaved splits to {splits_path}")


'''
# pytorch dataset that loads tiles from splits.json
# when you grab a tile from this dataset it gives you three things:
#   image tensor [3, 256, 256] normalized for mobilenet
#   label (0 = no waldo, 1 = waldo is in this tile)
#   bbox [4] with waldos position inside the tile, or [0,0,0,0] if no waldo
'''
class WaldoTileDataset(Dataset):

    def __init__(self, split="train", tile_size=256):
        splits_path = os.path.join(PROCESSED, "splits.json")
        with open(splits_path, "r") as f:
            all_splits = json.load(f)

        self.tiles = [t for t in all_splits[split] if t["tile_size"] == tile_size]
        self.tile_size = tile_size
        self.split = split 

        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        '''
        # color jitter for positive training tiles only
        # this randomly tweaks brightness, contrast, and saturation each time
        # so the model sees a slightly different waldo every epoch
        # we only change colors not positions so the bounding box stays the same
        # we dont apply this to val/test because we want clean untouched data for evaluation
        '''
        self.color_jitter = transforms.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CONTRAST,
            saturation=COLOR_JITTER_SATURATION
        )

        # additional augmentations for ALL training tiles (not just positive)       
        self.blur = transforms.GaussianBlur(kernel_size=BLUR_KERNEL_SIZE, sigma=(0.1, 1.0))
        self.sharpness = transforms.RandomAdjustSharpness(sharpness_factor=SHARPNESS_FACTOR, p=SHARPNESS_PROBABILITY)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile_info = self.tiles[idx]

        img_path = os.path.join(PROCESSED, str(self.tile_size), tile_info["file"])
        img = Image.open(img_path).convert("RGB")

        # get the bounding box (or zeros if no waldo)
        if tile_info["local_bbox"] is not None:
            bbox = torch.tensor(tile_info["local_bbox"], dtype=torch.float32)
        else:
            bbox = torch.zeros(4, dtype=torch.float32)

        label = tile_info["has_waldo"] # 0 or 1

        if self.split != "train" or not AUGMENT_TRAINING:
            img_tensor = self.transform(img)
            return img_tensor, label, bbox

        img = img.resize((INPUT_SIZE, INPUT_SIZE))

        '''
        # horizontal flip with 50% chance
        # waldo looks the same facing left or right so this is a safe augmentation
        '''
        if random.random() < FLIP_PROBABILITY:
            img = TF.hflip(img) # flip the image
            if label == 1:
                # flip the bbox: new_xmin = tile_width minus old_xmax
                xmin, ymin, xmax, ymax = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
                # scale bbox to INPUT_SIZE since we already resized
                scale_x = INPUT_SIZE / self.tile_size
                scale_y = INPUT_SIZE / self.tile_size
                new_xmin = INPUT_SIZE - (xmax * scale_x)
                new_xmax = INPUT_SIZE - (xmin * scale_x)
                bbox = torch.tensor([new_xmin, ymin * scale_y, new_xmax, ymax * scale_y], dtype=torch.float32)

        '''
        # color jitter on positive tiles only
        # changes brightness, contrast, saturation randomly
        '''
        if label == 1:
            img = self.color_jitter(img)

        '''
        # gaussian blur with 20% chance on all training tiles
        # makes the model robust to slightly blurry images
        '''
        if random.random() < BLUR_PROBABILITY:
            img = self.blur(img)

        '''
        # random sharpness adjustment on all training tiles
        '''
        img = self.sharpness(img)

        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(img_tensor)

        return img_tensor, label, bbox


def get_dataloader(split="train", tile_size=256, batch_size=BATCH_SIZE, shuffle=True):
    
    dataset = WaldoTileDataset(split=split, tile_size=tile_size)

    '''
    # for training we use weighted sampling to handle the class imbalance
    # we have about 14000 negative tiles but only 56 positive ones
    # without this the model would almost never see a positive tile during training
    # the sampler gives positive tiles a higher chance of being picked
    # we compute the weight dynamically so it adjusts if the dataset changes
    '''
    if split == "train" and AUGMENT_TRAINING:
        num_positive = sum(1 for t in dataset.tiles if t["has_waldo"])
        num_negative = len(dataset.tiles) - num_positive

        '''
        # we divide by SAMPLING_BALANCE_FACTOR so its not a full 50/50 split
        '''
        positive_weight = num_negative / num_positive / SAMPLING_BALANCE_FACTOR
        weights = [positive_weight if t["has_waldo"] else 1.0 for t in dataset.tiles]

        epoch_samples = min(len(dataset), 4000)
        sampler = WeightedRandomSampler(weights, num_samples=epoch_samples)
       
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    elif split == "val":

        val_samples = min(len(dataset), 1000)
        val_sampler = WeightedRandomSampler(
            [1.0] * len(dataset), num_samples=val_samples
        )
        loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    return loader


if __name__ == "__main__":
    run_pipeline()

    # quick test to make sure the dataset and dataloader work
    print("\ntesting dataset")
    for size in TILE_SIZES:
        ds = WaldoTileDataset(split="train", tile_size=size)
        print(f"  {size}x{size} train set: {len(ds)} tiles")

    print("\ntesting dataloader")
    loader = get_dataloader(split="train", tile_size=256, batch_size=4)
    images, labels, bboxes = next(iter(loader))
    print(f"  batch images shape: {images.shape}") # should be [4, 3, 256, 256]
    print(f"  batch labels: {labels}")
    print(f"  batch bboxes shape: {bboxes.shape}") # should be [4, 4]
