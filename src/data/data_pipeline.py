'''
data_pipeline.py
handles everything related to getting our data ready for the model
takes raw wheres waldo images and their xml annotations, chops them
into 256x256 tiles using multiple grid offsets, runs random crop
augmentation for extra positive tiles, figures out which tiles
contain waldo, splits into train/val/test sets, and provides a
pytorch dataset and dataloader for training and evaluation
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
from config import (
    RAW_IMAGES, RAW_ANNOTATIONS, PROCESSED,
    TILE_SIZES, TILE_OFFSETS, OVERLAP_THRESHOLD,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
    INPUT_SIZE, BATCH_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
    AUGMENT_TRAINING, SAMPLING_BALANCE_FACTOR,
    COLOR_JITTER_BRIGHTNESS, COLOR_JITTER_CONTRAST, COLOR_JITTER_SATURATION,
    FLIP_PROBABILITY,
    BLUR_PROBABILITY, BLUR_KERNEL_SIZE,
    SHARPNESS_PROBABILITY, SHARPNESS_FACTOR
)


def parse_annotation(xml_path):
    '''
    reads one Pascal VOC xml annotation file and extracts the bounding box

    args:
        xml_path: path to the xml annotation file

    returns:
        img_w: image width in pixels
        img_h: image height in pixels
        bbox: [xmin, ymin, xmax, ymax] waldo bounding box coordinates
    '''
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


def waldo_overlap(waldo_box, tile_box):
    '''
    computes what fraction of waldo is inside a given tile
    used to decide if a tile counts as positive (contains waldo)

    args:
        waldo_box: [xmin, ymin, xmax, ymax] waldo bounding box
        tile_box: [xmin, ymin, xmax, ymax] tile bounding box

    returns:
        fraction of waldo area that overlaps with the tile (0.0 to 1.0)
    '''
    xA = max(waldo_box[0], tile_box[0])
    yA = max(waldo_box[1], tile_box[1])
    xB = min(waldo_box[2], tile_box[2])
    yB = min(waldo_box[3], tile_box[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    waldo_area = (waldo_box[2] - waldo_box[0]) * (waldo_box[3] - waldo_box[1])
    return inter / waldo_area # fraction of waldo thats inside the tile


def tile_image(image_id):
    '''
    chops one image into 256x256 tiles using multiple grid offsets
    each offset shifts where the tile grid starts so waldo lands on
    different tile boundaries giving us more diverse positive tiles

    args:
        image_id: string id of the image (matches filename without extension)

    returns:
        list of tile info dicts with image_id, tile_size, file, x, y,
        has_waldo, and local_bbox for each generated tile
    '''
    img_path = os.path.join(RAW_IMAGES, f"{image_id}.jpg")
    xml_path = os.path.join(RAW_ANNOTATIONS, f"{image_id}.xml")

    img = Image.open(img_path)
    img_w, img_h, waldo_bbox = parse_annotation(xml_path)

    tiles_info = [] # will hold metadata for every tile we create

    for tile_size in TILE_SIZES:
        tile_dir = os.path.join(PROCESSED, f"{tile_size}")
        os.makedirs(tile_dir, exist_ok=True)

        '''
        tile with multiple grid offsets so waldo lands on different boundaries
        the (0,0) offset is the original grid, the others shift by half a tile
        this means a waldo that was split across two tiles in one grid
        might be fully inside a single tile in another grid
        '''
        for ox, oy in TILE_OFFSETS:
            # skip offsets that dont fit in the image
            # no point starting at offset 128 if the image is only 300px wide
            if ox >= img_w or oy >= img_h:
                continue

            for y in range(oy, img_h - tile_size + 1, tile_size):
                for x in range(ox, img_w - tile_size + 1, tile_size):
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
                    # include offset in the filename so tiles from different grids dont collide
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


def make_splits(all_tiles, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    '''
    splits tiles into train/val/test sets by image id (not by tile)
    this prevents data leaking where tiles from the same image end up
    in both training and testing

    args:
        all_tiles: list of all tile info dicts
        train_ratio: fraction for training (default 0.7)
        val_ratio: fraction for validation (default 0.15)

    returns:
        dict with "train", "val", "test" keys each containing a list of tiles
    '''
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
    '''
    main function that runs the complete data pipeline
    tiles all images, runs augmentation, creates train/val/test splits,
    and saves everything to splits.json
    '''
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
    run the random crop augmentation to generate extra positive tiles
    this takes random 256x256 patches centered around waldo from the larger images
    each patch has waldo at a different position with different surrounding context
    we import it here so the augmented tiles get included in the same splits
    '''
    from data.augment import run_augmentation
    augmented_tiles = run_augmentation(tile_size=TILE_SIZES[0])
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


class WaldoTileDataset(Dataset):
    '''
    pytorch dataset that loads tiles from splits.json
    each tile returns three things:
      image tensor [3, 256, 256] normalized for the backbone
      label (0 = no waldo, 1 = waldo is in this tile)
      bbox [4] with waldos position, or [0,0,0,0] if no waldo
    '''

    def __init__(self, split="train", tile_size=256):
        '''
        read PROCESSED from the config module at construction time
        and store it as an instance attribute so DataLoader worker
        processes use the correct path even when config was overridden
        by ablation scripts before creating the dataset
        '''
        import config as _cfg
        self._processed_dir = _cfg.PROCESSED

        splits_path = os.path.join(self._processed_dir, "splits.json")
        with open(splits_path, "r") as f:
            all_splits = json.load(f)

        # only grab tiles of the requested size
        self.tiles = [t for t in all_splits[split] if t["tile_size"] == tile_size]
        self.tile_size = tile_size
        self.split = split # we need to know if we are training so we can apply augmentation

        # the base transform for val/test tiles (no augmentation)
        # resize + toTensor + normalize is all they get
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        '''
        color jitter for positive training tiles only
        this randomly tweaks brightness, contrast, and saturation each time
        so the model sees a slightly different waldo every epoch
        we only change colors not positions so the bounding box stays the same
        we dont apply this to val/test because we want clean untouched data for evaluation
        '''
        self.color_jitter = transforms.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CONTRAST,
            saturation=COLOR_JITTER_SATURATION
        )

        '''
        additional augmentations for ALL training tiles (not just positive)
        these create more visual variety so the model generalizes better
        inspired by augmentation techniques used in the FindingWaldo project
        (github.com/utsavraj/FindingWaldo) which used 6 different augmentations
        we apply these with random probability so not every tile gets every augmentation
        '''
        #
        '''
        we originally had random grayscale here too but took it out
        waldos whole thing is his red and white stripes
        our HSV mask channel is literally built to find red and white pixels
        if we convert to grayscale theres no red or white to find anymore
        so we were basically telling the model "learn to find waldo by color"
        and also "sometimes ignore all color information", which is contradictory
        '''
        self.blur = transforms.GaussianBlur(kernel_size=BLUR_KERNEL_SIZE, sigma=(0.1, 1.0))
        self.sharpness = transforms.RandomAdjustSharpness(sharpness_factor=SHARPNESS_FACTOR, p=SHARPNESS_PROBABILITY)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile_info = self.tiles[idx]

        img_path = os.path.join(self._processed_dir, str(self.tile_size), tile_info["file"])
        img = Image.open(img_path).convert("RGB")

        # get the bounding box (or zeros if no waldo)
        if tile_info["local_bbox"] is not None:
            bbox = torch.tensor(tile_info["local_bbox"], dtype=torch.float32)
        else:
            bbox = torch.zeros(4, dtype=torch.float32)

        label = tile_info["has_waldo"] # 0 or 1

        # for val and test we just resize, convert to tensor, and normalize
        # no augmentation at all so evaluation is on clean untouched data
        if self.split != "train" or not AUGMENT_TRAINING:
            img_tensor = self.transform(img)
            return img_tensor, label, bbox

        # everything below only applies to training tiles
        # resize first so all augmentations work on the same size image
        img = img.resize((INPUT_SIZE, INPUT_SIZE))

        '''
        horizontal flip with 50% chance
        we do this manually instead of using transforms.RandomHorizontalFlip
        because we also need to flip the bounding box x coordinates
        waldo looks the same facing left or right so this is a safe augmentation
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

        # color jitter on positive tiles only
        # changes brightness, contrast, saturation randomly
        if label == 1:
            img = self.color_jitter(img)

        # gaussian blur with 20% chance on all training tiles
        # makes the model robust to slightly blurry images
        if random.random() < BLUR_PROBABILITY:
            img = self.blur(img)

        # random sharpness adjustment on all training tiles
        img = self.sharpness(img)

        # convert to tensor and normalize with imagenet values
        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(img_tensor)

        return img_tensor, label, bbox


def get_dataloader(split="train", tile_size=256, batch_size=BATCH_SIZE, shuffle=True):
    '''
    main entry point for loading data throughout the project
    returns a pytorch dataloader that yields (images, labels, bboxes) batches

    args:
        split: "train", "val", or "test"
        tile_size: size of tiles to load (default 256)
        batch_size: number of tiles per batch
        shuffle: whether to randomize order (ignored for train, uses sampler)

    returns:
        DataLoader yielding (images [N,3,256,256], labels [N], bboxes [N,4])
    '''
    dataset = WaldoTileDataset(split=split, tile_size=tile_size)

    '''
    for training we use weighted sampling to handle the class imbalance
    we have about 14000 negative tiles but only 56 positive ones
    without this the model would almost never see a positive tile during training
    the sampler gives positive tiles a higher chance of being picked
    we compute the weight dynamically so it adjusts if the dataset changes
    '''
    if split == "train" and AUGMENT_TRAINING:
        num_positive = sum(1 for t in dataset.tiles if t["has_waldo"])
        num_negative = len(dataset.tiles) - num_positive

        '''
        we divide by SAMPLING_BALANCE_FACTOR so its not a full 50/50 split
        with factor 5 we get roughly 1 positive for every 5 negatives
        full 50/50 would make the model too eager and see waldo everywhere
        '''
        positive_weight = num_negative / num_positive / SAMPLING_BALANCE_FACTOR
        weights = [positive_weight if t["has_waldo"] else 1.0 for t in dataset.tiles]

        '''
        cap num_samples so each epoch doesnt take forever on large datasets
        with 12000+ tiles drawing every single one each epoch is wasteful
        the weighted sampler ensures we still see positive tiles frequently
        4000 samples per epoch is enough variety while keeping epochs fast
        (4000 / batch_size_8 = 500 batches per epoch instead of 1600+)
        '''
        epoch_samples = min(len(dataset), 4000)
        sampler = WeightedRandomSampler(weights, num_samples=epoch_samples)
        # cant use shuffle and sampler at the same time, the sampler handles randomness
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    elif split == "val":
        '''
        cap validation samples so we dont spend half the epoch on validation
        we just need the val loss direction for early stopping, not exact numbers
        1000 samples is enough to get a stable loss estimate
        '''
        val_samples = min(len(dataset), 1000)
        val_sampler = WeightedRandomSampler(
            [1.0] * len(dataset), num_samples=val_samples
        )
        loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    '''
    num_workers=4 loads tiles in parallel so the GPU never waits for data
    pin_memory=True speeds up CPU to GPU memory transfers
    persistent_workers=True keeps the worker processes alive between epochs
    so they dont have to restart every time (saves ~2s per epoch)
    '''
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
