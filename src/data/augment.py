'''
augment.py
generates extra positive tiles by randomly cropping 256x256 patches around
waldo in the source images. instead of only getting tiles from a fixed grid
(where waldo always lands in the same 1-2 tiles), we take random patches
with waldo at different positions giving us more diverse training examples.

each crop also gets flipped horizontally to double the count since waldo
looks the same facing left or right. only runs on images big enough to
crop from (512px minimum). called by data_pipeline.py during tiling.
'''

import os
import random
from PIL import Image
import xml.etree.ElementTree as ET

from config import (
    RAW_IMAGES, RAW_ANNOTATIONS, PROCESSED,
    INPUT_SIZE, OVERLAP_THRESHOLD, RANDOM_SEED
)


'''
how many random crops to take per image
more crops = more positive tiles but also more disk space and tiling time
15 is a good balance, gives us roughly 15 * number_of_large_images new positives
'''
CROPS_PER_IMAGE = 15

# minimum image size to bother cropping from
# if the image is barely bigger than 256x256 the crops will all look the same
MIN_IMAGE_SIZE = 512


def parse_annotation(xml_path):
    '''
    reads one xml annotation file and extracts image dimensions and bbox
    same logic as data_pipeline.py version

    args:
        xml_path: path to the Pascal VOC xml file

    returns:
        img_w, img_h, [xmin, ymin, xmax, ymax]
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    obj = root.find("object")
    bbox = obj.find("bndbox")
    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)

    return img_w, img_h, [xmin, ymin, xmax, ymax]


def waldo_overlap(waldo_box, tile_box):
    '''
    computes what fraction of waldo is inside a crop
    same function as data_pipeline.py so the overlap check is consistent

    args:
        waldo_box: [xmin, ymin, xmax, ymax]
        tile_box: [xmin, ymin, xmax, ymax]

    returns:
        overlap fraction (0.0 to 1.0)
    '''
    xA = max(waldo_box[0], tile_box[0])
    yA = max(waldo_box[1], tile_box[1])
    xB = min(waldo_box[2], tile_box[2])
    yB = min(waldo_box[3], tile_box[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    waldo_area = (waldo_box[2] - waldo_box[0]) * (waldo_box[3] - waldo_box[1])
    if waldo_area == 0:
        return 0.0
    return inter / waldo_area


def generate_random_crops(image_id, img, img_w, img_h, waldo_bbox, tile_size=INPUT_SIZE):
    '''
    generates random 256x256 crops around waldo with him at different positions
    each crop is guaranteed to contain at least 50% of waldo

    args:
        image_id: string id of the source image
        img: PIL Image object
        img_w: image width
        img_h: image height
        waldo_bbox: [xmin, ymin, xmax, ymax] waldo bounding box
        tile_size: size of each crop (default 256)

    returns:
        list of tile info dicts (same format as data_pipeline tiles)
    '''
    #
    '''
    the key insight is that waldo can be anywhere within the crop
    not just in the center. this gives us crops where waldo is in the
    top left, bottom right, partially cut off at an edge, etc.
    each of these is a genuinely different training example
    '''

    tile_dir = os.path.join(PROCESSED, str(tile_size))
    os.makedirs(tile_dir, exist_ok=True)

    tiles_info = []
    wx1, wy1, wx2, wy2 = waldo_bbox

    '''
    figure out the valid range for crop top-left corner
    the crop must overlap waldo so we constrain the range
    if waldo is at (100, 200) and tile is 256, the crop x can be
    anywhere from (100 - 256) to (100 + waldo_width) roughly
    but also must stay within image bounds
    '''
    x_min = max(0, wx2 - tile_size) # crop must start before waldo ends
    x_max = min(img_w - tile_size, wx1) # crop must end after waldo starts
    y_min = max(0, wy2 - tile_size)
    y_max = min(img_h - tile_size, wy1)

    # if the valid range is empty the image is too small or waldo is at the edge
    if x_min > x_max or y_min > y_max:
        return tiles_info

    attempts = 0
    max_attempts = CROPS_PER_IMAGE * 5 # give ourselves some room for rejected crops

    while len(tiles_info) < CROPS_PER_IMAGE and attempts < max_attempts:
        attempts += 1

        # pick a random crop position
        cx = random.randint(x_min, x_max)
        cy = random.randint(y_min, y_max)

        crop_box = [cx, cy, cx + tile_size, cy + tile_size]

        # check that waldo overlaps enough
        overlap = waldo_overlap(waldo_bbox, crop_box)
        if overlap < OVERLAP_THRESHOLD:
            continue

        # compute waldos position relative to this crop
        local_bbox = [
            max(wx1 - cx, 0),
            max(wy1 - cy, 0),
            min(wx2 - cx, tile_size),
            min(wy2 - cy, tile_size)
        ]

        # crop and save
        tile_img = img.crop(crop_box)
        tile_name = f"img{image_id}_{tile_size}_rand{len(tiles_info)}.jpg"
        tile_img.save(os.path.join(tile_dir, tile_name))

        tiles_info.append({
            "image_id": image_id,
            "tile_size": tile_size,
            "file": tile_name,
            "x": cx,
            "y": cy,
            "has_waldo": 1,
            "local_bbox": local_bbox
        })

        '''
        also save a horizontally flipped version
        waldo looks the same facing either direction
        and the flipped crop has different pixel patterns around him
        '''
        flipped_img = tile_img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_name = f"img{image_id}_{tile_size}_rand{len(tiles_info)}_flip.jpg"
        flipped_img.save(os.path.join(tile_dir, flip_name))

        # flip the bounding box x coordinates
        flipped_bbox = [
            tile_size - local_bbox[2],
            local_bbox[1],
            tile_size - local_bbox[0],
            local_bbox[3]
        ]

        tiles_info.append({
            "image_id": image_id,
            "tile_size": tile_size,
            "file": flip_name,
            "x": cx,
            "y": cy,
            "has_waldo": 1,
            "local_bbox": flipped_bbox
        })

    return tiles_info


def run_augmentation(tile_size=INPUT_SIZE):
    '''
    main function that generates random crop augmentations for all images
    called by data_pipeline.py during the tiling process

    args:
        tile_size: size of each crop, should match the tiling size (default 256)

    returns:
        list of tile info dicts for all generated crops
    '''

    random.seed(RANDOM_SEED)

    print(f"generating random crop augmentations (tile_size={tile_size})...")

    # find all image ids that have annotations
    all_ids = []
    for f in sorted(os.listdir(RAW_ANNOTATIONS)):
        if f.endswith(".xml"):
            all_ids.append(f.replace(".xml", ""))

    all_tiles = []
    skipped = 0

    for image_id in all_ids:
        img_path = os.path.join(RAW_IMAGES, f"{image_id}.jpg")
        xml_path = os.path.join(RAW_ANNOTATIONS, f"{image_id}.xml")

        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        '''
        skip images that are too small for meaningful random crops
        the tiny findingwaldo crops are like 50x50 pixels
        theres no point trying to take 256x256 random crops from those
        '''
        if img_w < MIN_IMAGE_SIZE or img_h < MIN_IMAGE_SIZE:
            skipped += 1
            continue

        _, _, waldo_bbox = parse_annotation(xml_path)

        tiles = generate_random_crops(image_id, img, img_w, img_h, waldo_bbox, tile_size=tile_size)
        if tiles:
            print(f"  image {image_id}: {len(tiles)} random crops")
            all_tiles.extend(tiles)

    print(f"\naugmentation complete:")
    print(f"  images processed: {len(all_ids) - skipped}")
    print(f"  images skipped (too small): {skipped}")
    print(f"  new positive tiles: {len(all_tiles)}")

    return all_tiles


