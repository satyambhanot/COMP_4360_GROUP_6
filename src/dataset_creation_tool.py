# dataset_creation_tool.py
# generates synthetic training images by pasting waldo cutouts onto random backgrounds
# gabe collected 10 waldo cutouts (transparent PNGs) and 100 background images
# this script combines them randomly to create new training data
#
# the idea is that our model struggles because it only has ~79 positive tiles
# from the real wheres waldo scenes. in those scenes waldo is always surrounded
# by hundreds of other characters and the model cant figure out what makes
# waldo different from everyone else
#
# by pasting waldo onto varied backgrounds we give the model clean examples
# of what waldo looks like in isolation, on simple backgrounds, and on
# complex backgrounds. this teaches the visual pattern of waldo specifically
# (red/white stripes, glasses, hat) without the confusion of the cluttered scenes
#
# output goes directly into data/raw/images/ and data/raw/annotations/
# so the existing data_pipeline.py picks them up automatically during tiling
# files are named gen_001.jpg, gen_002.jpg etc to avoid collision with
# existing image files (1.jpg through 124G.jpg)
#
# run this once, commit the results, then everyone has the same dataset
# usage: python src/dataset_creation_tool.py

import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image, ImageDraw
import random
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import RANDOM_SEED

# how many synthetic images to generate
# each image is one waldo cutout pasted on one random background
# with 10 waldo poses and 100 backgrounds we have 1000 possible combinations
# 200 gives us good variety without being excessive
NUM_GENERATED_IMAGES = 200

# waldo gets scaled to roughly this fraction of the background image
# 0.1 means waldo takes up about 10% of the image which is realistic
# for how big waldo appears in the actual puzzle illustrations
WALDO_BASE_RATIO = 0.1

# random scaling variation on top of the base ratio
# 0.7 to 1.3 means waldo can be 70% to 130% of the base size
# this teaches the model to detect waldo at slightly different scales
WALDO_SCALE_MIN = 0.7
WALDO_SCALE_MAX = 1.3

# valid image extensions for background files
# we skip anything that isnt an image (like .DS_Store or .gitkeep)
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BG_PATH = os.path.join(PROJECT_ROOT, "data", "data_creation", "backgrounds")
WALDO_PATH = os.path.join(PROJECT_ROOT, "data", "data_creation", "waldos")
OUTPUT_IMAGES = os.path.join(PROJECT_ROOT, "data", "raw", "images")
OUTPUT_ANNOTATIONS = os.path.join(PROJECT_ROOT, "data", "raw", "annotations")
VERIFICATION_PATH = os.path.join(PROJECT_ROOT, "data", "data_creation")


def get_image_files(directory):
    # returns a sorted list of image files in a directory
    # filters out non image files like .DS_Store, .gitkeep, thumbs.db etc
    # sorting makes the output deterministic regardless of OS file listing order
    files = []
    for f in sorted(os.listdir(directory)):
        ext = os.path.splitext(f)[1].lower()
        if ext in VALID_EXTENSIONS:
            files.append(f)
    return files


def scale_waldo_to_background(background_img, waldo_img):
    # scales waldo so hes roughly WALDO_BASE_RATIO of the background size
    # then applies a random scaling factor for variety
    # returns the resized waldo image
    bg_w, bg_h = background_img.size
    fg_w, fg_h = waldo_img.size

    # scale based on whichever dimension is proportionally larger
    # so waldo always fits within the target ratio
    if fg_w / bg_w > fg_h / bg_h:
        new_w = int(bg_w * WALDO_BASE_RATIO)
        new_h = int(new_w * (fg_h / fg_w))
    else:
        new_h = int(bg_h * WALDO_BASE_RATIO)
        new_w = int(new_h * (fg_w / fg_h))

    waldo_img = waldo_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # apply random scaling variation
    scale = random.uniform(WALDO_SCALE_MIN, WALDO_SCALE_MAX)
    new_w = int(new_w * scale)
    new_h = int(new_h * scale)

    # make sure waldo is at least 10x10 pixels (very small images cause issues)
    new_w = max(new_w, 10)
    new_h = max(new_h, 10)

    return waldo_img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def create_annotation_xml(image_id, img_width, img_height, xmin, ymin, xmax, ymax):
    # creates a Pascal VOC XML annotation string
    # matches the exact format our data_pipeline.py expects:
    #   <annotation>
    #     <size><width> <height></size>
    #     <object><bndbox><xmin> <ymin> <xmax> <ymax></bndbox></object>
    #   </annotation>

    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = f"{image_id}.jpg"
    ET.SubElement(annotation, "path").text = os.path.join(OUTPUT_IMAGES, f"{image_id}.jpg")

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Generated"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(annotation, "segmented").text = "0"

    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = "waldo"
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"

    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(xmin)
    ET.SubElement(bndbox, "ymin").text = str(ymin)
    ET.SubElement(bndbox, "xmax").text = str(xmax)
    ET.SubElement(bndbox, "ymax").text = str(ymax)

    # pretty print the xml to match the format of our existing annotations
    xml_str = ET.tostring(annotation, encoding='utf-8')
    xml_pretty = minidom.parseString(xml_str).toprettyxml(indent="\t")
    # remove the <?xml version...?> header line to match existing files
    xml_pretty = "\n".join(xml_pretty.split("\n")[1:])

    return xml_pretty


def generate_images():
    # main function that generates all the synthetic training images
    # each image is a random background with a random waldo cutout pasted on it

    random.seed(RANDOM_SEED)

    # load the file lists
    bg_files = get_image_files(BG_PATH)
    waldo_files = get_image_files(WALDO_PATH)

    if len(bg_files) == 0:
        print(f"no background images found in {BG_PATH}")
        return []
    if len(waldo_files) == 0:
        print(f"no waldo cutouts found in {WALDO_PATH}")
        return []

    print(f"found {len(bg_files)} background images and {len(waldo_files)} waldo cutouts")
    print(f"generating {NUM_GENERATED_IMAGES} synthetic training images...")

    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    os.makedirs(OUTPUT_ANNOTATIONS, exist_ok=True)

    generated = [] # track what we created for verification
    errors = 0

    for i in range(NUM_GENERATED_IMAGES):
        image_id = f"gen_{i:03d}"

        try:
            # pick a random background and convert to RGB
            # some backgrounds are PNG with alpha or webp, converting to RGB
            # makes sure we always get a clean 3 channel image for saving as JPG
            bg_file = random.choice(bg_files)
            bg = Image.open(os.path.join(BG_PATH, bg_file)).convert("RGB")
            bg_w, bg_h = bg.size

            # pick a random waldo cutout and keep as RGBA for the transparency mask
            # the waldo PNGs have transparent backgrounds so when we paste
            # only waldo himself shows up, not a white rectangle around him
            waldo_file = random.choice(waldo_files)
            waldo = Image.open(os.path.join(WALDO_PATH, waldo_file)).convert("RGBA")

            # scale waldo to fit the background naturally
            waldo = scale_waldo_to_background(bg, waldo)
            waldo_w, waldo_h = waldo.size

            # make sure waldo actually fits on this background
            # if the background is tiny and waldo is still too big after scaling
            # just skip this one and try the next
            if waldo_w >= bg_w or waldo_h >= bg_h:
                print(f"  skipping {image_id}: waldo ({waldo_w}x{waldo_h}) doesnt fit on background ({bg_w}x{bg_h})")
                errors += 1
                continue

            # pick a random position for waldo
            waldo_x = random.randint(0, bg_w - waldo_w)
            waldo_y = random.randint(0, bg_h - waldo_h)

            # paste waldo onto the background using the alpha channel as mask
            # this means only waldos pixels get pasted, not the transparent area
            bg.paste(waldo, (waldo_x, waldo_y), mask=waldo)

            # compute the bounding box (where waldo ended up)
            xmin = waldo_x
            ymin = waldo_y
            xmax = waldo_x + waldo_w
            ymax = waldo_y + waldo_h

            # save the composite image as JPG
            img_path = os.path.join(OUTPUT_IMAGES, f"{image_id}.jpg")
            bg.save(img_path, "JPEG", quality=95)

            # create and save the XML annotation
            xml_content = create_annotation_xml(image_id, bg_w, bg_h, xmin, ymin, xmax, ymax)
            xml_path = os.path.join(OUTPUT_ANNOTATIONS, f"{image_id}.xml")
            with open(xml_path, "w") as f:
                f.write(xml_content)

            generated.append({
                "image_id": image_id,
                "img_path": img_path,
                "xml_path": xml_path,
                "bg_file": bg_file,
                "waldo_file": waldo_file,
                "bbox": [xmin, ymin, xmax, ymax],
                "img_size": [bg_w, bg_h]
            })

        except Exception as e:
            print(f"  error generating {image_id}: {e}")
            errors += 1

    print(f"\ngeneration complete:")
    print(f"  successfully created: {len(generated)} images")
    print(f"  errors/skipped: {errors}")

    return generated


def verify_generated(generated):
    # double checks that every generated image has a matching XML
    # and that the bounding box is within the image bounds
    # this catches any bugs in the generation process before we commit

    print(f"\nverifying {len(generated)} generated images...")
    issues = 0

    for item in generated:
        image_id = item["image_id"]

        # check image file exists
        img_path = os.path.join(OUTPUT_IMAGES, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            print(f"  MISSING IMAGE: {img_path}")
            issues += 1
            continue

        # check XML file exists
        xml_path = os.path.join(OUTPUT_ANNOTATIONS, f"{image_id}.xml")
        if not os.path.exists(xml_path):
            print(f"  MISSING XML: {xml_path}")
            issues += 1
            continue

        # parse the XML and verify bbox is within image bounds
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        xml_w = int(size.find("width").text)
        xml_h = int(size.find("height").text)

        bbox = root.find("object/bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # verify bbox is within image bounds
        if xmin < 0 or ymin < 0 or xmax > xml_w or ymax > xml_h:
            print(f"  BBOX OUT OF BOUNDS: {image_id} bbox=({xmin},{ymin},{xmax},{ymax}) image=({xml_w}x{xml_h})")
            issues += 1

        # verify bbox makes sense (xmax > xmin, ymax > ymin)
        if xmax <= xmin or ymax <= ymin:
            print(f"  INVALID BBOX: {image_id} bbox=({xmin},{ymin},{xmax},{ymax})")
            issues += 1

    if issues == 0:
        print("  all good, no issues found")
    else:
        print(f"  found {issues} issues!")

    return issues


def save_verification_grid(generated):
    # saves a 3x3 grid of sample images with bounding boxes drawn on them
    # gabe can open this one image and instantly see if the boxes are correct
    # much faster than checking 200 individual images

    if len(generated) < 9:
        print("not enough images to create verification grid")
        return

    # pick 9 random samples
    samples = random.sample(generated, 9)

    # figure out a good cell size for the grid
    # we resize each sample to 400x300 so the grid is a reasonable size
    cell_w, cell_h = 400, 300
    grid = Image.new("RGB", (cell_w * 3, cell_h * 3), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for idx, item in enumerate(samples):
        row = idx // 3
        col = idx % 3

        # open the generated image
        img = Image.open(item["img_path"]).convert("RGB")
        img_w, img_h = img.size

        # compute scale to fit in the cell
        scale_x = cell_w / img_w
        scale_y = cell_h / img_h
        scale = min(scale_x, scale_y)

        # resize the image to fit the cell
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # paste into the grid
        offset_x = col * cell_w + (cell_w - new_w) // 2
        offset_y = row * cell_h + (cell_h - new_h) // 2
        grid.paste(img_resized, (offset_x, offset_y))

        # draw the bounding box scaled to match
        xmin, ymin, xmax, ymax = item["bbox"]
        box_x1 = offset_x + int(xmin * scale)
        box_y1 = offset_y + int(ymin * scale)
        box_x2 = offset_x + int(xmax * scale)
        box_y2 = offset_y + int(ymax * scale)
        draw.rectangle([box_x1, box_y1, box_x2, box_y2], outline="green", width=3)

        # label with the image id
        draw.text((offset_x + 5, offset_y + 5), item["image_id"], fill="green")

    # save the grid
    grid_path = os.path.join(VERIFICATION_PATH, "verification_grid.jpg")
    grid.save(grid_path, "JPEG", quality=95)
    print(f"\nsaved verification grid to {grid_path}")
    print("open this image to visually check that waldo is inside the green boxes")


if __name__ == "__main__":
    print("=" * 50)
    print("  Waldo Dataset Creation Tool")
    print("=" * 50)
    print()

    generated = generate_images()

    if len(generated) > 0:
        issues = verify_generated(generated)
        save_verification_grid(generated)

        print(f"\ndone! {len(generated)} images saved to:")
        print(f"  images: {OUTPUT_IMAGES}/gen_*.jpg")
        print(f"  labels: {OUTPUT_ANNOTATIONS}/gen_*.xml")

        if issues == 0:
            print("\neverything looks good. you can commit these files now.")
        else:
            print(f"\nWARNING: {issues} issues found. check the output above before committing.")
