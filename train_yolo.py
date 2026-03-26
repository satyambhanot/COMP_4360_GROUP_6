# train_yolo.py
# standalone YOLOv8 training script for comparison against our SSD/RetinaNet models
#
# this is NOT part of the main pipeline. its a separate experiment to see how
# a modern state of the art detector performs on the same data
# we use it in the report to compare our approach against the best available
#
# YOLOv8 is fundamentally different from our SSD/RetinaNet approach:
#   1. it processes FULL images (1280x1280) instead of small tiles
#   2. it uses its own training loop (ultralytics library handles everything)
#   3. it has built in mosaic augmentation which combines 4 images into 1
#   4. it uses an anchor free detection head instead of anchor based
#
# the big advantage is that YOLOv8 sees the entire scene at once
# while our models only see small 256x256 patches. this gives YOLO
# way more context to figure out which character is waldo
#
# usage:
#   pip install ultralytics
#   python train_yolo.py
#
# results get saved to results/yolo/ for comparison

import sys
import os
import json
import random
import shutil
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import (
    RAW_IMAGES, RAW_ANNOTATIONS,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
    PROJECT_ROOT
)

# yolo specific settings
YOLO_MODEL = "yolov8n.pt" # nano model, small and fast. good for comparison
YOLO_EPOCHS = 30 # yolov8 converges much faster than SSD, 30 is usually enough
YOLO_IMG_SIZE = 1280 # full image resolution, the main advantage over our tiling approach
YOLO_BATCH_SIZE = 4 # small batch because full images at 1280 use a lot of GPU memory
YOLO_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "yolo_format")
YOLO_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "yolo")


def convert_voc_to_yolo(xml_path, img_width, img_height):
    # converts a Pascal VOC XML annotation to YOLO format
    # YOLO format is: class_id center_x center_y width height
    # all values normalized to 0-1 relative to image dimensions
    #
    # our XML has absolute pixel coordinates:
    #   <xmin>706</xmin> <ymin>513</ymin> <xmax>743</xmax> <ymax>562</ymax>
    # YOLO wants normalized center coordinates:
    #   0 0.353 0.430 0.018 0.039
    # (class 0, center at 35.3% from left, 43% from top, 1.8% width, 3.9% height)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    obj = root.find("object")
    if obj is None:
        return None

    bbox = obj.find("bndbox")
    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)

    # convert to YOLO format (normalized center + dimensions)
    cx = (xmin + xmax) / 2.0 / img_width
    cy = (ymin + ymax) / 2.0 / img_height
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height

    # class 0 = waldo (YOLO uses 0-indexed classes)
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def prepare_yolo_dataset():
    # converts our Pascal VOC dataset to YOLO format and splits into train/val
    # YOLO expects this directory structure:
    #   data/yolo_format/
    #     train/
    #       images/  (jpg files)
    #       labels/  (txt files with same names as images)
    #     val/
    #       images/
    #       labels/
    #     dataset.yaml  (tells YOLO where to find everything)

    print("converting dataset to YOLO format...")

    # create directory structure
    for split in ["train", "val"]:
        os.makedirs(os.path.join(YOLO_DATA_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATA_DIR, split, "labels"), exist_ok=True)

    # get all image ids and split them the same way data_pipeline does
    # we use the same random seed so the split matches our SSD/RetinaNet experiments
    # this makes the comparison fair since both models see the same train/val data
    image_ids = []
    for f in sorted(os.listdir(RAW_ANNOTATIONS)):
        if f.endswith(".xml"):
            image_ids.append(f.replace(".xml", ""))

    random.seed(RANDOM_SEED)
    random.shuffle(image_ids)

    n = len(image_ids)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_ids = set(image_ids[:train_end])
    val_ids = set(image_ids[train_end:val_end])
    # test ids are the rest but YOLO only needs train and val for training
    # we evaluate on val since YOLO reports metrics on val automatically

    converted = 0
    skipped = 0

    for image_id in image_ids:
        img_path = os.path.join(RAW_IMAGES, f"{image_id}.jpg")
        xml_path = os.path.join(RAW_ANNOTATIONS, f"{image_id}.xml")

        if not os.path.exists(img_path):
            skipped += 1
            continue

        # figure out which split this image belongs to
        if image_id in train_ids:
            split = "train"
        elif image_id in val_ids:
            split = "val"
        else:
            continue # skip test images, YOLO trains on train+val only

        # get image dimensions from the XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        # convert annotation to YOLO format
        yolo_label = convert_voc_to_yolo(xml_path, img_w, img_h)
        if yolo_label is None:
            skipped += 1
            continue

        # copy image to YOLO directory
        dst_img = os.path.join(YOLO_DATA_DIR, split, "images", f"{image_id}.jpg")
        shutil.copy2(img_path, dst_img)

        # write YOLO label file (same name as image but .txt extension)
        dst_label = os.path.join(YOLO_DATA_DIR, split, "labels", f"{image_id}.txt")
        with open(dst_label, "w") as f:
            f.write(yolo_label + "\n")

        converted += 1

    # count splits
    train_count = len(os.listdir(os.path.join(YOLO_DATA_DIR, "train", "images")))
    val_count = len(os.listdir(os.path.join(YOLO_DATA_DIR, "val", "images")))

    print(f"  converted {converted} images, skipped {skipped}")
    print(f"  train: {train_count} images")
    print(f"  val: {val_count} images")

    # create dataset.yaml that YOLO needs
    yaml_content = f"""# YOLO dataset config for wheres waldo
# generated by train_yolo.py

path: {YOLO_DATA_DIR}
train: train/images
val: val/images

nc: 1
names:
  0: waldo
"""
    yaml_path = os.path.join(YOLO_DATA_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  saved dataset config to {yaml_path}")

    return yaml_path


def train_yolo(yaml_path):
    # trains YOLOv8 using the ultralytics library
    # this is the simplest way to use YOLO, the library handles everything:
    # data loading, augmentation (mosaic, mixup, HSV jitter), training loop,
    # validation, checkpointing, and metric computation

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. run: pip install ultralytics")
        print("then try again")
        return None

    print(f"\ntraining YOLOv8 ({YOLO_MODEL})...")
    print(f"  epochs: {YOLO_EPOCHS}")
    print(f"  image size: {YOLO_IMG_SIZE}")
    print(f"  batch size: {YOLO_BATCH_SIZE}")

    # load pretrained YOLOv8 nano model
    # nano is the smallest variant, good enough for comparison
    # and trains fast on a GTX 1080 Ti
    model = YOLO(YOLO_MODEL)

    # train on our waldo dataset
    # YOLO handles everything: augmentation, loss, optimizer, scheduling
    results = model.train(
        data=yaml_path,
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMG_SIZE,
        batch=YOLO_BATCH_SIZE,
        project=YOLO_RESULTS_DIR,
        name="train",
        exist_ok=True,
        verbose=True
    )

    return model


def evaluate_yolo(model):
    # runs evaluation on the validation set and saves results
    # in the same JSON format as our evaluate.py so we can compare

    if model is None:
        return

    print("\nevaluating YOLOv8 on validation set...")

    # YOLO computes metrics automatically during training
    # but we can also run explicit validation
    metrics = model.val()

    # extract the key metrics we care about
    # YOLO reports metrics differently than our evaluate.py
    # but we can map them to the same format
    results = {
        "model_type": "yolov8",
        "metrics": {
            "precision": float(metrics.box.mp) if hasattr(metrics.box, 'mp') else 0,
            "recall": float(metrics.box.mr) if hasattr(metrics.box, 'mr') else 0,
            "mAP50": float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0,
            "mAP50_95": float(metrics.box.map) if hasattr(metrics.box, 'map') else 0,
        },
        "training": {
            "model": YOLO_MODEL,
            "epochs": YOLO_EPOCHS,
            "img_size": YOLO_IMG_SIZE,
            "batch_size": YOLO_BATCH_SIZE,
        }
    }

    # compute F1 from precision and recall
    p = results["metrics"]["precision"]
    r = results["metrics"]["recall"]
    results["metrics"]["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0

    # save results
    os.makedirs(YOLO_RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(YOLO_RESULTS_DIR, "yolo_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nYOLOv8 results:")
    print(f"  precision: {results['metrics']['precision']:.4f}")
    print(f"  recall:    {results['metrics']['recall']:.4f}")
    print(f"  F1:        {results['metrics']['f1']:.4f}")
    print(f"  mAP@50:    {results['metrics']['mAP50']:.4f}")
    print(f"  mAP@50-95: {results['metrics']['mAP50_95']:.4f}")
    print(f"\nsaved results to {results_path}")

    return results


if __name__ == "__main__":
    print("=" * 50)
    print("  YOLOv8 Training (Comparison Experiment)")
    print("=" * 50)
    print()
    print("this is a standalone comparison against our SSD/RetinaNet approach")
    print("it uses the same training data but a completely different architecture")
    print()

    # step 1: convert our dataset to YOLO format
    yaml_path = prepare_yolo_dataset()

    # step 2: train YOLOv8
    model = train_yolo(yaml_path)

    # step 3: evaluate and save results
    if model is not None:
        evaluate_yolo(model)

    print("\ndone! check results/yolo/ for outputs")
