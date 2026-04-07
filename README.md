# Where's Waldo? - CNN Based Character Detection in Cluttered Scenes

**Course:** COMP 4360 Machine Learning, University of Manitoba
**Team:** Rudy Kissel, Satyam Bhanot, Gabriel Barnabe
**Instructor:** Dr. Christopher Henry
**Date:** March 2026


## Quick Setup

```bash
bash scripts/setup.sh
```

This checks your environment and installs any missing dependencies.

Requirements: Python 3.8+, PyTorch >= 2.1.0 (CUDA recommended), torchvision >= 0.16.0, Pillow, matplotlib, ultralytics


## Quick Run

If a checkpoint is already saved in `checkpoints/`, you can skip training and just see the results:

```bash
bash scripts/demo.sh
```

This loads the saved model, runs evaluation, and saves all plots and metrics to `results/`. Takes around 10 minutes.

To train from scratch:

```bash
bash scripts/run.sh
```

This tiles the raw images, trains the model, evaluates it, and saves everything.


## What This Project Does

We wanted to see if a CNN could find Waldo in those busy puzzle illustrations. The approach is based on a strawberry detection paper (Lamb & Chuah, IEEE Big Data 2018) that we adapted for this problem.

The idea is to split each full image into 256x256 tiles, then run a preprocessing step that uses HSV color masking to highlight Waldo's red and white stripes and Sobel edge detection to pick up the stripe pattern. A detection model then looks at each tile and decides if Waldo is there. We tested four different model architectures and ran five ablation studies to figure out which design choices actually helped.


## Results

### Model Comparison

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| YOLOv8 Nano | 89.8% | 66.2% | 0.762 | n/a |
| Custom SSD MobileNetV2 | 93.5% | 45.3% | 0.610 | 85.9% |
| RetinaNet ResNet50 FPN | 98.7% | 41.4% | 0.583 | 85.6% |
| Pretrained SSD300 VGG16 | 24.4% | 100% | 0.392 | 24.4% |

YOLOv8 does best overall since it looks at the whole image rather than tiles. Among our tiled models, Custom SSD has the best F1. RetinaNet is extremely precise (only 3 false positives across 1770 negative tiles) but it misses a lot of Waldos. The pretrained SSD300 basically called everything Waldo.

### Ablation Studies

**Preprocessing Channels**

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| RGB + HSV + Sobel (full pipeline) | 0.975 | 0.207 | 0.342 |
| RGB + HSV only | 0.000 | 0.000 | 0.000 |
| RGB only | 1.000 | 0.037 | 0.071 |

**SSD Detection Heads**

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| 5 heads | 0.167 | 0.002 | 0.003 |
| 4 heads | 0.758 | 0.121 | 0.209 |
| 3 heads | 0.643 | 0.063 | 0.115 |

**Tile Resolution**

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| 256x256 | 1.000 | 0.038 | 0.073 |
| 128x128 | 0.000 | 0.000 | 0.000 |

**Color Information**

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| RGB color | 0.985 | 0.114 | 0.204 |
| Grayscale | 0.978 | 0.077 | 0.143 |
| Binary B/W | 0.000 | 0.000 | 0.000 |

**Backbone Initialization**

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| Pretrained ImageNet | 0.409 | 0.016 | 0.030 |
| Random init | 0.000 | 0.000 | 0.000 |


## Running Experiments

To train all 4 models, run all 5 ablation studies, and generate the comparison report:

```bash
bash scripts/run_all_experiments.sh
```

Or run each one separately:

```bash
python experiments/run_retinanet.py
python experiments/run_custom_ssd.py
python experiments/run_pretrained_ssd.py
python experiments/run_yolo.py
python experiments/run_ablation.py           # preprocessing ablation
python experiments/run_ablation_heads.py     # number of SSD heads
python experiments/run_ablation_tiles.py     # tile resolution
python experiments/run_ablation_color.py     # color vs grayscale vs binary
python experiments/run_ablation_backbone.py  # pretrained vs random init
python experiments/generate_report.py        # charts from all saved results
```

Other useful scripts:

```bash
python src/scripts/test_image.py path/to/image.jpg   # run detection on any image
python src/scripts/visualize_data.py                  # see what the preprocessing looks like
python experiments/reeval_all.py                      # re-run evaluation on saved checkpoints
bash scripts/clean.sh                                  # delete generated files
bash scripts/clean.sh --all                            # also delete saved experiment results
```


## Project Structure

```
COMP_4360_GROUP_6/
├── scripts/            # setup, run, demo, and clean shell scripts
├── src/
│   ├── config.py       # all settings live here
│   ├── train.py        # training loop
│   ├── evaluate.py     # evaluation and plot generation
│   ├── models/         # model definitions, SSD blocks, anchors, NMS
│   └── data/           # tiling, preprocessing, augmentation, dataset class
├── experiments/        # one script per model and ablation study
├── data/
│   ├── raw/            # 458 images with Pascal VOC XML annotations
│   └── processed/      # tiles and splits, created at runtime
├── checkpoints/        # saved model weights
└── saved_results/      # outputs from each experiment
```


## How the Pipeline Works

**Data:** Each image gets split into 256x256 tiles using 4 different grid offsets (shifted by 128px horizontally and vertically). This means Waldo ends up at different positions across tiles rather than always near a boundary. Train/val/test split is 70/15/15 by image ID so tiles from the same source image never show up in both training and test.

**Preprocessing:** Each tile gets two extra channels added to the standard RGB. One is an HSV mask that highlights Waldo's red and white color range. The other is Sobel edge detection run on the HSV mask (not the raw image), which picks up the stripe edges specifically. A 1x1 conv layer at the start of the model folds all 5 channels back to 3 for the backbone.

**Models:** We use RetinaNet with ResNet50+FPN, SSD300 with VGG16 (both pretrained on COCO with frozen backbones), a custom SSD we built on top of MobileNetV2, and YOLOv8 Nano as a comparison point.

**Evaluation:** Outputs precision, recall, F1, accuracy, confusion matrix, PR curve, ROC curve, confidence distribution, F1 vs threshold plot, and saved tile images showing true positives, false positives, and false negatives with bounding boxes.


## Dataset

We used 458 images total. 19 are original Where's Waldo illustrations we annotated by hand. 239 come from the FindingWaldo dataset including some brightness and gamma variants. The remaining 200 are synthetic images we made by pasting Waldo cutouts onto background images at random positions. All annotations are bounding boxes in Pascal VOC XML format. The test set has roughly 570 positive tiles and 1770 negative tiles.


## Hardware

We trained on an NVIDIA GTX 1080 Ti (11GB VRAM). RetinaNet at 512x512 takes about 5 to 7 minutes per epoch. The code also works on Apple Silicon (MPS) and CPU but GPU is strongly recommended.


## References

[1] N. Lamb and S. Chuah, "A Strawberry Detection System Using Convolutional Neural Networks," *2018 IEEE International Conference on Big Data*, pp. 2515-2520.

[2] vc1492a, "Hey-Waldo," GitHub. https://github.com/vc1492a/Hey-Waldo

[3] bckenstler, "TheresWaldo," GitHub. https://github.com/bckenstler/TheresWaldo

[4] tadejmagajna, "FindingWaldo," GitHub. https://github.com/tadejmagajna/FindingWaldo

[5] A. Howard et al., "Searching for MobileNetV3," *ICCV*, 2019, pp. 1314-1324.

[6] W. Liu et al., "SSD: Single Shot MultiBox Detector," *ECCV*, 2016, pp. 21-37.

[7] T. Lin et al., "Focal Loss for Dense Object Detection," *ICCV*, 2017, pp. 2980-2988.

[8] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLO," 2023. https://github.com/ultralytics/ultralytics


## License

Course project for COMP 4360 at the University of Manitoba. Not for redistribution. Where's Waldo images are copyrighted by Martin Handford / Walker Books.
