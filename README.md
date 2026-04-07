# Where's Waldo? — CNN Based Character Detection in Cluttered Scenes

**Course:** COMP 4360 Machine Learning, University of Manitoba
**Team:** Rudy Kissel, Satyam Bhanot, Gabriel Barnabe
**Instructor:** Dr. Christopher Henry
**Date:** March 2026


## Overview

This project adapts a strawberry detection system (Lamb & Chuah, IEEE Big Data 2018) to detect Waldo in "Where's Waldo?" illustrations. Each full puzzle image is chopped into 256×256 tiles, a preprocessing step highlights Waldo's red and white stripes using HSV color masking and Sobel edge detection, and a detection model classifies each tile. We compare four architectures and run five ablation studies to understand what actually matters.


## Results

### Model Comparison

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| **YOLOv8 Nano** | 89.8% | 66.2% | 0.762 | n/a |
| **Custom SSD MobileNetV2** | 93.5% | 45.3% | 0.610 | 85.9% |
| **RetinaNet ResNet50 FPN** | 98.7% | 41.4% | 0.583 | 85.6% |
| **Pretrained SSD300 VGG16** | 24.4% | 100% | 0.392 | 24.4% |

YOLOv8 leads overall because it sees the full image at once. Among tiled models, Custom SSD has the best F1. RetinaNet achieves the highest precision — only 3 false positives across 1770 negative tiles — but misses more Waldos. Pretrained SSD300 flagged almost every tile as Waldo.

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
| 256×256 | 1.000 | 0.038 | 0.073 |
| 128×128 | 0.000 | 0.000 | 0.000 |

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


## How to Run

### Quick Start — recommended for graders

If a trained checkpoint already exists in `checkpoints/`, skip training entirely:

```bash
bash scripts/demo.sh
```

Checks dependencies, tiles images if needed, loads the saved checkpoint, and runs full evaluation with all plots and metrics. Takes about 10 minutes.

### Full Pipeline — training from scratch

```bash
bash scripts/setup.sh    # install any missing dependencies
bash scripts/run.sh      # tile images, train, evaluate, generate plots
```

### Run All Experiments

Trains all 4 architectures, all 5 ablation studies, and generates the comparison report:

```bash
bash scripts/run_all_experiments.sh
```

Or run individually:

```bash
python experiments/run_retinanet.py
python experiments/run_custom_ssd.py
python experiments/run_pretrained_ssd.py
python experiments/run_yolo.py
python experiments/run_ablation.py           # preprocessing ablation
python experiments/run_ablation_heads.py     # SSD head count ablation
python experiments/run_ablation_tiles.py     # tile resolution ablation
python experiments/run_ablation_color.py     # color information ablation
python experiments/run_ablation_backbone.py  # backbone init ablation
python experiments/generate_report.py        # comparison charts from all results
```

### Other Commands

```bash
python src/scripts/test_image.py path/to/image.jpg   # detect Waldo in any image
python src/scripts/visualize_data.py                  # visualize preprocessing channels
python experiments/reeval_all.py                      # re-evaluate saved checkpoints
bash scripts/clean.sh                                  # reset generated files
bash scripts/clean.sh --all                            # also remove saved results
```


## Project Structure

```
COMP_4360_GROUP_6/
├── scripts/            # shell scripts for setup, training, demo, and cleanup
├── src/
│   ├── config.py       # all settings in one place
│   ├── train.py        # training loop (AMP, focal loss, early stopping)
│   ├── evaluate.py     # evaluation with plots and metrics
│   ├── models/         # WaldoDetector, SSD blocks, anchors, NMS
│   └── data/           # tiling, preprocessing, augmentation, dataset class
├── experiments/        # one script per model/ablation, plus report generator
├── data/
│   ├── raw/            # 458 images + Pascal VOC XML annotations
│   └── processed/      # generated tiles and splits (created at runtime)
├── checkpoints/        # saved model weights
└── saved_results/      # evaluation outputs organized by experiment
```


## Pipeline Summary

- **Data:** Raw images are tiled into 256×256 patches using 4 overlapping grid offsets so Waldo does not always fall on the same tile boundary. Split 70/15/15 by image ID to prevent data leakage.
- **Preprocessing:** Each tile gets two extra channels — an HSV mask isolating Waldo's red/white colors and Sobel edges computed on that mask. A learnable 1×1 conv merges 5 channels back to 3 for the backbone.
- **Models:** RetinaNet ResNet50+FPN and SSD300 VGG16 (both pretrained COCO, backbone frozen), a custom SSD built on MobileNetV2, and YOLOv8 Nano as a comparison.
- **Evaluation:** Tile-level classification metrics (precision, recall, F1, accuracy), confusion matrix, PR curve, ROC curve, confidence distribution, F1 vs threshold, and sample tiles showing TP/FP/FN detections with bounding boxes.


## Dataset

458 images total: 19 original Where's Waldo illustrations, 239 images from the FindingWaldo dataset (including brightness/gamma variants), and 200 synthetic images generated by pasting Waldo cutouts onto background scenes. All annotations are Pascal VOC XML bounding boxes. After tiling: roughly 570 positive tiles and 1770 negative tiles in the test set.


## Hardware and Requirements

Training was done on an NVIDIA GTX 1080 Ti (11GB VRAM). RetinaNet at 512×512 takes about 5–7 minutes per epoch. The pipeline also supports Apple Silicon (MPS) and CPU.

- Python 3.8+
- PyTorch >= 2.1.0 with CUDA recommended
- torchvision >= 0.16.0
- Pillow, matplotlib, ultralytics (YOLOv8)

```bash
bash scripts/setup.sh    # checks and installs only missing dependencies
```


## References

[1] N. Lamb and S. Chuah, "A Strawberry Detection System Using Convolutional Neural Networks," *2018 IEEE International Conference on Big Data*, pp. 2515–2520.

[2] vc1492a, "Hey-Waldo," GitHub. https://github.com/vc1492a/Hey-Waldo

[3] bckenstler, "TheresWaldo," GitHub. https://github.com/bckenstler/TheresWaldo

[4] tadejmagajna, "FindingWaldo," GitHub. https://github.com/tadejmagajna/FindingWaldo

[5] A. Howard et al., "Searching for MobileNetV3," *ICCV*, 2019, pp. 1314–1324.

[6] W. Liu et al., "SSD: Single Shot MultiBox Detector," *ECCV*, 2016, pp. 21–37.

[7] T. Lin et al., "Focal Loss for Dense Object Detection," *ICCV*, 2017, pp. 2980–2988.

[8] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLO," 2023. https://github.com/ultralytics/ultralytics


## License

University course project for COMP 4360, University of Manitoba. Not intended for redistribution. Where's Waldo images are copyrighted by Martin Handford / Walker Books.
