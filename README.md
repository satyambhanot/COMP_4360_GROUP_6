# Where's Waldo?
# CNN Based Character Detection in Cluttered Scenes

**Course:** COMP 4360 Machine Learning, University of Manitoba
**Team:** Rudy Kissel, Satyam Bhanot, Gabriel Barnabe
**Instructor:** Dr. Christopher Henry
**Date:** March 2026


## What This Project Does

Finding Waldo in those crowded puzzle pages is hard enough for humans, so we thought it would be interesting to see if a CNN could learn to do it too. This project takes a strawberry detection system (Lamb & Chuah, IEEE Big Data 2018) and adapts it for detecting Waldo in "Where's Waldo?" illustrations.

The core idea is straightforward: chop each large puzzle image into small tiles, run a preprocessing step that highlights Waldo's red and white stripes using HSV color masking and Sobel edge detection, then feed those tiles through a detection model. We started with a custom SSD built from scratch, moved to a pretrained SSD300, and eventually landed on RetinaNet as our best performing model. We also trained YOLOv8 as a state of the art comparison point.


## Results

We tested four different architectures on the same dataset and train/val/test split:

| Model | Precision | Recall | F1 Score | Notes |
|---|---|---|---|---|
| RetinaNet ResNet50 FPN | 97.3% | 38.7% | 0.55 | Only 7 false positives across 3287 tiles |
| Pretrained SSD300 VGG16 | 20.9% | 100% | 0.35 | Detected everything as Waldo |
| Custom SSD MobileNetV2 | 76.5% | 27.8% | 0.41 | Trained from scratch, limited by data |
| YOLOv8 Nano (comparison) | 93.5% | 85.3% | 0.89 | Full image at 1280x1280, not tiled |

YOLOv8 outperforms our tiled approach because it sees the entire scene at once, giving it much more context to work with. But our RetinaNet model achieves very high precision, meaning when it says it found Waldo, it is almost always right.

The ablation study on preprocessing channels showed:

| Variant | F1 Score |
|---|---|
| RGB + HSV + Sobel (full pipeline) | 0.085 |
| RGB + HSV only | 0.251 |
| RGB only | 0.213 |


## Project Structure

```
COMP_4360_GROUP_6/
├── README.md
├── .gitignore
├── requirements.txt                    # pinned dependency versions
├── setup.sh                            # checks and installs missing dependencies
├── run.sh                              # runs the full pipeline end to end
├── demo.sh                             # quick eval using existing checkpoint (no training)
├── run_all_experiments.sh              # trains all models + ablation + comparison report
├── clean.sh                            # resets generated files for a fresh run
│
├── src/
│   ├── config.py                       # all settings in one place
│   ├── utils.py                        # shared helpers (device selection, target formatting)
│   ├── train.py                        # training loop with AMP, early stopping, focal loss
│   ├── evaluate.py                     # evaluation with 10+ output plots
│   ├── train_yolo.py                   # standalone YOLOv8 comparison script
│   │
│   ├── models/
│   │   ├── model.py                    # WaldoDetector, WaldoDetectorPretrained, WaldoDetectorRetinaNet
│   │   ├── ssd_block.py               # single SSD detection block for the custom model
│   │   ├── anchors.py                 # anchor generation and prediction reshaping
│   │   └── nms.py                     # non maximum suppression
│   │
│   ├── data/
│   │   ├── data_pipeline.py           # tiling, dataset class, dataloader creation
│   │   ├── pre_processing.py          # HSV mask, Sobel edges, batch preprocessing
│   │   ├── augment.py                 # random crop augmentation around Waldo
│   │   └── paste_waldo.py             # synthetic Waldo on background image generator
│   │
│   └── scripts/
│       ├── print_summary.py           # model summary and results printing
│       ├── sanity_check.py            # post training confidence check
│       ├── visualize_data.py          # preprocessing channel visualization
│       └── test_image.py              # detect Waldo in any image
│
├── experiments/
│   ├── base_experiment.py             # shared experiment template class
│   ├── run_retinanet.py               # trains and evaluates RetinaNet
│   ├── run_pretrained_ssd.py          # trains and evaluates SSD300 VGG16
│   ├── run_custom_ssd.py              # trains and evaluates custom SSD
│   ├── run_yolo.py                    # trains and evaluates YOLOv8
│   ├── run_ablation.py                # preprocessing ablation controller (3 variants)
│   ├── run_ablation_variant.py        # single ablation variant (subprocess)
│   ├── reeval_all.py                  # re evaluate all saved experiments
│   ├── reeval_ablation_variant.py     # re evaluate a single ablation variant
│   └── generate_report.py             # comparison charts from all experiment results
│
├── data/
│   ├── raw/
│   │   ├── images/                    # 458 images (258 original + 200 synthetic)
│   │   └── annotations/               # 458 Pascal VOC XML bounding box annotations
│   ├── assets/
│   │   ├── waldos/                    # 10 Waldo cutout PNGs with transparent backgrounds
│   │   └── backgrounds/              # 100 background images for synthetic generation
│   └── processed/                     # generated tiles and splits (created at runtime)
│
├── checkpoints/                       # saved model weights and training history
├── results/                           # evaluation outputs, plots, and run logs
└── saved_results/                     # experiment outputs organized by model
```


## How to Run

### Quick Start (recommended for graders)

If a trained checkpoint already exists (`checkpoints/best_model.pth`), you can see results without waiting for training:

```bash
bash demo.sh
```

This checks dependencies, generates tiles if needed, loads the saved model, and runs the full evaluation with all plots and metrics. Takes about 10 minutes instead of hours.

### Full Pipeline (training + evaluation)

```bash
bash setup.sh       # install dependencies (only missing ones)
bash run.sh         # tile images, train model, evaluate, generate plots
```

The complete output gets saved to `results/run_log.txt`.

### Run All Experiments

To train all 4 model architectures, run the ablation study, and generate a comparison report:

```bash
bash run_all_experiments.sh
```

Results are saved to `saved_results/<model_name>/` for each experiment. Or run experiments individually:

```bash
python experiments/run_retinanet.py
python experiments/run_pretrained_ssd.py
python experiments/run_custom_ssd.py
python experiments/run_yolo.py
python experiments/run_ablation.py
python experiments/generate_report.py    # comparison charts from all results
```

### Other Useful Commands

```bash
python src/scripts/test_image.py path/to/image.jpg   # detect Waldo in any image
python src/scripts/visualize_data.py                  # see preprocessing channels (HSV, Sobel)
bash clean.sh                                         # reset generated files for a fresh run
bash clean.sh --all                                   # also remove saved experiment results
```

### Run Steps Individually

```bash
bash setup.sh                           # install dependencies
python src/data/data_pipeline.py        # generate tiles from raw images
python src/train.py                     # train the model
python src/evaluate.py                  # evaluate and generate plots
python src/scripts/sanity_check.py      # quick model validation
python src/scripts/print_summary.py results   # print evaluation metrics
```


## How the Pipeline Works

### Configuration (src/config.py)

Every setting lives in one file. Paths, tile sizes, color thresholds, model parameters, training hyperparameters, augmentation settings. Other files import from here so changing a setting only requires editing one place. The main knobs to turn:

- `MODEL_TYPE` switches between `"retinanet"`, `"pretrained_ssd"`, and `"custom_ssd"`
- `NUM_EPOCHS = 150` with early stopping patience of 25
- `SAMPLING_BALANCE_FACTOR = 10` controls positive tile sampling frequency
- `NMS_CONFIDENCE_THRESHOLD = 0.5` confidence cutoff for keeping detections

### Data Pipeline (src/data/data_pipeline.py)

Takes the raw images and their XML annotations and chops each image into tiles using 4 different grid offsets: (0,0), (128,0), (0,128), (128,128). This way Waldo falls on different tile boundaries in each grid, giving us more diverse positive samples. The pipeline also runs random crop augmentation (src/data/augment.py) to generate additional crops with Waldo at random positions within the tile.

Tiles get split into train/val/test sets by image id (70/15/15) so the model never sees tiles from the same source image in both training and testing. This prevents data leaking where the model could just memorize background patterns.

Training augmentations include weighted random sampling (positive tiles show up roughly once per 10 negatives), horizontal flips with bounding box adjustment, color jitter on positive tiles, gaussian blur, and random sharpness.

### Preprocessing (src/data/pre_processing.py)

This is our main contribution from the paper adaptation. Each tile gets two extra channels:

1. **HSV mask**: converts to HSV color space and highlights pixels that match Waldo's red or white colors. The thresholds are configurable in config.py.
2. **Sobel edge detection on the HSV mask**: runs Sobel filters specifically on the HSV mask output, not on the raw image. This focuses edge detection on Waldo's stripe pattern rather than picking up every edge in the busy scene.

The model receives 5 channels total (3 RGB + 1 HSV mask + 1 Sobel edges) and uses a learnable 1x1 convolution to merge them back to 3 channels for the backbone.

### Models (src/models/model.py)

We have three model classes, all in one file:

**WaldoDetectorRetinaNet** is our best model. ResNet50 backbone with a Feature Pyramid Network, pretrained on COCO. The backbone stays frozen to preserve those pretrained features since we only have around 70 unique positive tiles. Focal loss is built in, which helps a lot with the massive class imbalance between positive and negative tiles.

**WaldoDetectorPretrained** uses the SSD300 VGG16 architecture from COCO. We swap out the classification head from 91 classes to 2 (Waldo and background) and keep everything else frozen. It ended up detecting nearly everything as Waldo, likely because the model is too eager to find objects with limited fine tuning data.

**WaldoDetector** is our custom SSD built on MobileNetV2. Five custom detection heads (src/models/ssd_block.py) at different scales, generating 1364 total anchors. This was our starting point and helped us understand the SSD architecture before moving to pretrained models.

### Training (src/train.py)

Handles all three model types with mixed precision training (AMP) for speed. Pretrained models use SGD with momentum, while the custom SSD uses Adam. All models use gradient clipping, ReduceLROnPlateau scheduling, and early stopping. Focal loss (gamma=2.0, alpha=0.25) handles the class imbalance by down weighting the easy negatives that dominate each batch.

Training caps at 4000 samples per epoch and 1000 for validation to keep training time reasonable with the tiled approach.

### Evaluation (src/evaluate.py)

Loads the best checkpoint and runs it on the test tiles. Uses an HSV pre filter to skip tiles with no red or white pixels (since Waldo cannot be in a tile without his colors) and test time augmentation (horizontal flip) for better accuracy.

Generates a full set of outputs:
- Confusion matrix, precision recall curve, ROC curve
- F1 vs confidence threshold analysis
- Confidence distribution comparing positive and negative tiles
- Training loss curves
- Sample tiles showing true positives, false positives, and false negatives with bounding boxes
- Full image demos with predicted detection boxes on original puzzle images


## Dataset

We use 458 images total:
- 19 original Where's Waldo puzzle illustrations with hand drawn bounding box annotations
- 102 images from the FindingWaldo dataset
- 137 brightness and gamma variant images from FindingWaldo
- 200 synthetic images (10 Waldo cutouts pasted onto 100 background scenes at random positions)

The synthetic generation (src/data/paste_waldo.py) helped significantly. With only the original images, we were working with roughly 70 positive tiles after tiling. The synthetic images and random crop augmentation bring that up to several thousand positive training samples.

All annotations use Pascal VOC XML format with bounding boxes around Waldo.


## Experiment Framework

The experiments/ folder contains a self contained experiment system. Each experiment script (like run_retinanet.py) inherits from base_experiment.py, overrides the config settings it needs, trains the model, evaluates it, and saves everything to its own folder.

The ablation study (run_ablation.py) is a bit special because it runs each variant as a separate Python process. This way GPU memory gets fully released between variants, which matters when you are running three full training runs back to back on a single GPU.

After running experiments, generate_report.py loads all the saved results and creates comparison bar charts and a combined results JSON.


## Hardware

Training was done on an NVIDIA GTX 1080 Ti (11GB VRAM). RetinaNet at 512x512 takes about 5 to 7 minutes per epoch. The pipeline also supports Apple Silicon (MPS) and CPU, but GPU training is strongly recommended.


## Requirements

- Python 3.8+
- PyTorch (>=2.1.0) with CUDA support recommended
- torchvision (>=0.16.0)
- Pillow (>=10.0.0), matplotlib (>=3.7.0)
- ultralytics (>=8.0.0) for YOLOv8 comparison

All dependencies are pinned with version ranges in `requirements.txt`. Run `bash setup.sh` to check and install only what is missing.


## References

[1] N. Lamb and S. Chuah, "A Strawberry Detection System Using Convolutional Neural Networks," *2018 IEEE International Conference on Big Data (Big Data)*, Seattle, WA, USA, 2018, pp. 2515-2520.

[2] vc1492a, "Hey-Waldo," GitHub repository. [Online]. Available: https://github.com/vc1492a/Hey-Waldo

[3] bckenstler, "TheresWaldo," GitHub repository. [Online]. Available: https://github.com/bckenstler/TheresWaldo

[4] tadejmagajna, "FindingWaldo," GitHub repository. [Online]. Available: https://github.com/tadejmagajna/FindingWaldo

[5] A. Howard et al., "Searching for MobileNetV3," *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 1314-1324.

[6] W. Liu et al., "SSD: Single Shot MultiBox Detector," *European Conference on Computer Vision (ECCV)*, 2016, pp. 21-37.

[7] T. Lin et al., "Focal Loss for Dense Object Detection," *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 2980-2988.

[8] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLO," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics


## License

This is a university course project for COMP 4360 at the University of Manitoba. Not intended for redistribution. Where's Waldo images are copyrighted by Martin Handford / Walker Books.
