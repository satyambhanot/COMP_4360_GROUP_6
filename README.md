# Where's Waldo?
# CNN-Based Character Detection in Cluttered Scenes

**Course:** COMP 4360 Machine Learning, University of Manitoba
**Team:** Rudy Kissel, Satyam Bhanot, Gabriel Barnabe
**Instructor:** Dr. Christopher Henry
**Date:** March 2026

This project adapts a strawberry detection system (Lamb & Chuah, IEEE Big Data 2018) to the domain of locating Waldo in "Where's Waldo?" puzzle illustrations. We use an SSD300 VGG16 pretrained on COCO as our primary model, replacing the classification head to detect Waldo specifically. Our main contribution is a custom preprocessing pipeline that adds HSV color masking and Sobel edge detection channels to exploit Waldo's red and white stripe pattern. The model processes high resolution puzzle images as 256x256 tiles, predicting per tile bounding boxes for Waldo's location.

We also built a custom SSD with a MobileNetV2 backbone and 5 detection heads trained from scratch, which serves as a baseline for comparison. Both models use the same HSV + Sobel preprocessing.


## Project Structure

```
COMP_4360_GROUP_6/
├── README.md
├── .gitignore
├── requirements.txt
├── run.sh                        # runs the entire pipeline start to finish
├── train.py                      # entry point for training the model
├── evaluate.py                   # evaluation with 10+ output plots
├── augment_images.py             # generates random crop tiles around waldo
├── evaluate_ablation.py          # ablation studies (preprocessing impact)
├── compare_results.py            # cross model comparison charts
├── train_yolo.py                 # standalone YOLOv8 comparison experiment
├── test_confidence.py            # quick confidence check on test tiles
│
├── src/
│   ├── config.py                 # all settings in one place
│   ├── data_pipeline.py          # tiling, dataset, dataloader, augmentation
│   ├── pre_processing.py         # HSV mask, Sobel edges, preprocess_batch
│   ├── ssd.py                    # single SSD detection block (custom model)
│   ├── ConcatenateAnchors.py     # anchor generation and prediction reshaping
│   ├── model.py                  # WaldoDetector + WaldoDetectorPretrained
│   └── nms.py                    # non maximum suppression
│
├── data/
│   ├── raw/
│   │   ├── images/               # 258 wheres waldo images (19 original + 239 variants)
│   │   └── annotations/          # 258 XML bounding box annotations (Pascal VOC format)
│   └── processed/                # generated tiles and splits.json
│
├── checkpoints/                  # saved model weights and training history
└── results/                      # evaluation outputs, plots, and run log
```


## How to Run

The easiest way to run everything is the shell script. It handles the full pipeline from tiling to training to evaluation:

```bash
bash run.sh
```

This will:
1. Install dependencies from requirements.txt
2. Tile the raw images into 256x256 tiles
3. Print a model summary
4. Train the model (with early stopping)
5. Run a sanity check on training tiles
6. Test confidence on test tiles
7. Run full evaluation with metrics and visualizations

All output gets saved to `results/` and the complete log goes to `results/run_log.txt`.

To run individual steps manually:

```bash
pip install -r requirements.txt
python src/data_pipeline.py    # generate tiles
python train.py                # train the model
python evaluate.py             # evaluate and generate plots
```


## Pipeline Steps

### Step 1: Configuration (src/config.py)

All settings for the project live in one file. Paths, tile sizes, color thresholds, model architecture parameters, training hyperparameters, augmentation settings. Other files import from here so we only have to change a setting in one place.

Key settings:
- `MODEL_TYPE = "retinanet"` switches between custom SSD, pretrained SSD300, and RetinaNet
- `NUM_EPOCHS = 1000` with early stopping patience of 25
- `SAMPLING_BALANCE_FACTOR = 10` controls how often positive tiles are sampled during training
- `NMS_CONFIDENCE_THRESHOLD = 0.5` confidence cutoff for detections

### Step 2: Data Pipeline (src/data_pipeline.py)

Takes the 258 raw images and their XML annotations and chops each image into 256x256 tiles using 4 different grid offsets (0,0), (128,0), (0,128), (128,128). This means Waldo falls on different tile boundaries in each grid, giving us more diverse positive tiles. Also runs random crop augmentation (augment_images.py) to generate additional positive tiles with Waldo at random positions.

Splits tiles into train/val/test sets by image id (70/15/15) so the model never sees tiles from the same image in both training and testing. This prevents data leaking where the model memorizes background patterns instead of learning what Waldo looks like.

Also handles data augmentation for training:
- Weighted random sampling so positive tiles show up more often (roughly 1 positive per 10 negatives in each batch)
- Horizontal flip with bounding box adjustment
- Color jitter on positive tiles only (brightness, contrast, saturation)
- Gaussian blur (20% chance)
- Random sharpness adjustment (20% chance)
- Random crop augmentation: 15 random 256x256 patches per large image, each with Waldo at a different position plus a flipped version of each crop

### Step 3: Pre Processing (src/pre_processing.py)

Takes a batch of image tiles and creates two extra channels that give the model hints about where Waldo might be:

1. **HSV mask**: converts to HSV color space and highlights pixels that are red or white, because Waldo's stripes are red and white. Uses configurable hue, saturation, and value thresholds from config.py
2. **Sobel edge detection on the HSV mask**: runs Sobel filters on the HSV mask output (not on the raw image). This focuses edge detection specifically on Waldo's stripe pattern rather than detecting every edge in the busy scene

The model gets 5 channels total: 3 RGB + 1 HSV mask + 1 Sobel edges. A learnable 1x1 convolution merges these back to 3 channels for the backbone.

### Step 4: Feature Extraction

**Pretrained SSD (primary model):** Uses the VGG16 backbone from SSD300 pretrained on COCO (330,000 images, 90 object classes). The backbone is frozen so the pretrained features are preserved. With only ~70 positive training tiles, unfreezing the backbone caused it to overfit and detect everything as Waldo. The frozen COCO features already extract useful edges, textures, and shapes that transfer well to cartoon illustrations.

**Custom SSD (baseline model):** Uses the first 11 layers of MobileNetV2 pretrained on ImageNet as a frozen feature extractor. Turns a 256x256 tile into a 16x16 feature map with 64 channels.

### Step 5: Detection Heads

**Pretrained SSD:** Uses the original SSD300 detection heads from COCO. We replace only the classification head from 91 classes to 2 (Waldo + background). The box regression head stays the same. Weight decay (L2 regularization) prevents overfitting.

**Custom SSD (src/ssd.py):** 5 custom SSD blocks chained together. Each block predicts bounding box offsets and class scores at its scale, then pools the feature map to half size for the next block:
- Head 1: 16x16 (1024 anchors)
- Head 2: 8x8 (256 anchors)
- Head 3: 4x4 (64 anchors)
- Head 4: 2x2 (16 anchors)
- Head 5: 1x1 (4 anchors)
- Total: 1364 anchors

### Step 6: Anchor Generation (src/ConcatenateAnchors.py)

Generates anchor box positions at each grid size. Reshapes the raw SSD output from convolution format into flat lists of predictions. Combines predictions from all heads into two final tensors: all bounding boxes [batch, total_anchors, 4] and all class scores [batch, total_anchors, 2]. Only used for the custom SSD since torchvision handles this internally for the pretrained model.

### Step 7: Non Maximum Suppression (src/nms.py)

After the model makes predictions for every anchor, many anchors in the same area will fire. NMS keeps only the highest confidence prediction in each region and removes overlapping boxes (IoU threshold 0.5). Only used during inference, not during training.

### Step 8: Training (train.py)

Handles both model types:

**Pretrained SSD:**
- SGD optimizer with momentum 0.9
- Learning rate 0.001 (backbone frozen, only heads train)
- Weight decay 0.0001 (L2 regularization)
- Focal loss (gamma=2.0, alpha=0.25) to down-weight easy negatives
- Torchvision SSD returns its own loss dict (classification + bbox regression)
- Gradient clipping (max_norm=1.0)

**Custom SSD:**
- Adam optimizer at 0.001
- Custom SSD loss: smooth L1 for bounding boxes + cross entropy for classification
- Hard negative mining to handle class imbalance

Both models use:
- ReduceLROnPlateau scheduler (halves LR after 10 epochs of no improvement)
- Early stopping (stops training after 25 epochs of no improvement)
- Checkpoints saved for best model and last model
- Training history saved to JSON for plotting

### Step 9: Evaluation (evaluate.py)

Loads the best checkpoint and runs on test tiles the model has never seen. Uses an HSV pre-filter to skip tiles with no red/white pixels (Waldo's colors) and test-time augmentation (horizontal flip) for better accuracy. Outputs:

- **Metrics**: accuracy, precision, recall, F1 score, confusion matrix
- **Threshold analysis**: tests confidence thresholds from 0.1 to 0.9, finds the best F1 threshold
- **Precision recall curve**: standard PR curve for the report
- **Confidence distribution**: histogram comparing positive vs negative tile confidences
- **Training curves**: train and validation loss per epoch
- **Sample tiles**: saves examples of true positives (green box), false positives (red box), and false negatives (yellow box)
- **Full image demo**: runs detection on 3 full sized Waldo images, draws predicted bounding boxes

All results saved to `results/` folder as images and JSON.


## Dataset

We use 258 images total:
- 19 original Where's Waldo puzzle illustrations with hand drawn bounding box annotations
- 102 images from the FindingWaldo dataset (originals)
- 137 brightness and gamma variant images from FindingWaldo

All annotations are in Pascal VOC XML format. The data pipeline generates tiles at 256x256 using 4 grid offsets plus random crop augmentation. The augmentation significantly increases the number of positive tiles from ~69 (fixed grid only) to several hundred, which helps the model learn what Waldo specifically looks like.


## Models

We tried three architectures during this project, progressing from simple to more capable:

| | RetinaNet (primary) | Pretrained SSD300 | Custom SSD (baseline) |
|---|---|---|---|
| Backbone | ResNet50 + FPN (COCO) | VGG16 (COCO) | MobileNetV2 (ImageNet) |
| Detection heads | RetinaNet FPN heads | SSD300 heads | 5 custom SSD blocks |
| Input size | 800x800 | 300x300 | 256x256 |
| Focal loss | Built in | Manual (added by us) | Manual |
| Backbone training | Frozen | Frozen | Frozen |
| Optimizer | SGD (momentum 0.9) | SGD (momentum 0.9) | Adam |
| Config flag | `MODEL_TYPE = "retinanet"` | `MODEL_TYPE = "pretrained_ssd"` | `MODEL_TYPE = "custom_ssd"` |

We also compare against YOLOv8 (via `train_yolo.py`) as a state of the art reference point. YOLOv8 processes full images at 1280x1280 instead of tiles, giving it significantly more context.


## Ablation Studies

`evaluate_ablation.py` runs two ablation studies:

1. **Preprocessing impact**: trains the model three ways (RGB only, RGB + HSV, RGB + HSV + Sobel) to measure how much our custom preprocessing channels contribute
2. **Architecture comparison**: loads saved results from each model type (custom SSD, pretrained SSD, RetinaNet, YOLOv8) and generates side by side comparison charts

Each preprocessing experiment trains for 30 epochs with the same random seed for fair comparison. Results are saved to `results/ablation/` as charts and JSON.


## Configuration

All settings are centralized in `src/config.py`. This includes file paths, tile sizes, overlap thresholds, train/val/test ratios, ImageNet normalization values, HSV color thresholds, model architecture parameters, data augmentation settings, training hyperparameters, and NMS thresholds. Change settings there instead of editing individual files.


## Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- torchvision
- Pillow
- matplotlib

Install with:
```bash
pip install -r requirements.txt
```


## Hardware

Training was done on an NVIDIA GTX 1080 Ti. The pipeline also supports Apple Silicon (MPS) and CPU, but training on CPU will be significantly slower. run.sh prints hardware info at the start of each run for reproducibility.


## References

[1] N. Lamb and S. Chuah, "A Strawberry Detection System Using Convolutional Neural Networks," *2018 IEEE International Conference on Big Data (Big Data)*, Seattle, WA, USA, 2018, pp. 2515-2520.

[2] vc1492a, "Hey-Waldo," GitHub repository. [Online]. Available: https://github.com/vc1492a/Hey-Waldo

[3] bckenstler, "TheresWaldo," GitHub repository. [Online]. Available: https://github.com/bckenstler/TheresWaldo

[4] tadejmagajna, "FindingWaldo," GitHub repository. [Online]. Available: https://github.com/tadejmagajna/FindingWaldo

[5] A. Howard et al., "Searching for MobileNetV3," *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 1314-1324.

[6] W. Liu et al., "SSD: Single Shot MultiBox Detector," *European Conference on Computer Vision (ECCV)*, 2016, pp. 21-37.


## License

This is a university course project for COMP 4360 at the University of Manitoba. Not intended for redistribution. Where's Waldo images are copyrighted by Martin Handford / Walker Books.
