# Where's Waldo?
# CNN-Based Character Detection in Cluttered Scenes

**Course:** COMP 4360 Machine Learning, University of Manitoba
**Team:** Rudy Kissel, Satyam Bhanot, Gabriel Barnabe
**Instructor:** Dr. Christopher Henry
**Date:** March 2026

This project adapts a strawberry detection system (Lamb & Chuah, IEEE Big Data 2018) to the domain of locating Waldo in "Where's Waldo?" puzzle illustrations. We use an SSD300 VGG16 pretrained on COCO as our primary model, replacing the classification head to detect Waldo specifically. Our main contribution is a custom preprocessing pipeline that adds HSV color masking and Sobel edge detection channels to exploit Waldo's red and white stripe pattern. The model processes high resolution puzzle images as tiles at 64x64, 128x128, and 256x256 resolutions, predicting per tile bounding boxes for Waldo's location.

We also built a custom SSD with a MobileNetV2 backbone and 5 detection heads trained from scratch, which serves as a baseline for comparison. Both models use the same HSV + Sobel preprocessing.


## Project Structure

```
COMP_4360_GROUP_6/
├── README.md
├── .gitignore
├── requirements.txt
├── run.sh                        # runs the entire pipeline start to finish
├── train.py                      # entry point for training the model
├── evaluate.py                   # entry point for testing and visualization
├── evaluate_ablation.py          # ablation study plans (comments only for now)
├── test_confidence.py            # quick confidence check on test tiles
│
├── src/
│   ├── config.py                 # all settings in one place
│   ├── data_pipeline.py          # tiling, dataset, dataloader, augmentation
│   ├── pre_processing.py         # HSV mask, Sobel edges, preprocess_batch
│   ├── MobileV2Backbone.py       # MobileNetV2 feature extractor (custom model)
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
2. Tile the raw images at three resolutions
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
- `USE_PRETRAINED_SSD = True` switches between pretrained SSD300 and custom SSD
- `NUM_EPOCHS = 1000` with early stopping patience of 50
- `SAMPLING_BALANCE_FACTOR = 20` controls how often positive tiles are sampled during training
- `NMS_CONFIDENCE_THRESHOLD = 0.3` for pretrained SSD (0.5 for custom)

### Step 2: Data Pipeline (src/data_pipeline.py)

Takes the 258 raw images and their XML annotations and chops each image into tiles at three resolutions (64x64, 128x128, 256x256). Checks which tiles contain Waldo by computing overlap with the bounding box annotation. If at least 50% of Waldo is inside a tile, that tile gets labeled as positive.

Splits tiles into train/val/test sets by image id (70/15/15) so the model never sees tiles from the same image in both training and testing. This prevents data leaking where the model memorizes background patterns instead of learning what Waldo looks like.

Also handles data augmentation for training:
- Weighted random sampling so positive tiles show up more often (we have ~259 positive vs ~90,000 negative tiles)
- Horizontal flip with bounding box adjustment
- Color jitter on positive tiles only (brightness, contrast, saturation)
- Random grayscale conversion (10% chance)
- Gaussian blur (20% chance)
- Random sharpness adjustment (20% chance)

### Step 3: Pre Processing (src/pre_processing.py)

Takes a batch of image tiles and creates two extra channels that give the model hints about where Waldo might be:

1. **HSV mask**: converts to HSV color space and highlights pixels that are red or white, because Waldo's stripes are red and white. Uses configurable hue, saturation, and value thresholds from config.py
2. **Sobel edge detection on the HSV mask**: runs Sobel filters on the HSV mask output (not on the raw image). This focuses edge detection specifically on Waldo's stripe pattern rather than detecting every edge in the busy scene

The model gets 5 channels total: 3 RGB + 1 HSV mask + 1 Sobel edges. A learnable 1x1 convolution merges these back to 3 channels for the backbone.

### Step 4: Feature Extraction

**Pretrained SSD (primary model):** Uses the VGG16 backbone from SSD300 pretrained on COCO (330,000 images, 90 object classes). The entire backbone is unfrozen and fine tuned with a lower learning rate (0.0001) so it can adapt from COCO photos to Waldo's cartoon style without losing what it already knows.

**Custom SSD (baseline model):** Uses the first 11 layers of MobileNetV2 pretrained on ImageNet as a frozen feature extractor. Turns a 256x256 tile into a 16x16 feature map with 64 channels.

### Step 5: Detection Heads

**Pretrained SSD:** Uses the original SSD300 detection heads from COCO. We replace only the classification head from 91 classes to 2 (Waldo + background). The box regression head stays the same. Dropout (20%) is applied after the preprocessing channel merge for regularization.

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
- Differential learning rates: backbone at 0.0001, detection heads at 0.001
- Weight decay 0.0001 (L2 regularization)
- Torchvision SSD returns its own loss dict (classification + bbox regression)
- Gradient clipping (max_norm=1.0)

**Custom SSD:**
- Adam optimizer at 0.001
- Custom SSD loss: smooth L1 for bounding boxes + cross entropy for classification
- Hard negative mining to handle class imbalance

Both models use:
- ReduceLROnPlateau scheduler (halves LR after 10 epochs of no improvement)
- Early stopping (stops training after 50 epochs of no improvement)
- Checkpoints saved for best model and last model
- Training history saved to JSON for plotting

### Step 9: Evaluation (evaluate.py)

Loads the best checkpoint and runs on test tiles the model has never seen. Outputs:

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

All annotations are in Pascal VOC XML format. The data pipeline generates roughly 90,000+ tiles across the three resolutions, of which about 259 contain Waldo (extreme class imbalance).


## Models

| | Pretrained SSD300 (primary) | Custom WaldoDetector (baseline) |
|---|---|---|
| Backbone | VGG16 (COCO pretrained) | MobileNetV2 (ImageNet pretrained) |
| Detection heads | SSD300 heads (COCO pretrained) | 5 custom SSD blocks (from scratch) |
| Backbone training | Fine tuned (unfrozen, lower LR) | Frozen |
| Total parameters | ~35M | ~2.5M |
| Preprocessing | HSV mask + Sobel on HSV | HSV mask + Sobel on HSV |
| Optimizer | SGD (momentum 0.9) | Adam |
| Config flag | `USE_PRETRAINED_SSD = True` | `USE_PRETRAINED_SSD = False` |


## Ablation Studies (Planned)

We have 5 ablation studies designed in `evaluate_ablation.py`:

1. **Sobel edge bypass**: full model vs no Sobel channel
2. **Model size**: 5 SSD heads vs 3 SSD heads
3. **Tile resolution**: 64x64 vs 128x128 vs 256x256
4. **Color information**: full color vs grayscale vs black and white
5. **Pretrained vs random backbone**: ImageNet weights vs random initialization

Each experiment trains on the same data with the same random seed for fair comparison. The model already supports ablation flags (`use_hsv`, `use_sobel`, `use_pretrained`, `num_heads`).


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
