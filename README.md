# Where's Waldo?
# CNN-Based Character Detection in Cluttered Scenes

**Course:** COMP 4360 Machine Learning, University of Manitoba
**Team:** Rudy Kissel, Satyam Bhanot, Gabriel Barnabe
**Instructor:** Dr. Christopher Henry
**Date:** March 2026

This project adapts a strawberry detection system (Lamb & Chuah, IEEE Big Data 2018) to the domain of locating Waldo in "Where's Waldo?" puzzle illustrations. We replace the original 3-layer CNN with a MobileNet V3 Large backbone, extend the SSD detection heads from 4 to 6 levels, and introduce dual HSV masking with Sobel edge detection to exploit Waldo's distinctive red-and-white stripe pattern. The model processes high-resolution puzzle images as overlapping tiles at 64x64, 128x128, and 256x256 resolutions, predicting per-tile bounding boxes for Waldo's location.


## Project Structure

```
COMP_4360_GROUP_6/
├── README.md
├── .gitignore
├── requirements.txt
├── train.py                      # entry point for training the model
├── evaluate.py                   # entry point for testing and visualization
├── evaluate_ablation.py          # ablation studies to test each component
│
├── src/
│   ├── config.py                 # all settings in one place
│   ├── data_pipeline.py          # tiling, dataset, dataloader
│   ├── pre_processing.py         # HSV mask, Sobel edges, preprocess_batch
│   ├── MobileV2Backbone.py       # MobileNetV2 feature extractor
│   ├── ssd.py                    # single SSD detection block
│   ├── ConcatenateAnchors.py     # anchor generation and prediction reshaping
│   ├── model.py                  # full model wiring everything together
│   └── nms.py                    # non maximum suppression
│
├── data/
│   ├── raw/
│   │   ├── images/               # 19 original wheres waldo images
│   │   └── annotations/          # 19 XML bounding box annotations
│   └── processed/                # generated tiles and splits.json
│
└── checkpoints/                  # saved model weights
```


## Pipeline Steps

The full detection pipeline follows these steps. Each step maps to a file in the codebase.

### Step 1: Configuration (config.py) [DONE]
- All settings for the project live in one file
- Paths, tile sizes, thresholds, model architecture, training parameters
- Other files import from here instead of hardcoding values
- Makes it easy to change a setting without hunting through every file

### Step 2: Data Pipeline (data_pipeline.py) [DONE]
- Takes the 19 raw images and their XML annotations
- Chops each image into tiles at three resolutions (64x64, 128x128, 256x256)
- Checks which tiles contain Waldo using overlap with the bounding box
- Splits tiles into train/val/test sets by image id so theres no data leaking
- Provides a PyTorch Dataset and DataLoader so other files can easily load batches

### Step 3: Pre Processing (pre_processing.py) [DONE]
- Takes a batch of image tiles and creates two extra "cheat sheet" channels
- HSV mask highlights red and white pixels (Waldos stripes)
- Sobel edge detector highlights strong edges (Waldos stripe pattern creates horizontal edges)
- Concatenates these with the original 3 RGB channels to give the model 5 channels instead of 3
- Gives the model extra hints about where Waldo might be

### Step 4: Feature Extraction (MobileV2Backbone.py) [DONE]
- Uses a pretrained MobileNetV2 (trained on ImageNet) as a feature extractor
- Takes only the first 11 layers of MobileNet
- Turns a 256x256 tile into a 16x16 feature map with 64 channels
- MobileNet has already learned to recognize shapes, edges, and textures so we dont start from scratch

### Step 5: SSD Detection Heads (ssd.py) [DONE]
- A single SSD block that takes a feature map and predicts two things
- Bounding box offsets (where is Waldo relative to anchor positions)
- Class scores (Waldo or not Waldo)
- Downsamples the feature map by half so the next head can detect at a coarser scale
- We chain multiple of these together to detect Waldo at different sizes

### Step 6: Concatenate Anchors (ConcatenateAnchors.py) [DONE]
- Generates anchor box positions at each grid size (16x16, 8x8, 4x4, 2x2, 1x1)
- Reshapes the raw SSD output from convolution format into flat lists of predictions
- Combines predictions from all heads into two final tensors
- All bounding boxes [batch, 1364, 4] and all class scores [batch, 1364, 2]
- 1364 total anchors across all heads (1024 + 256 + 64 + 16 + 4)

### Step 7: Full Model (model.py) [NEEDS WORK]
- Wire everything together into one nn.Module
- Image tile goes in, Waldo predictions come out
- Pre processing adds extra channels (3 to 5)
- A 1x1 conv merges 5 channels back to 3 for the backbone
- Backbone extracts features
- SSD heads make predictions at different scales
- ConcatenateAnchors combines all predictions

### Step 8: Non Maximum Suppression (nms.py) [NEEDS WORK]
- Filters out overlapping bounding box predictions
- When multiple anchors predict Waldo in roughly the same spot
- Picks the one with the highest confidence and throws away the rest
- Only used during inference, not during training

### Step 9: Training (train.py) [NEEDS WORK]
- Loads training and validation data from the data pipeline
- Runs tiles through the model and computes loss
- SSD loss combines smooth L1 (for bounding boxes) and cross entropy (for classification)
- Handles class imbalance (about 14000 negative tiles but only 56 positive)
- Saves checkpoints to the checkpoints folder after each epoch

### Step 10: Evaluation (evaluate.py) [NEEDS WORK]
- Loads a saved model checkpoint
- Runs it on test tiles the model has never seen
- Computes accuracy, precision, and recall
- Visualizes results by drawing predicted bounding boxes on tiles
- Can stitch predictions back onto the full image to show where Waldo is

### Step 11: Ablation Studies (evaluate_ablation.py) [NEEDS WORK]
- Tests if each piece of the pipeline actually helps
- Runs the model with different components turned off
- No HSV mask, no Sobel edges, no pre processing at all, fewer SSD heads
- Compares results to show which components matter most
- This is what shows the professor we understand why each part is there


## Configuration

All settings are centralized in `src/config.py`. This includes file paths, tile sizes, overlap thresholds, train/val/test ratios, ImageNet normalization values, HSV color thresholds, model architecture parameters (backbone layers, number of SSD heads, anchors per head), and training hyperparameters. Change settings there instead of editing individual files.


## Setup

```
pip install -r requirements.txt
python src/data_pipeline.py
```

The first command installs dependencies. The second tiles the raw images and creates the train/val/test splits.


## References

[1] N. Lamb and S. Chuah, "A Strawberry Detection System Using Convolutional Neural Networks," *2018 IEEE International Conference on Big Data (Big Data)*, Seattle, WA, USA, 2018, pp. 2515-2520.

[2] vc1492a, "Hey-Waldo," GitHub repository. [Online]. Available: https://github.com/vc1492a/Hey-Waldo

[3] bckenstler, "TheresWaldo," GitHub repository. [Online]. Available: https://github.com/bckenstler/TheresWaldo

[4] A. Howard et al., "Searching for MobileNetV3," *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 1314-1324.

[5] W. Liu et al., "SSD: Single Shot MultiBox Detector," *European Conference on Computer Vision (ECCV)*, 2016, pp. 21-37.

## License

This is a university course project for COMP 4360 at the University of Manitoba. Not intended for redistribution. Where's Waldo images are copyrighted by Martin Handford / Walker Books.
