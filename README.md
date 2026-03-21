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
├── train.py                      # entry point for training
├── evaluate.py                   # entry point for testing and visualization
│
├── src/
│   ├── config.py                 # all settings in one place
│   ├── data_pipeline.py          # tiling, dataset, dataloader
│   ├── pre_processing.py         # HSV mask, Sobel edges, preprocess_batch
│   ├── MobileV2Backbone.py       # MobileNetV2 feature extractor
│   ├── ssd.py                    # single SSD detection block
│   ├── ConcatenateAnchors.py     # anchor generation and prediction reshaping
│   └── model.py                  # full model wiring everything together
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

### Step 1: Data Pipeline (data_pipeline.py) [DONE]
Takes the 19 raw images and their XML annotations, chops each image into tiles at three resolutions (64x64, 128x128, 256x256), checks which tiles contain Waldo using overlap with the bounding box, splits tiles into train/val/test sets by image id so theres no data leaking, and provides a PyTorch Dataset and DataLoader so other files can easily load batches of tiles.

### Step 2: Pre Processing (pre_processing.py) [DONE]
Takes a batch of image tiles and creates two extra "cheat sheet" channels. The HSV mask highlights red and white pixels (Waldos stripes), and the Sobel edge detector highlights strong edges (Waldos stripe pattern creates horizontal edges). These get concatenated with the original 3 RGB channels to give the model 5 channels of input instead of 3, giving it extra hints about where Waldo might be.

### Step 3: Feature Extraction (MobileV2Backbone.py) [DONE]
Uses a pretrained MobileNetV2 (trained on ImageNet) as a feature extractor. We take only the first 11 layers, which turns a 256x256 tile into a 16x16 feature map with 64 channels. The idea is that MobileNet has already learned to recognize shapes, edges, and textures, so we dont need to learn that from scratch.

### Step 4: SSD Detection Heads (ssd.py) [DONE]
A single SSD block that takes a feature map and does two things: predicts bounding box offsets (where is Waldo relative to anchor positions) and predicts class scores (Waldo or not Waldo). Then it downsamples the feature map by half so the next head can make predictions at a coarser scale. We chain 6 of these together to detect Waldo at different sizes.

### Step 5: Concatenate Anchors (ConcatenateAnchors.py) [DONE]
Right now it just collects the SSD head outputs and stuffs them into a list without actually doing anything with them. It doesnt reshape the predictions, doesnt generate anchor positions, and doesnt combine anything into a usable format. It needs to generate anchor box positions at each scale, reshape all the predictions from all 6 heads, and combine them into two clean tensors: all bounding boxes [batch, total_anchors, 4] and all class scores [batch, total_anchors, 2].

### Step 6: Full Model (model.py) [NEEDS WORK]
This is where we wire everything together into one nn.Module. Image tile goes in, Waldo predictions come out. The flow is: pre_processing adds extra channels, a 1x1 conv merges 5 channels back to 3, backbone extracts features, 6 SSD heads make predictions at different scales, ConcatenateAnchors combines all predictions, and NMS filters out overlapping boxes.

### Step 7: Training (train.py) [NEEDS WORK]
The training loop. Loads data from the pipeline, runs it through the model, computes loss (smooth L1 for bounding boxes + cross entropy for classification), handles class imbalance (we have about 14000 negative tiles but only 56 positive ones), updates weights, and saves checkpoints after each epoch.

### Step 8: Evaluation (evaluate.py) [NEEDS WORK]
Tests the trained model on tiles it has never seen. Computes accuracy, precision, and recall. Visualizes results by drawing predicted bounding boxes on tiles and stitching them back onto the full image to show where the model thinks Waldo is.

### Step 9: Non Maximum Suppression [NEEDS WORK]
Filters out overlapping bounding box predictions and keeps only the best ones. When multiple anchors predict Waldo in roughly the same spot, NMS picks the one with the highest confidence and throws away the rest.


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
