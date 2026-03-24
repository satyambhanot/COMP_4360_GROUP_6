# the full flow through the model:
#
#   1. images come in as [N, 3, 256, 256] from the dataloader
#
#   2. preprocess_batch adds the HSV mask and sobel edges
#      now we have [N, 5, 256, 256] (3 RGB + 1 HSV + 1 Sobel)
#
#   3. but mobilenet expects 3 channels not 5
#      so we use a 1x1 conv to squish 5 channels back down to 3
#      this is a learnable layer so the model figures out
#      the best way to combine those 5 channels
#      now we have [N, 3, 256, 256] again
#
#   4. run it through the mobilenet backbone (first 11 layers)
#      this extracts features and shrinks the spatial size
#      now we have [N, 64, 16, 16]
#
#   5. pass the features through 5 SSD heads one after another
#      each head does two things:
#        predicts bounding box offsets (loc) for its anchors
#        predicts class scores (cls) for waldo vs background
#      then it pools the feature map to half the size for the next head
#
#      head 1: 16x16 -> predicts 1024 anchors -> pools to 8x8
#      head 2: 8x8   -> predicts 256 anchors  -> pools to 4x4
#      head 3: 4x4   -> predicts 64 anchors   -> pools to 2x2
#      head 4: 2x2   -> predicts 16 anchors   -> pools to 1x1
#      head 5: 1x1   -> predicts 4 anchors    -> pools to 0x0 (we ignore this)
#
#   6. concatenate_all_heads takes all 5 heads worth of predictions
#      and stacks them into one big tensor
#      total anchors: 1024 + 256 + 64 + 16 + 4 = 1364
#      loc output: [N, 1364, 4]
#      cls output: [N, 1364, 2]
#
# things i need to be careful about:
#   the backbone weights are frozen so we dont mess up what mobilenet learned
#   the ssd heads go in nn.ModuleList not a regular python list
#     otherwise pytorch wont track them and they wont save in checkpoints
#   the last head pools to 0x0 but we dont care because we already got
#     the loc and cls predictions before the pooling happens
#
# whats trainable and whats frozen:
#   frozen (we dont touch these weights):
#     the mobilenet backbone, it already knows how to extract features
#   trainable (these learn during training):
#     the 1x1 conv that merges 5 channels to 3
#     all 5 SSD heads (the loc and cls conv layers inside each one)
#   not a layer at all (just a function, no weights):
#     preprocess_batch, it just does math on the pixels
#     concatenate_all_heads, it just reshapes tensors

# TODO: add imports
# TODO: build the WaldoDetector class with __init__ and forward
# TODO: add a test at the bottom

from pre_processing import preprocess_batch
from MobileV2Backbone import MobileV2Backbone
import ConcatenateAnchors
import config
from data_pipeline import get_dataloader
from ssd import ssd_block


import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection._utils import BoxCoder
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import nms

import torchvision.ops as ops

from PIL import Image, ImageDraw

def draw_boxes(image, boxes, labels=None, scores=None, colour="green", width=2):
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(boxes):
        coord_list = box.tolist()  # convert tensor to list
        x_min, y_min, x_max, y_max = coord_list

        # Draw rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline = colour, width = width)

        text = ""
        if labels is not None:
            text += f"{labels[i]}"
        if scores is not None:
            text += f": {scores[i]:.2f}"

        if text:
            draw.text((x_min, y_min), text, fill = colour)

    return image

def create_tiles_from_image(image: Image, tile_size: int, overlap: int):
    tiles = []
    tile_coords = []
    
    width, height = image.size

    for h in range(0, height, height - overlap):
        for w in range(0, width, width - overlap):
            # Add some padding if its needed
            pad_bottom = max(0, (h + tile_size) - height)
            pad_right  = max(0, (w + tile_size) - width)

            # Create the tile as a section of the original image
            tile = image.crop((w, h, w + tile_size, h + tile_size))

            # Save the tile and the their offsets in the original image
            tiles.append(tile)
            tile_coords.append((w, h))  # keep global offset

    return tiles, tile_coords

class WaldoSSD(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = config.NUM_CLASSES,
        num_heads: int = config.NUM_SSD_HEADS,
        num_anchors: int = config.NUM_ANCHORS,
        in_channels: int = config.BACKBONE_CHANNELS,
        kernel_size: int = config.SSD_KERNEL_SIZE,
        device = 'cpu'
    ):
        super().__init__()
        # Backbone
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_heads = num_heads

                # Anchor Setup
        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))   # Set up BoxCoder to calculate positioning/size deltas for ground truth boxes relative to the anchors. Weights are used to scale regression targets (x, y, w, h).
        self.aspect_ratios = [[1.0, 2.0, 0.5] for _ in range(num_heads)]           # Configure the anchors (Square box, wide rectangle box, tall rectangle box).
        self.anchor_generator = DefaultBoxGenerator(                # Creates the anchors
            aspect_ratios = self.aspect_ratios,                         # Use the aspect ratios for the boxes
            min_ratio = 0.07,                                           # The smallest an aspect ratio can be scaled relative to the image size
            max_ratio = 0.9                                             # The maximum amount an aspect ration can be scaled relative to the image size
        )
        
        # After self.anchor_generator = DefaultBoxGenerator(...)
        print("--- Anchor Generator Diagnostics ---")
        num_anchors_per_level = self.anchor_generator.num_anchors_per_location()
        print(f"Anchors per location (per level): {num_anchors_per_level}")

        # Total anchors calculation
        # Assuming feature map sizes of 16x16, 8x8, 4x4
        grid_sizes = [16*16, 8*8, 4*4]
        total_expected = sum(n * g for n, g in zip(num_anchors_per_level, grid_sizes))
        print(f"Total anchors generated: {total_expected}")
        print("------------------------------------")

        self.device = device

        # SSD Heads
        self.blocks = nn.ModuleList([
            ssd_block(
                in_channels = in_channels,
                num_classes = num_classes,
                kernel_size = kernel_size,
                num_anchors = num_anchors,
                device = device
            )
            for _ in range(num_heads)
        ])


    def forward(self, image_batch, tile_coords=None, labels=None, ground_truth_bboxs=None):

        offset_preds = []
        class_preds = []
        feature_maps = []
        batch_size = image_batch.size(0)
        
        # For training
        labels = labels
        bboxes = ground_truth_bboxs

        print(f"Input to preprocessor is: {image_batch.shape}")
        processed_batch = image_batch # preprocess_batch(image_batch)         # Preprocess the images by applying HSV and the Sobol edge detector.
        feature_map = self.backbone(processed_batch)            # Pass the preprocessed images to the backbone.
        print(f"Output from the backbone is {feature_map.shape}")
        
        # Pass the feature maps generated by the backbone to the SSD heads.
        for block in self.blocks:

            print(f"Input to block is {feature_map.shape}")
            offset_pred, class_pred, next_feature_map = block(feature_map) # Get preditions from the block.

            print(f" -- Offset preds: {offset_pred.shape}")
            print(f" -- Class preds: {class_pred.shape}")
            
            # Flatten offsets
            # Move channels to the end as these are our predictions
            offset_pred = offset_pred.permute(0, 2, 3, 1).contiguous()
            class_pred = class_pred.permute(0, 2, 3, 1).contiguous()
            
            # Reshape to [Batch, total anchors, values per anchor]
            offset_pred = offset_pred.view(batch_size, -1, 4)
            class_pred = class_pred.view(batch_size, -1, self.num_classes)
            
            print(f" -- Offset preds (reshape): {offset_pred.shape}")
            print(f" -- Class preds (reshape): {class_pred.shape}")


            offset_preds.append(offset_pred)    # Save the offset predictions for NMS / training.
            class_preds.append(class_pred)      # Save the class prediction for NMS / training.
            feature_maps.append(feature_map)

            feature_map = next_feature_map      # Setup the feature maps for the next SSD block. 

        offset_preds = torch.cat(offset_preds, dim=1)  # (batch, total_anchors, 4)
        class_preds  = torch.cat(class_preds, dim=1)   # (batch, total_anchors, num_classes)

        print(f"Concatenated offset preds: {offset_preds.shape}")
        print(f"Concatenated class preds: {class_preds.shape}")
        print(f"Feature maps: {len(feature_maps)}")
        
        # Wrap image batch so we can use the anchor_generator
        B, C, H, W = image_batch.shape
        print(f"Batch shape: {image_batch.shape}")

        image_sizes = [(H, W)] * B
        print(f"Image size: {image_sizes}")
        
        image_list = ImageList(image_batch, image_sizes)
        anchors = self.anchor_generator(image_list, feature_maps)
        print(f"Anchors: {anchors}")
        
        # Output changes if its training or not
        if self.training:
            return offset_preds, class_preds, anchors

        if not self.training:
            print("Not training.")
            
            bbox_preds = self.box_coder.decode(offset_preds, anchors)
            
            # Adjust for tile positions if tiles are used
            if tile_coords is not None:
                for i, (w_off, h_off) in enumerate(tile_coords):
                    bbox_preds[i, :, 0] += w_off  # xmin
                    bbox_preds[i, :, 2] += w_off  # xmax
                    bbox_preds[i, :, 1] += h_off  # ymin
                    bbox_preds[i, :, 3] += h_off  # ymax

            print(f"Adjusted Bbox preds: {bbox_preds}")
            # Flatten all tiles
            bbox_preds = bbox_preds.view(-1, 4)
            class_preds = class_preds.view(-1, self.num_classes)

            # Take highest class probability per anchor
            scores, labels = class_preds[:, 1:].max(dim=1) # Ignore the background class

            # Move to CPU if needed
            bbox_preds = bbox_preds.cpu()
            scores = scores.cpu()
            labels = labels.cpu()
            
            print(f"Scores shape: {scores.shape}")
            print(f"Bbox shape: {bbox_preds.shape}")
            # Non-Maximum Suppression
            keep = nms(bbox_preds, scores, iou_threshold=0.7)
            
            # Final predictions
            final_boxes = bbox_preds[keep]
            final_scores = scores[keep]
            final_labels = labels[keep]

            # Pick the single best prediction
            best_idx = final_scores.argmax()
            print(f"\n\nBest IDX: {best_idx}\n\n")

            best_box = final_boxes[best_idx]
            best_score = final_scores[best_idx]
            best_label = final_labels[best_idx]

            return best_box, best_score, best_label


if __name__ == "__main__":

    # Testing
    image = Image.open("./data/raw/images/1.jpg")

    # To take a large input image and convert it into something the model can use.
    tiles, tile_coords = create_tiles_from_image(image, 256, 5)             # Create the tiles.            
    transform = transforms.ToTensor()                                       # Transform function to convert from a PIL image to a tensor.
    tile_batch = torch.stack([transform(img) for img in tiles], dim=0)      # Create a batch of images that can be passed through the model.

    print(f"Shape of the tile batch: {tile_batch.shape}")

    # Try to use the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_batch.to(device)
    model = WaldoSSD(backbone=MobileV2Backbone())
    model.eval()
    
    with torch.no_grad():  # no gradient needed
        box, score, label = model(
            image_batch = tile_batch,   # use image_tensor for small images
            tile_coords = tile_coords   # None if not tiled
        )

    print(f"best_box: {box}, best_pred: {score}, best_label: {label}")
    image = draw_boxes(image, [box], [label], [score])
    image.show()

