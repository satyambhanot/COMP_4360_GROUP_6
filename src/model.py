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
import MobileV2Backbone
import ConcatenateAnchors
import config
from data_pipeline import get_dataloader
from ssd import ssd_block

import torch
import torch.nn as nn

from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection._utils import BoxCoder
import torchvision.ops as ops
from torchvision.ops import nms

def apply_nms(bbox_preds, class_preds, conf_threshold=0.3, iou_threshold=0.5):
    """
    bbox_preds: (N,4) tensor of predicted boxes
    class_preds: (N,num_classes) tensor of predicted class scores
    conf_threshold: minimum score to consider a box
    iou_threshold: NMS IoU threshold
    """
    final_boxes = []
    final_scores = []
    final_labels = []

    num_classes = class_preds.shape[1]

    for cls in range(1, num_classes):  # usually skip background class 0
        scores = class_preds[:, cls]
        mask = scores > conf_threshold  # filter low-confidence boxes
        if mask.sum() == 0:
            continue

        boxes = bbox_preds[mask]
        scores = scores[mask]

        keep = nms(boxes, scores, iou_threshold)

        final_boxes.append(boxes[keep])
        final_scores.append(scores[keep])
        final_labels.append(torch.full((len(keep),), cls, dtype=torch.int64, device=bbox_preds.device))

    if len(final_boxes) == 0:
        return torch.empty((0,4)), torch.empty((0,)), torch.empty((0,), dtype=torch.int64)

    final_boxes = torch.cat(final_boxes, dim=0)
    final_scores = torch.cat(final_scores, dim=0)
    final_labels = torch.cat(final_labels, dim=0)

    return final_boxes, final_scores, final_labels

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
            tile = image[h:h + tile_size, w:w + tile_size]

            # Save the tile and the their offsets in the original image
            tiles.append(tile)
            tile_coords.append((x, y))  # keep global offset

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
        is_training: bool = False
    ):
        super().__init__()
        # Backbone
        self.backbone = backbone

        # Anchor Setup
        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))   # Set up BoxCoder to calculate positioning/size deltas for ground truth boxes relative to the anchors. Weights are used to scale regression targets (x, y, w, h).
        self.aspect_ratios = [[1.0, 2.0, 0.5]] * num_heads          # Configurate the anchors (Square box, wide rectangle box, tall rectangle box).
        self.anchor_generator = DefaultBoxGenerator(                # Creates the anchors
            aspect_ratios = aspect_ratios,                              # Use the aspect ratios for the boxes
            min_ratio = 0.07,                                           # The smallest an aspect ratio can be scaled relative to the image size
            max_ratio = 0.9                                             # The maximum amount an aspect ration can be scaled relative to the image size
        )

        # SSD Heads
        self.blocks = nn.ModuleList([
            ssd_block(
                in_channels = in_channels,
                num_classes = num_classes,
                kernel_size = kernel_size,
                num_anchors = num_anchors
            )
            for _ in range(num_heads)
        ])

    def forward(self, image):

        offset_preds = []
        class_preds = []
        feature_maps = []

        image_batch = None
        
        # For evaluation
        tile_coords = None
        tiles = None

        # To take a large input image and convert it into something the model can use.
        tile_coords, tiles = create_tiles_from_image(image)     # Create the tiles.            
        transform = transforms.toTensor()                       # Transform function to convert from a PIL image to a tensor.
        image_batch = [transform(img) for img in tiles, dim=0]  # Create a batch of images that can be passed through the model.
        
        processed_batch = preprocess_batch(image_batch)         # Preprocess the images by applying HSV and the Sobol edge detector.
        feature_map = self.backbone(processed_batch)            # Pass the preprocessed images to the backbone.
        
        # Pass the feature maps generated by the backbone to the SSD heads.
        for block in self.blocks:
            offset_pred, class_pred, next_feature_map = block(feature_maps) # Get preditions from the block.

            offset_preds.append(offset_pred)    # Save the offset predictions for NMS / training.
            class_preds.append(class_pred)      # Save the class prediction for NMS / training.
            feature_map = next_feature_map      # Setup the feature maps for the next SSD block. 

        offset_preds = torch.cat(offset_preds, dim=1)  # (batch, total_anchors, 4)
        class_preds  = torch.cat(class_preds, dim=1)   # (batch, total_anchors, num_classes)

        anchors = self.anchor_generator(image_batch, feature_maps)  # depends on your implementation
        
        # Output changes if its training or not
        if self.training:
            return offset_preds, class_preds, anchors

        if not self.training:
            # Get the bounding box predicitions
            bbox_preds = self.box_coder.decode(location_preds, anchors)

            # Adjust for tile positions
            for i, (w_off, h_off) in enumerate(tile_coords):
                bbox_preds[i, :, 0] += w_off  # Min width
                bbox_preds[i, :, 2] += w_off  # Max width
                bbox_preds[i, :, 1] += h_off  # Min height
                bbox_preds[i, :, 3] += h_off  # Max height
    
        # Flatten all tile predictions into one batch
        bbox_preds = bbox_preds.view(-1, 4)
        class_preds = class_preds.view(-1, class_preds.size(-1))

        # Non-Maximum Suppression
        final_bbox_pred, final_bbox_score, final_label = apply_nms(bbox_preds, class_preds)
        
        return final_boxes, final_scores, final_labels

if __name__ == "__main__":

    # Training Example
    # Generate Dataset
    loader = get_dataloader()
    images, labels, bboxes = next(iter(loader))

    # Pass through the model
    pred_boxed = WaldoSSD(images)

    print(pred_boxes)

