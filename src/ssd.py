# SSD Block
import torch 
import torch.nn as nn

# Input follows: channels, classes, kernel size, number of anchors.
# Returns location predictions,  class predictions, and next feature maps. (loc_preds, cls_preds, next_features)
class ssd_block(nn.Module):
    def __init__(self, in_channels: int,  num_classes: int, kernel_size: int, num_anchors: int, device):
        super().__init__()

        # Predict bounding box offsets (x, y, w, h).
        self.loc = nn.Conv2d(
            in_channels = in_channels,
            out_channels = num_anchors * 4, # This is the number of feature maps (x, y, w, h).
            kernel_size = kernel_size,
            stride = 1,
            padding = 1,
            device = device
        )

        # Predict class probabilities (Waldo / background).
        self.cls = nn.Conv2d(
            in_channels,
            out_channels = num_anchors * num_classes,
            kernel_size = kernel_size,
            stride = 1,
            padding = 1,
            device = device
        )

        # Downsample the feature map.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        loc = self.loc(x)
        cls = self.cls(x)
        x = self.pool(x)

        return loc, cls, x

"""
# Create a feature map filled with random values.
# Shape follows: batches, channels, height, width.
feature_map = torch.randn(1, 256, 64, 64)

# Input follows: channels, classes, kernel size, number of anchors.
ssd = ssd_block(256, 2, 3, 4)

# Location Predictions, Class Predictions, Next Feature Maps.
loc_preds, cls_preds, next_features = ssd(feature_map)

print(feature_map.shape)
print(loc_preds.shape)
print(cls_preds.shape)
print(next_features.shape)
"""
