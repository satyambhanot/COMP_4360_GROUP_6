# SSD Block
import torch 
import torch.nn as nn

# Input follows: channels, classes, kernel size, number of anchors.
# Returns location predictions,  class predictions, and next feature maps. (loc_preds, cls_preds, next_features)
class ssd_block(nn.Module):

    def __init__(self, in_channels: int,  num_classes: int, kernel_size: int, num_anchors: int):
        super().__init__()

        # Predict bounding box offsets (x, y, w, h).
        self.loc = nn.Conv2d(
            in_channels,
            num_anchors * 4, # This is the number of feature maps (x, y, w, h).
            kernel_size,
            padding=1
        )

        # Predict class probabilities (Waldo / background).
        self.cls = nn.Conv2d(
            in_channels,
            num_anchors * num_classes,
            kernel_size,
            padding=1
        )

        # Downsample the feature map.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        loc = self.loc(x)
        cls = self.cls(x)

        # only pool if the feature map is bigger than 1x1
        # the last SSD head gets a 1x1 input and pooling that would give 0x0 which crashes
        # we still return loc and cls from the 1x1 head, we just skip the pooling
        if x.shape[2] > 1 and x.shape[3] > 1:
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
