from abc import abstractmethod
import torch.nn as nn
class BaseModel:
    def __init__(self,hsv,sobel,type,num_channels):
        self.use_hsv = hsv
        self.use_sobel = sobel
        self.model_type = type
        self.input_channels = num_channels
        if self.use_hsv:
            self.input_channels += 1 # HSV mask adds 1 channel
        if self.use_sobel:
            self.input_channels += 1 # sobel edges adds 1 channel
        if self.input_channels > 3:
            self.channel_conv = nn.Conv2d(self.input_channels, 3, kernel_size=1)
        else:
            self.channel_conv = None # no need for this layer if we already have 3 channels


    @abstractmethod
    def forward(self, images, targets=None):
        pass