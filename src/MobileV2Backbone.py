import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms, models
from data_pipeline import get_dataloader

BATCH_SIZE = 10

class MobileV2Backbone:

    def PassBatchThroughMobileV2Backbone(self,batch):

        mobileNet = models.mobilenet_v2(weights='DEFAULT') #entire mobilenet model that will describe our input images

        featureExtractor = mobileNet.features[:11] #outputs 16*16 * 64 channels
        featureExtractor.eval() #ensures model uses pre-trained knowledge to extract features

        with(torch.no_grad()): #tells pytorch not to track history of gradient calculations
            output = featureExtractor(batch) #size 64 * 64 * 16

        # Note: 4 numbers in the images.shape array are: [Batch size N, Number of Channels C, Height H, Width W]

        return output


def CreateWaldoModel():
    loader = get_dataloader(split="train", tile_size=256, batch_size=BATCH_SIZE)
    images, labels, bboxes = next(iter(loader))

    backbone = MobileV2Backbone()
    output = backbone.PassBatchThroughMobileV2Backbone(images)
    print(f"input shape: {images.shape}")
    print(f"output shape: {output.shape}")


if __name__ == "__main__":
    CreateWaldoModel()
