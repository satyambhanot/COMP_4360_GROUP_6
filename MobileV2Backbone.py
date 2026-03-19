import torch
import matplotlib.pyplot as plt
from ImageDataLoader import ImageDataLoader
from PIL import Image
from torchvision import datasets, transforms, models
DATA_PATH = "./Dataset/Hey-Waldo-master/128"
BATCH_SIZE = 10

class MobileV2Backbone:

    def PassBatchThroughMobileV2Backbone(self,batch):

        mobileNet = models.mobilenet_v2(weights='DEFAULT') #entire mobilenet model that will describe our input images

        featureExtractor = mobileNet.features[:11] #outputs 16*16 * 64 channels
        featureExtractor.eval() #ensures model uses pre-trained knowledge to extract features

        data_iterator = iter(batch)

        images, labels = next(data_iterator) # grab 1st batch

        with(torch.no_grad()): #tells pytorch not to track history of gradient calculations
            output = featureExtractor(images) #size 64 * 64 * 16

        # Note: 4 numbers in the images.shape array are: [Batch size N, Number of Channels C, Height H, Width W]   

        return output


def CreateWaldoModel():
    loader = ImageDataLoader(10,"./Dataset/Hey-Waldo-master/128")
    dataset = loader.LoadDataset()
    batch = loader.LoadBatch(dataset)

    backbone = MobileV2Backbone()
    backbone.PassBatchThroughMobileV2Backbone(batch)


CreateWaldoModel()

## Steps:

## Step 1: Import images from dataset

## Step 2: Apply HSV Masking

## Step 3: Apply sobel edge detection

## Step 4: Apply slice and order ROI

## Step 5: Implement MobileNet V2 Backbone (Feature extractor)

## Step 6: Implement SSD (Single shot detector) layers

## Step 7: Concatenate anchors

## Step 8: Apply non-maximum suppresion




