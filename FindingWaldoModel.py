import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
DATA_PATH = "./Dataset/Hey-Waldo-master/128"
BATCH_SIZE = 10

def LoadDataset():
    data_path = DATA_PATH #path to specific folder (i.e: 128 or 128-gray)

    preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]) #normalize/preprocess for images

    dataset = datasets.ImageFolder(root = data_path, transform=preprocess) #automatically assigns label for waldo and notwaldo folders (waldo = 1, notwaldo = 0)

    return dataset

def LoadBatch(dataset):
    load_data = DataLoader(dataset,batch_size = BATCH_SIZE, shuffle=True) 

    return load_data

def PassBatchThroughMobileV2Backbone(batch):

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
    dataset = LoadDataset()
    batch = LoadBatch(dataset)

    PassBatchThroughMobileV2Backbone(batch)


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




