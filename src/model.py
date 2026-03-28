# model.py
'''
this is the main model file where we connect all the pieces together
a tile image goes in and waldo predictions come out


but when i traced through the sizes it only works with 5
the backbone gives us 16x16 and each head cuts it in half
16 -> 8 -> 4 -> 2 -> 1, thats 5 heads
after that its 1x1 and theres nothing left to halve

the full flow through the model:

  1. images come in as [N, 3, 256, 256] from the dataloader
  2. preprocess_batch adds the HSV mask and sobel edges of the HSV mask
     now we have [N, 5, 256, 256] (3 RGB + 1 HSV + 1 Sobel of HSV)
      but this is configurable, we can turn off HSV or sobel for ablation studies
  3. mobilenet expects 3 channels not 5
      so we use a 1x1 conv to squish the extra channels back down to 3
      this is a learnable layer so the model figures out
      the best way to combine those channels
      if preprocessing is off we skip this step entirely

  4. run it through the mobilenet backbone (first 11 layers)
      this extracts features and shrinks the spatial size
      now we have [N, 64, 16, 16]

   5. pass the features through 5 SSD heads one after another
      each head predicts bounding box offsets and class scores
      then pools the feature map to half the size for the next head

      head 1: 16x16 -> 1024 anchors -> pools to 8x8
      head 2: 8x8   -> 256 anchors  -> pools to 4x4
      head 3: 4x4   -> 64 anchors   -> pools to 2x2
      head 4: 2x2   -> 16 anchors   -> pools to 1x1
      head 5: 1x1   -> 4 anchors    -> pools to 0x0 (we ignore this)
   6. concatenate_all_heads combines everything into final tensors
      total anchors: 1024 + 256 + 64 + 16 + 4 = 1364
      loc output: [N, 1364, 4]
      cls output: [N, 1364, 2]

 whats trainable and whats frozen:
   frozen (we dont update these weights):
     the mobilenet backbone, it already knows how to extract features from imagenet
   trainable (these learn during training):
     the 1x1 conv that merges the extra channels down to 3
     all 5 SSD heads (the loc and cls conv layers inside each one)
   not a layer at all (just a function with no weights):
     preprocess_batch, it just does math on the pixels
     concatenate_all_heads, it just reshapes tensors

 for ablation studies we added flags to turn parts on and off:
   use_hsv: whether to include the HSV red/white mask channel
   use_sobel: whether to include the sobel edge channel
   use_pretrained: whether to load imagenet weights or start from random
   num_heads: how many SSD heads to use (default 5, can try 3 for model size ablation)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.detection import ssd300_vgg16, retinanet_resnet50_fpn_v2
from torchvision.models.detection.ssd import SSDClassificationHead

from pre_processing import preprocess_batch, ApplyHSVMask, ApplySobelEdgeDetection
from ssd_block import ssd_block
from BaseModel import BaseModel
from anchors import concatenate_all_heads
from config import *

from data_pipeline import get_dataloader


class WaldoDetector(nn.Module,BaseModel):

    def __init__(self, use_hsv=True, use_sobel=True, use_pretrained=True, num_heads=NUM_SSD_HEADS):
        nn.Module.__init__(self)
        BaseModel.__init__(use_hsv,use_sobel,"custom_ssd",3)

        if use_pretrained:
            mobilenet = models.mobilenet_v2(weights='DEFAULT')
        else:
            mobilenet = models.mobilenet_v2(weights=None)

        self.backbone = mobilenet.features[:BACKBONE_LAYERS]

        # freeze the backbone weights so training doesnt mess up what mobilenet already learned
      
        for param in self.backbone.parameters():
            param.requires_grad = False

        # create the SSD detection heads using nn.ModuleList
        self.ssd_heads = nn.ModuleList([
            ssd_block(
                in_channels=BACKBONE_CHANNELS,
                num_classes=NUM_CLASSES,
                kernel_size=SSD_KERNEL_SIZE,
                num_anchors=NUM_ANCHORS
            )
            for _ in range(num_heads)
        ])

        # the feature sizes that each head will see
        # backbone outputs 16x16, each head halves it: 16, 8, 4, 2, 1
        all_sizes = [16, 8, 4, 2, 1]
        self.feature_sizes = all_sizes[:num_heads]

    def ProcessImages(self,images):
         # add extra channels based on which flags are enabled
        if self.use_hsv and self.use_sobel:
            x = preprocess_batch(images) # [N, 5, 256, 256]
        elif self.use_hsv and not self.use_sobel:
            hsv_mask = ApplyHSVMask(images)
            x = torch.cat([images, hsv_mask], dim=1) # [N, 4, 256, 256]
        elif not self.use_hsv and self.use_sobel:
            sobel = ApplySobelEdgeDetection(images)
            x = torch.cat([images, sobel], dim=1) # [N, 4, 256, 256]
        else:
            x = images # [N, 3, 256, 256]

        if self.channel_conv is not None:
            x = self.channel_conv(x) # [N, 5 or 4, 256, 256] -> [N, 3, 256, 256]

        x = self.backbone(x) # [N, 3, 256, 256] -> [N, 64, 16, 16]
        return x
    
     # images come in as [N, 3, 256, 256] from the dataloader
    def forward(self, images):

        x = self.ProcessImages(images)

        all_locs = []
        all_cls = []

        for head in self.ssd_heads:
            loc, cls, x = head(x) # x gets pooled for the next head
            all_locs.append(loc)
            all_cls.append(cls)

        # concatenate_all_heads reshapes and combines predictions from all heads
        locs, cls_scores, anchors = concatenate_all_heads(
            all_locs, all_cls, self.feature_sizes
        )

        '''
        locs: [N, 1364, 4] predicted box offsets for every anchor
        cls_scores: [N, 1364, 2] waldo vs background scores for every anchor
        anchors: [1364, 4] the actual anchor positions on the image
        we always return the same thing whether training or evaluating
        train.py uses these raw predictions to compute the loss
        evaluate.py runs NMS on these to get the final filtered predictions
        '''
        return locs, cls_scores, anchors


"""
This model uses a full SSD300 VGG16 pretrained on COCO (330k images, 90 classes)
The detection heads already know how to find objects from COCO training
We just replace the classification head from 91 classes to 2 (waldo + background) and add our HSV + Sobel preprocessing as extra input channels
The big advantage over our custom WaldoDetector is that the detection heads dont start from random weights, they already know how to match anchors to objects and predict bounding box offsets. they just need to learn what waldo looks like
torchvision SSD has a different interface than our custom model:
training: takes (images_list, targets_list) and returns a loss dict
eval: takes (images_list) and returns detection dicts with boxes, scores, labels so train.py and evaluate.py need to handle both formats
"""
class WaldoDetectorPretrained(nn.Module,BaseModel):
 

    def __init__(self, use_hsv=True, use_sobel=True):
        nn.Module.__init__(self)
        BaseModel.__init__(use_hsv,use_sobel,"pretrained_ssd",3)

        # load the full SSD300 VGG16 pretrained on COCO
        # this gives us a complete object detector with backbone + detection heads + anchors + NMS
        self.ssd = ssd300_vgg16(weights="DEFAULT")

        '''
        # replace the classification head from 91 COCO classes to 2 (waldo + background)
        # the box regression head stays the same (still predicts 4 offsets per anchor)
        # we need to match the number of anchors per feature map level
        '''
        num_anchors = self.ssd.anchor_generator.num_anchors_per_location()
        in_channels = [cls.in_channels for cls in self.ssd.head.classification_head.module_list]
        self.ssd.head.classification_head = SSDClassificationHead(in_channels, num_anchors, NUM_CLASSES)

        # set the minimum score threshold for detections during evaluation
        self.ssd.score_thresh = 0.1

        # freeze the VGG16 backbone so it keeps its pretrained COCO features
        for param in self.ssd.backbone.parameters():
            param.requires_grad = False

    '''
    # returns the trainable parameter groups
    '''
    def get_param_groups(self):
        
        head_params = list(self.ssd.head.parameters())

        if self.channel_conv is not None:
            head_params += list(self.channel_conv.parameters())
        return head_params

    def ApplyPreprocessingToImages(self,images):
        '''
        preprocessing needs the ImageNet normalized input to work correctly because it undoes normalization internally before computing HSV and Sobel
        '''
        if self.use_hsv and self.use_sobel:
            x = preprocess_batch(images)
        elif self.use_hsv and not self.use_sobel:
            hsv_mask = ApplyHSVMask(images)
            x = torch.cat([images, hsv_mask], dim=1)
        elif not self.use_hsv and self.use_sobel:
            sobel = ApplySobelEdgeDetection(images)
            x = torch.cat([images, sobel], dim=1)
        else:
            x = images

        if self.channel_conv is not None:
            x = self.channel_conv(x)
        return x
    
    def ProcessImages(self,images):
        
        x = self.ApplyPreprocessingToImages(images)

        '''
        undo the ImageNet normalization from our dataloader
        the torchvision SSD will apply its own normalization internally
        (it uses different mean/std values than ImageNet)
        if we dont undo ours the images get double normalized
        '''
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(x.device)
        x = (x * std + mean).clamp(0, 1) # back to [0, 1] range
      
        x = x * 255.0 # scale from [0, 1] to [0, 255] 

        # step 5: resize from 256x256 to 300x300 (what SSD300 expects)
        x = F.interpolate(x, size=(PRETRAINED_SSD_INPUT_SIZE, PRETRAINED_SSD_INPUT_SIZE), mode="bilinear", align_corners=False)

        return x
    
    '''
    images is a batched tensor [N, 3, 256, 256] from our dataloader
    the dataloader already applied ImageNet normalization
    but torchvision SSD has its own internal normalization
    so we need to undo ours first, then let the SSD apply its own
    otherwise the images get double normalized which corrupts the input
    '''
    def forward(self, images, targets=None):
       
        x = self.ProcessImages(images)
  
        image_list = [x[i] for i in range(x.shape[0])]

        if self.training and targets is not None:
            return self.ssd(image_list, targets)
        else:
            return self.ssd(image_list)

'''
    RetinaNet with ResNet50 backbone and Feature Pyramid Network (FPN)
    uses focal loss to solve the class imbalance problem
    which is exactly what we struggle with in wheres waldo
    key advantages over SSD300:
    1. Feature Pyramid Network: detects objects at multiple scales by combining
    features from different backbone layers. SSD only uses features from
    specific layers which means it can miss objects at certain sizes
    2. larger input: we feed 512x512 instead of SSD's 300x300
    more pixels means more context around waldo, which helps the model
    tell him apart from other characters in the cluttered scenes
    3. ResNet50 backbone: deeper and more powerful than VGG16
    better at extracting the fine details like waldos stripe pattern
    4. focal loss built in: the architecture was designed from the ground up to handle class imbalance. we dont need to manually add focal loss like we did for the pretrained SSD
    torchvision interface is the same as SSD:
    training: model(images, targets) returns loss dict
    eval: model(images) returns list of detection dicts
'''
class WaldoDetectorRetinaNet(nn.Module,BaseModel):

    def __init__(self, use_hsv=True, use_sobel=True):
        nn.Module.__init__(self)
        BaseModel.__init__(self,use_hsv,use_sobel,"retinanet",3)

        # load RetinaNet with ResNet50V2 FPN pretrained on COCO dataset
        self.detector = retinanet_resnet50_fpn_v2(weights="DEFAULT")

        '''
        replace the classification head from 91 COCO classes to 2 (waldo + background)
        RetinaNet uses a different head structure than SSD
        we need to find the number of anchors per location and the in_channels
        '''
        num_anchors = self.detector.head.classification_head.num_anchors
        in_channels = self.detector.head.classification_head.conv[0][0].in_channels

        # replace the classification head with one that predicts 2 classes
        self.detector.head.classification_head.num_classes = NUM_CLASSES
        cls_logits = torch.nn.Conv2d(in_channels, num_anchors * NUM_CLASSES, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, -4.0) # bias init from RetinaNet paper
        self.detector.head.classification_head.cls_logits = cls_logits

        '''
        set a reasonable score threshold for detections
        too low and we get flooded with false positives
        too high and we miss waldos
        '''
        self.detector.score_thresh = 0.3

        # freeze the ResNet50 backbone
        for param in self.detector.backbone.parameters():
            param.requires_grad = False

    '''
    # returns trainable params only (heads + 1x1 conv, backbone is frozen)
    '''
    def get_param_groups(self):
        head_params = list(self.detector.head.parameters())
        if self.channel_conv is not None:
            head_params += list(self.channel_conv.parameters())
        return head_params

    def ApplyPreprocessingToImages(self,images):
        if self.use_hsv and self.use_sobel:
            x = preprocess_batch(images)
        elif self.use_hsv and not self.use_sobel:
            hsv_mask = ApplyHSVMask(images)
            x = torch.cat([images, hsv_mask], dim=1)
        elif not self.use_hsv and self.use_sobel:
            sobel = ApplySobelEdgeDetection(images)
            x = torch.cat([images, sobel], dim=1)
        else:
            x = images

        # step 2: merge extra channels back to 3
        if self.channel_conv is not None:
            x = self.channel_conv(x)

        return x
    
    def ProcessImages(self,images):
        
        x = self.ApplyPreprocessingToImages(images)
       
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(x.device)
        x = (x * std + mean).clamp(0, 1)

        x = x * 255.0

        x = F.interpolate(x, size=(RETINANET_INPUT_SIZE, RETINANET_INPUT_SIZE),mode="bilinear", align_corners=False)

        return x
    
    def forward(self, images, targets=None):
       
        x = self.ProcessImages(images)

        # convert to image list format that torchvision expects
        image_list = [x[i] for i in range(x.shape[0])]

        # pass through RetinaNet
        if self.training and targets is not None:
            return self.detector(image_list, targets)
        else:
            return self.detector(image_list)



if __name__ == "__main__":
    # quick test to make sure all the pieces connect properly
    # load a small batch of real tiles and run them through the model

    print("loading a batch of tiles...")
    loader = get_dataloader(split="train", tile_size=256, batch_size=4)
    images, labels, bboxes = next(iter(loader))
    print(f"input images: {images.shape}") # should be [4, 3, 256, 256]

    # test the full model with all preprocessing on (default)
    print("\ntesting full model (HSV + Sobel)...")
    model = WaldoDetector(use_hsv=True, use_sobel=True)
    model.eval()
    with torch.no_grad():
        locs, cls_scores, anchors = model(images)
    print(f"  loc predictions: {locs.shape}") # [4, 1364, 4]
    print(f"  cls predictions: {cls_scores.shape}") # [4, 1364, 2]
    print(f"  anchors: {anchors.shape}") # [1364, 4]
    print(f"  total anchors: {anchors.shape[0]}")

    # test with no preprocessing (for ablation comparison)
    print("\ntesting model with no preprocessing (raw RGB only)...")
    model_raw = WaldoDetector(use_hsv=False, use_sobel=False)
    model_raw.eval()
    with torch.no_grad():
        locs2, cls2, anchors2 = model_raw(images)
    print(f"  loc predictions: {locs2.shape}") # [4, 1364, 4]
    print(f"  cls predictions: {cls2.shape}") # [4, 1364, 2]

    # test with fewer heads (for model size ablation)
    print("\ntesting model with 3 SSD heads...")
    model_small = WaldoDetector(num_heads=3)
    model_small.eval()
    with torch.no_grad():
        locs3, cls3, anchors3 = model_small(images)
    print(f"  loc predictions: {locs3.shape}") # should have fewer anchors
    print(f"  cls predictions: {cls3.shape}")
    print(f"  anchors: {anchors3.shape}")
    # 3 heads: 16x16 + 8x8 + 4x4 = 1024 + 256 + 64 = 1344 anchors
    print(f"  total anchors: {anchors3.shape[0]}")
