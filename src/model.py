'''
model.py
defines all three model architectures for our waldo detector
a tile image goes in and waldo predictions come out

we have three models:
  WaldoDetector: custom SSD with MobileNetV2 backbone (our baseline)
  WaldoDetectorPretrained: SSD300 VGG16 pretrained on COCO
  WaldoDetectorRetinaNet: RetinaNet ResNet50 FPN pretrained on COCO (our best)

the custom SSD flow (WaldoDetector):
  1. images come in as [N, 3, 256, 256] from the dataloader
  2. preprocess_batch adds HSV mask + sobel edges -> [N, 5, 256, 256]
  3. 1x1 conv squishes 5 channels back to 3 for the backbone
  4. mobilenet backbone extracts features -> [N, 64, 16, 16]
  5. five SSD heads predict at scales 16x16, 8x8, 4x4, 2x2, 1x1
  6. concatenate_all_heads combines into [N, 1364, 4] locs + [N, 1364, 2] cls

the pretrained models (SSD300 and RetinaNet) follow a similar flow but use
torchvisions internal detection pipeline instead of our custom heads

for ablation studies we added flags: use_hsv, use_sobel, use_pretrained, num_heads
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.detection import ssd300_vgg16, retinanet_resnet50_fpn_v2
from torchvision.models.detection.ssd import SSDClassificationHead

from pre_processing import preprocess_batch, apply_hsv_mask, apply_sobel_edge_detection
from ssd_block import SSDBlock
from anchors import concatenate_all_heads
from config import (
    BACKBONE_LAYERS, BACKBONE_CHANNELS,
    NUM_CLASSES, NUM_ANCHORS, SSD_KERNEL_SIZE,
    NUM_SSD_HEADS, MODEL_TYPE,
    PRETRAINED_SSD_INPUT_SIZE, RETINANET_INPUT_SIZE,
    IMAGENET_MEAN, IMAGENET_STD
)
from data_pipeline import get_dataloader


class WaldoDetector(nn.Module):

    def __init__(self, use_hsv=True, use_sobel=True, use_pretrained=True, num_heads=NUM_SSD_HEADS):
        super().__init__()

        # save these flags so forward knows what preprocessing to do
        self.use_hsv = use_hsv
        self.use_sobel = use_sobel

        # figure out how many input channels we have after preprocessing
        # starts at 3 for RGB, then we add 1 for HSV mask and 1 for sobel if enabled
        self.input_channels = 3
        if use_hsv:
            self.input_channels += 1 # HSV mask adds 1 channel
        if use_sobel:
            self.input_channels += 1 # sobel edges adds 1 channel

        # if we have more than 3 channels we need a 1x1 conv to squish them back to 3
        # because mobilenet was trained on 3 channel images and expects 3 channels
        # this is a learnable layer so the model figures out the best way to combine them
        # if preprocessing is completely off (just raw RGB) we skip this
        if self.input_channels > 3:
            self.channel_conv = nn.Conv2d(self.input_channels, 3, kernel_size=1)
        else:
            self.channel_conv = None # no need for this layer if we already have 3 channels

        # load mobilenet and grab only the first BACKBONE_LAYERS layers
        # we load it once here and reuse it for every batch instead of loading it fresh each time
        # use_pretrained controls whether we start with imagenet weights or random weights
        # this is useful for our ablation study where we test if transfer learning helps
        if use_pretrained:
            mobilenet = models.mobilenet_v2(weights='DEFAULT')
        else:
            mobilenet = models.mobilenet_v2(weights=None)

        self.backbone = mobilenet.features[:BACKBONE_LAYERS]

        # freeze the backbone weights so training doesnt mess up what mobilenet already learned
        # the backbone just extracts features, we dont want to change how it does that
        # gradients still flow through the backbone to reach the 1x1 conv
        # but the backbones own weights dont get updated
        for param in self.backbone.parameters():
            param.requires_grad = False

        # create the SSD detection heads using nn.ModuleList
        # we use ModuleList instead of a regular python list so pytorch knows about them
        # this way their weights get saved in checkpoints and moved to GPU properly
        # each head takes 64 channels from the backbone and predicts loc and cls
        self.ssd_heads = nn.ModuleList([
            SSDBlock(
                in_channels=BACKBONE_CHANNELS,
                num_classes=NUM_CLASSES,
                kernel_size=SSD_KERNEL_SIZE,
                num_anchors=NUM_ANCHORS
            )
            for _ in range(num_heads)
        ])

        # the feature sizes that each head will see
        # backbone outputs 16x16, each head halves it: 16, 8, 4, 2, 1
        # we need this list for concatenate_all_heads so it knows the grid sizes
        # if someone uses fewer heads for ablation we only take the first N sizes
        all_sizes = [16, 8, 4, 2, 1]
        self.feature_sizes = all_sizes[:num_heads]

    def forward(self, images):
        # images come in as [N, 3, 256, 256] from the dataloader

        # step 1: preprocessing
        # add extra channels based on which flags are enabled
        # if both HSV and sobel are on we get 5 channels (the default)
        # if only one is on we get 4 channels
        # if both are off we just pass the raw RGB through
        if self.use_hsv and self.use_sobel:
            # full preprocessing, this is what we normally use
            x = preprocess_batch(images) # [N, 5, 256, 256]
        elif self.use_hsv and not self.use_sobel:
            # only HSV mask, no sobel edges (for ablation study 1)
            hsv_mask = apply_hsv_mask(images)
            x = torch.cat([images, hsv_mask], dim=1) # [N, 4, 256, 256]
        elif not self.use_hsv and self.use_sobel:
            # only sobel edges, no HSV mask (for ablation study 1)
            # we reuse the existing apply_sobel_edge_detection function from pre_processing.py
            # instead of duplicating the sobel code here
            sobel = apply_sobel_edge_detection(images)
            x = torch.cat([images, sobel], dim=1) # [N, 4, 256, 256]
        else:
            # no preprocessing at all, just raw RGB (for ablation study)
            x = images # [N, 3, 256, 256]

        # step 2: if we added extra channels, squish them back to 3 for the backbone
        if self.channel_conv is not None:
            x = self.channel_conv(x) # [N, 5 or 4, 256, 256] -> [N, 3, 256, 256]

        # step 3: run through the mobilenet backbone to extract features
        x = self.backbone(x) # [N, 3, 256, 256] -> [N, 64, 16, 16]

        # step 4: pass through all SSD heads collecting predictions from each
        # each head predicts loc (bounding box offsets) and cls (waldo vs background)
        # then pools the feature map to half size for the next head
        all_locs = []
        all_cls = []

        for head in self.ssd_heads:
            loc, cls, x = head(x) # x gets pooled for the next head
            all_locs.append(loc)
            all_cls.append(cls)

        # step 5: reshape and combine predictions from all heads
        # concatenate_all_heads handles the reshaping from convolution format
        # to flat rows and generates the anchor positions for each head
        locs, cls_scores, anchors = concatenate_all_heads(
            all_locs, all_cls, self.feature_sizes
        )

        # locs: [N, 1364, 4] predicted box offsets for every anchor
        # cls_scores: [N, 1364, 2] waldo vs background scores for every anchor
        # anchors: [1364, 4] the actual anchor positions on the image
        # we always return the same thing whether training or evaluating
        # train.py uses these raw predictions to compute the loss
        # evaluate.py runs NMS on these to get the final filtered predictions
        return locs, cls_scores, anchors


class WaldoDetectorPretrained(nn.Module):
    # this model uses a full SSD300 VGG16 pretrained on COCO (330k images, 90 classes)
    # the detection heads already know how to find objects from COCO training
    # we just replace the classification head from 91 classes to 2 (waldo + background)
    # and add our HSV + Sobel preprocessing as extra input channels
    #
    # the big advantage over our custom WaldoDetector is that the detection heads
    # dont start from random weights, they already know how to match anchors to objects
    # and predict bounding box offsets. they just need to learn what waldo looks like
    #
    # torchvision SSD has a different interface than our custom model:
    #   training: takes (images_list, targets_list) and returns a loss dict
    #   eval: takes (images_list) and returns detection dicts with boxes, scores, labels
    # so train.py and evaluate.py need to handle both formats

    def __init__(self, use_hsv=True, use_sobel=True):
        super().__init__()

        self.use_hsv = use_hsv
        self.use_sobel = use_sobel
        self.model_type = "pretrained_ssd" # so checkpoint loading knows which model this is

        # figure out input channels for our preprocessing (same logic as WaldoDetector)
        self.input_channels = 3
        if use_hsv:
            self.input_channels += 1
        if use_sobel:
            self.input_channels += 1

        # 1x1 conv to merge our preprocessing channels back to 3 for the SSD backbone
        if self.input_channels > 3:
            self.channel_conv = nn.Conv2d(self.input_channels, 3, kernel_size=1)
        else:
            self.channel_conv = None

        # load the full SSD300 VGG16 pretrained on COCO
        # this gives us a complete object detector with backbone + detection heads + anchors + NMS
        # torchvision.models.detection.ssd300_vgg16 is from pytorch
        self.ssd = ssd300_vgg16(weights="DEFAULT")

        # replace the classification head from 91 COCO classes to 2 (waldo + background)
        # the box regression head stays the same (still predicts 4 offsets per anchor)
        # we need to match the number of anchors per feature map level
        num_anchors = self.ssd.anchor_generator.num_anchors_per_location()
        in_channels = [cls.in_channels for cls in self.ssd.head.classification_head.module_list]
        self.ssd.head.classification_head = SSDClassificationHead(in_channels, num_anchors, NUM_CLASSES)

        # set the minimum score threshold for detections during evaluation
        # by default torchvision returns everything above 0.01 which floods us with junk
        # setting it to 0.1 means only detections the model is at least 10% confident about survive
        self.ssd.score_thresh = 0.1

        # freeze the VGG16 backbone so it keeps its pretrained COCO features
        # we tried unfreezing it before but with only ~70 positive training tiles
        # the backbone just learned to fire on everything instead of learning
        # what waldo specifically looks like. the COCO features are already good
        # at extracting edges, textures, and shapes from images
        # those low level features work fine for cartoons too
        # we only want the detection heads to learn "what is waldo"
        # not the backbone to forget "how to see"
        for param in self.ssd.backbone.parameters():
            param.requires_grad = False

    def get_param_groups(self):
        # returns the trainable parameter groups
        # backbone is frozen so we only return head params and the 1x1 conv
        # train.py uses this to set up the optimizer
        head_params = list(self.ssd.head.parameters())
        # include the 1x1 conv since its also learning from scratch
        if self.channel_conv is not None:
            head_params += list(self.channel_conv.parameters())
        return head_params

    def forward(self, images, targets=None):
        # images is a batched tensor [N, 3, 256, 256] from our dataloader
        # the dataloader already applied ImageNet normalization
        # but torchvision SSD has its own internal normalization
        # so we need to undo ours first, then let the SSD apply its own
        # otherwise the images get double normalized which corrupts the input

        # step 1: apply our preprocessing (same as WaldoDetector)
        # preprocessing needs the ImageNet normalized input to work correctly
        # because it undoes normalization internally before computing HSV and Sobel
        if self.use_hsv and self.use_sobel:
            x = preprocess_batch(images)
        elif self.use_hsv and not self.use_sobel:
            hsv_mask = apply_hsv_mask(images)
            x = torch.cat([images, hsv_mask], dim=1)
        elif not self.use_hsv and self.use_sobel:
            sobel = apply_sobel_edge_detection(images)
            x = torch.cat([images, sobel], dim=1)
        else:
            x = images

        # step 2: merge extra channels back to 3
        # we used to have dropout here but it was messing things up
        # putting dropout before the backbone means the VGG16 batch norm layers
        # get noisy input during training but clean input during eval
        # that mismatch between train and eval was hurting more than helping
        # weight decay already handles regularization and the backbone was pretrained
        # on 330k COCO images so its pretty resistant to overfitting on its own
        if self.channel_conv is not None:
            x = self.channel_conv(x)

        # step 3: undo the ImageNet normalization from our dataloader
        # the torchvision SSD will apply its own normalization internally
        # (it uses different mean/std values than ImageNet)
        # if we dont undo ours the images get double normalized
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(x.device)
        x = (x * std + mean).clamp(0, 1) # back to [0, 1] range

        # step 4: scale from [0, 1] to [0, 255] because torchvision SSD expects uint8 range
        x = x * 255.0

        # step 5: resize from 256x256 to 300x300 (what SSD300 expects)
        x = F.interpolate(x, size=(PRETRAINED_SSD_INPUT_SIZE, PRETRAINED_SSD_INPUT_SIZE), mode="bilinear", align_corners=False)

        # step 6: torchvision SSD expects a list of images, not a batched tensor
        image_list = [x[i] for i in range(x.shape[0])]

        # step 7: pass through the SSD
        # the SSD will normalize the images internally using its own mean/std
        # during training: returns loss dict {"classification": ..., "bbox_regression": ...}
        # during eval: returns list of detection dicts with "boxes", "scores", "labels"
        if self.training and targets is not None:
            return self.ssd(image_list, targets)
        else:
            return self.ssd(image_list)


class WaldoDetectorRetinaNet(nn.Module):
    # RetinaNet with ResNet50 backbone and Feature Pyramid Network (FPN)
    # this is the natural evolution of SSD in the one stage detector family
    # the paper that introduced it (Lin et al, Focal Loss for Dense Object Detection,
    # ICCV 2017) literally invented focal loss to solve the class imbalance problem
    # which is exactly what we struggle with in wheres waldo
    #
    # key advantages over SSD300:
    #   1. Feature Pyramid Network: detects objects at multiple scales by combining
    #      features from different backbone layers. SSD only uses features from
    #      specific layers which means it can miss objects at certain sizes
    #   2. larger input: we feed 512x512 instead of SSD's 300x300
    #      more pixels means more context around waldo, which helps the model
    #      tell him apart from other characters in the cluttered scenes
    #   3. ResNet50 backbone: deeper and more powerful than VGG16
    #      better at extracting the fine details like waldos stripe pattern
    #   4. focal loss built in: the architecture was designed from the ground up
    #      to handle class imbalance. we dont need to manually add focal loss
    #      like we did for the pretrained SSD
    #
    # torchvision interface is the same as SSD:
    #   training: model(images, targets) returns loss dict
    #   eval: model(images) returns list of detection dicts

    def __init__(self, use_hsv=True, use_sobel=True):
        super().__init__()

        self.use_hsv = use_hsv
        self.use_sobel = use_sobel
        self.model_type = "retinanet"

        # figure out how many input channels we have after preprocessing
        self.input_channels = 3
        if use_hsv:
            self.input_channels += 1
        if use_sobel:
            self.input_channels += 1

        # 1x1 conv to merge our preprocessing channels back to 3 for the backbone
        # same approach as the pretrained SSD model
        if self.input_channels > 3:
            self.channel_conv = nn.Conv2d(self.input_channels, 3, kernel_size=1)
        else:
            self.channel_conv = None

        # load RetinaNet with ResNet50 FPN pretrained on COCO
        # retinanet_resnet50_fpn_v2 is the improved version with better defaults
        self.detector = retinanet_resnet50_fpn_v2(weights="DEFAULT")

        # replace the classification head from 91 COCO classes to 2 (waldo + background)
        # RetinaNet uses a different head structure than SSD
        # we need to find the number of anchors per location and the in_channels
        num_anchors = self.detector.head.classification_head.num_anchors
        in_channels = self.detector.head.classification_head.conv[0][0].in_channels

        # replace the classification head with one that predicts 2 classes
        # we rebuild it using the same architecture but with NUM_CLASSES outputs
        self.detector.head.classification_head.num_classes = NUM_CLASSES
        cls_logits = torch.nn.Conv2d(in_channels, num_anchors * NUM_CLASSES, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, -4.0) # bias init from RetinaNet paper
        self.detector.head.classification_head.cls_logits = cls_logits

        # set a reasonable score threshold for detections
        # too low and we get flooded with false positives
        # too high and we miss waldos
        self.detector.score_thresh = 0.3

        # freeze the ResNet50 backbone
        # same reason as SSD: with limited training data the backbone overfits
        # if we let it update. the COCO features are already good at extracting
        # edges, textures, and shapes which is what we need
        for param in self.detector.backbone.parameters():
            param.requires_grad = False

    def get_param_groups(self):
        # returns trainable params only (heads + 1x1 conv, backbone is frozen)
        head_params = list(self.detector.head.parameters())
        if self.channel_conv is not None:
            head_params += list(self.channel_conv.parameters())
        return head_params

    def forward(self, images, targets=None):
        # same preprocessing flow as the pretrained SSD
        # add HSV and Sobel channels, merge back to 3, undo normalization,
        # scale to 0-255, resize to the models expected input size

        # step 1: apply our preprocessing
        if self.use_hsv and self.use_sobel:
            x = preprocess_batch(images)
        elif self.use_hsv and not self.use_sobel:
            hsv_mask = apply_hsv_mask(images)
            x = torch.cat([images, hsv_mask], dim=1)
        elif not self.use_hsv and self.use_sobel:
            sobel = apply_sobel_edge_detection(images)
            x = torch.cat([images, sobel], dim=1)
        else:
            x = images

        # step 2: merge extra channels back to 3
        if self.channel_conv is not None:
            x = self.channel_conv(x)

        # step 3: undo ImageNet normalization from our dataloader
        # torchvision RetinaNet has its own internal normalization
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(x.device)
        x = (x * std + mean).clamp(0, 1)

        # step 4: scale to 0-255 range (torchvision expects this)
        x = x * 255.0

        # step 5: resize to RetinaNet's input size (512x512)
        # this is the big advantage over SSD300: we see ~3x more pixels
        # which means more context around waldo to help identify him
        x = F.interpolate(x, size=(RETINANET_INPUT_SIZE, RETINANET_INPUT_SIZE),
                          mode="bilinear", align_corners=False)

        # step 6: convert to image list format that torchvision expects
        image_list = [x[i] for i in range(x.shape[0])]

        # step 7: pass through RetinaNet
        # same interface as SSD: training returns loss dict, eval returns detections
        if self.training and targets is not None:
            return self.detector(image_list, targets)
        else:
            return self.detector(image_list)


def create_model(use_hsv=True, use_sobel=True, use_pretrained_backbone=True, num_heads=NUM_SSD_HEADS):
    # helper function that creates the right model based on MODEL_TYPE config
    # this way train.py and evaluate.py dont need to know which model class to use
    # they just call create_model() and get the right one

    if MODEL_TYPE == "retinanet":
        model = WaldoDetectorRetinaNet(use_hsv=use_hsv, use_sobel=use_sobel)
        model.model_type = "retinanet"
    elif MODEL_TYPE == "pretrained_ssd":
        model = WaldoDetectorPretrained(use_hsv=use_hsv, use_sobel=use_sobel)
        model.model_type = "pretrained_ssd"
    else:
        model = WaldoDetector(use_hsv=use_hsv, use_sobel=use_sobel, use_pretrained=use_pretrained_backbone, num_heads=num_heads)
        model.model_type = "custom_ssd"

    return model
