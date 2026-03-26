# config.py
# all the settings for the waldo detector live here
# instead of hunting through every file to change a number, just change it here
# every other file imports from this one

import os

# paths
# PROJECT_ROOT is computed so paths work no matter where you run from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_IMAGES = os.path.join(PROJECT_ROOT, "data/raw/images")
RAW_ANNOTATIONS = os.path.join(PROJECT_ROOT, "data/raw/annotations")
PROCESSED = os.path.join(PROJECT_ROOT, "data/processed")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# data pipeline
# controls how we chop up images and divide them into train/val/test
TILE_SIZES = [256] # we only generate 256x256 tiles, the model and evaluation both expect this size

# grid offsets for tiling
# instead of always starting the tile grid at (0,0) we also start at half-tile offsets
# this means waldo falls on different tile boundaries in each grid
# a waldo that was split across two tiles in the (0,0) grid might be fully inside
# one tile in the (128,0) grid. this gives us more diverse positive tiles
# without any image distortion at all
TILE_OFFSETS = [(0, 0), (128, 0), (0, 128), (128, 128)]
OVERLAP_THRESHOLD = 0.5 # fraction of waldo that must be inside a tile to count as positive
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15 # test ratio is whatever is left over (0.15)
RANDOM_SEED = 42 # so splits are reproducible
INPUT_SIZE = 256 # all tiles get resized to this before going into the backbone
BATCH_SIZE = 16

# imagenet normalization
# mobilenet was trained on imagenet so we normalize our images the same way
# these are used in data_pipeline.py and pre_processing.py
# if they dont match between files the model will get confused
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# pre processing thresholds
# these control what counts as "red" and "white" for waldos stripes
# the HSV mask uses these to highlight pixels that look like waldo
# might need tuning if the model isnt detecting well
RED_HUE_MAX = 15 # hue <= 15 counts as red (near 0 on the hue wheel)
RED_HUE_MIN = 345 # hue >= 345 also counts as red (near 360, wraps around)
RED_SAT_MIN = 0.4 # needs at least this much saturation to be red
RED_VAL_MIN = 0.3 # needs at least this much brightness to be red
WHITE_SAT_MAX = 0.15 # saturation below this = white
WHITE_VAL_MIN = 0.7 # brightness above this = white

# model architecture
# be careful changing BACKBONE_LAYERS, it changes the output channels and spatial size
# with 11 layers we get 64 channels at 16x16 from a 256x256 input
BACKBONE_LAYERS = 11 # first N layers of mobilenetv2 to use as feature extractor
BACKBONE_CHANNELS = 64 # output channels from the backbone (depends on BACKBONE_LAYERS)
NUM_CLASSES = 2 # waldo or not waldo
NUM_ANCHORS = 4 # anchor boxes per spatial position in each SSD head
NUM_SSD_HEADS = 5 # backbone gives 16x16, each head halves it: 16, 8, 4, 2, 1 thats 5 heads
SSD_KERNEL_SIZE = 3

# model selection
# we tried three different architectures during this project:
#
#   "custom_ssd": our own SSD with MobileNetV2 backbone and 5 custom detection heads
#     this was our first attempt based on the Lamb & Chuah strawberry paper
#     it struggled because the heads trained from scratch with too little data
#
#   "pretrained_ssd": SSD300 VGG16 pretrained on COCO from torchvision
#     better than custom because the detection heads already knew how to find objects
#     but limited to 300x300 input which means tiny tiles with no context
#
#   "retinanet": RetinaNet with ResNet50 + FPN pretrained on COCO from torchvision
#     our best performing model. handles larger input (800x800) so the model
#     sees more context around waldo. feature pyramid network detects at multiple
#     scales. focal loss is built into the architecture (from the RetinaNet paper)
#     this is the natural evolution of SSD in the one stage detector lineage
#
# change MODEL_TYPE to switch between them
# all three use our HSV + Sobel preprocessing which is our main contribution
MODEL_TYPE = "retinanet" # "custom_ssd", "pretrained_ssd", or "retinanet"

# backward compatible flag so existing code that checks USE_PRETRAINED_SSD still works
# during the migration to MODEL_TYPE we keep this around
# eventually all code should check MODEL_TYPE directly
USE_PRETRAINED_SSD = MODEL_TYPE == "pretrained_ssd"

# model specific input sizes
# SSD300 expects exactly 300x300, thats baked into its architecture
# RetinaNet can take any size. we use 512x512 which is a good balance:
#   still 2x bigger than SSD's 300x300 so we get more context
#   but not so big that we waste compute upscaling 256x256 tiles
#   (at 800x800 we were stretching each tile 3x which just adds blur)
#   512x512 means 2x upscale which is cleaner and trains way faster
PRETRAINED_SSD_INPUT_SIZE = 300
RETINANET_INPUT_SIZE = 512

# batch size depends on model because bigger models use more GPU memory
# retinanet at 512x512 can handle larger batches than at 800x800
if MODEL_TYPE == "retinanet":
    BATCH_SIZE = 16 # retinanet at 512x512 fits well in 11GB VRAM
elif MODEL_TYPE == "pretrained_ssd":
    BATCH_SIZE = 24 # SSD300 is lightweight
# else custom_ssd uses the default BATCH_SIZE = 16 from above

# data augmentation
# wheres waldo has a massive class imbalance problem
# most tiles are negative (just random background clutter) and only a handful
# actually contain waldo. without augmentation the model just learns to always
# predict "no waldo" and gets high accuracy while being completely useless
#
# we handle this with two strategies:
#   1. weighted sampling: positive tiles get sampled more often during training
#      the weight is computed from the actual class counts in the dataset
#      SAMPLING_BALANCE_FACTOR controls how aggressive the oversampling is
#      factor 10 means roughly 1 positive tile per 10 negatives in each batch
#   2. color jitter: each time the model sees a positive tile it looks slightly
#      different with random changes to brightness, contrast, and saturation
#      this stops the model from memorizing the exact pixel values of the
#      positive tiles and forces it to learn the actual visual pattern
#      we only jitter positive training tiles, not val/test
AUGMENT_TRAINING = True
SAMPLING_BALANCE_FACTOR = 10 if MODEL_TYPE in ("pretrained_ssd", "retinanet") else 5
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
FLIP_PROBABILITY = 0.5 # chance of flipping a training tile horizontally
BLUR_PROBABILITY = 0.2 # chance of applying gaussian blur to a training tile
BLUR_KERNEL_SIZE = 3 # size of the blur kernel
SHARPNESS_PROBABILITY = 0.2 # chance of randomly adjusting sharpness
SHARPNESS_FACTOR = 2 # how much to adjust sharpness by

# nms (non maximum suppression)
# these control how we filter the models predictions during evaluation
# 0.3 was letting literally everything through so we bumped it to 0.5
# the model gives positive tiles avg 0.95 confidence so 0.5 is still very safe
# at 0.5 we lose maybe 1 waldo tile but cut dozens of false positives
NMS_CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5 # if two boxes overlap more than this we keep only the better one

# hsv pre filter for evaluation
# before running a tile through the model we check if it has any red/white pixels
# using our HSV mask. tiles with zero red/white content cant possibly contain waldo
# since his whole thing is the red and white striped shirt
# this is basically a cheap first stage filter that eliminates obvious negatives
# without needing the expensive model inference
# the threshold is the minimum fraction of pixels that must be red or white
# 0.005 means at least 0.5% of the tile needs to have waldos colors
HSV_FILTER_THRESHOLD = 0.005

# test time augmentation (TTA)
# during evaluation we also run each tile through the model flipped horizontally
# and take the max confidence between original and flipped versions
# waldo looks the same facing left or right so this is safe
# if hes partially cut off at a tile edge the flipped version might catch him better
# doubles the inference time but we only have a few hundred test tiles so its fast
USE_TTA = True

# focal loss
# the standard cross entropy loss treats all negative tiles equally
# but most negative tiles are easy (blue sky, green grass, empty walls)
# focal loss down weights the loss contribution from tiles the model
# is already confident about and amplifies the loss from confusing tiles
# (other characters that look similar to waldo but arent)
# the formula is: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
# gamma controls how much to down weight easy examples
# gamma=0 is standard cross entropy, gamma=2 is the typical focal loss value
# alpha balances the importance of positive vs negative classes
# this is from the RetinaNet paper (Lin et al, ICCV 2017)
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25

# training
# for all pretrained models (SSD and RetinaNet) we freeze the backbone
# and only train the detection heads plus our 1x1 preprocessing conv
# we tried unfreezing the backbone but with limited positive training tiles
# it just learned to detect everything as waldo instead of learning what
# waldo specifically looks like. freezing keeps the pretrained features intact
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001 # L2 regularization to prevent overfitting
NUM_EPOCHS = 150 # 150 is plenty, early stopping will kick in around 80-120 if it converges

# learning rate scheduler
# if the validation loss stops improving for LR_SCHEDULER_PATIENCE epochs
# we cut the learning rate by LR_SCHEDULER_FACTOR (halve it)
# this helps the model converge when its stuck at a plateau
# torch.optim.lr_scheduler.ReduceLROnPlateau from pytorch handles this
LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.5

# early stopping
# if the validation loss hasnt improved for this many epochs in a row
# we stop training to save time since the model isnt learning anymore
# we used to have this at 50 but realized thats way too long
# with LR scheduler patience of 10, by epoch 25 the LR has already been
# halved twice so the model is barely learning at that point anyway
# no reason to keep going for another 25 epochs at a tiny LR
EARLY_STOPPING_PATIENCE = 25
