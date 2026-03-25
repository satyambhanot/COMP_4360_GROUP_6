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
TILE_SIZES = [64, 128, 256]
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

# pretrained SSD option
# when True we use a full SSD300 VGG16 pretrained on COCO (330k images, 90 classes)
# the detection heads already know how to find objects so they just need to learn what waldo looks like
# when False we use our custom WaldoDetector with MobileNetV2 backbone and SSD heads trained from scratch
# both use our HSV + Sobel preprocessing which is our main contribution
# we can compare both approaches in the ablation studies
USE_PRETRAINED_SSD = True
PRETRAINED_SSD_INPUT_SIZE = 300 # SSD300 expects 300x300 input, our tiles are 256x256 so we resize

# data augmentation
# we have a huge class imbalance problem (about 14000 negative tiles vs 56 positive)
# if we dont do anything about it the model will just learn to always say "no waldo"
# and get 99.6% accuracy while being completely useless at actually finding waldo
#
# to fix this we do two things:
#   1. weighted sampling: positive tiles get picked more often during training
#      we compute the weight dynamically based on the actual counts
#      and divide by SAMPLING_BALANCE_FACTOR so its not a full 50/50 split
#      a factor of 5 means roughly 1 positive tile for every 5 negatives
#   2. color jitter: each time the model sees a positive tile it looks slightly different
#      random changes to brightness, contrast, and saturation
#      this stops the model from just memorizing the exact 56 positive tiles
#      we only do this on positive training tiles, not on val/test
#      and we only change colors not positions so the bounding box stays the same
AUGMENT_TRAINING = True
SAMPLING_BALANCE_FACTOR = 5 # higher means less aggressive balancing
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
FLIP_PROBABILITY = 0.5 # chance of flipping a training tile horizontally
GRAYSCALE_PROBABILITY = 0.1 # chance of converting a training tile to grayscale
BLUR_PROBABILITY = 0.2 # chance of applying gaussian blur to a training tile
BLUR_KERNEL_SIZE = 3 # size of the blur kernel
SHARPNESS_PROBABILITY = 0.2 # chance of randomly adjusting sharpness
SHARPNESS_FACTOR = 2 # how much to adjust sharpness by

# nms (non maximum suppression)
# these control how we filter the models predictions during evaluation
NMS_CONFIDENCE_THRESHOLD = 0.5 # predictions below this confidence get thrown away
NMS_IOU_THRESHOLD = 0.5 # if two boxes overlap more than this we keep only the better one

# training
# will be used by train.py
# pretrained SSD needs a lower learning rate because its weights are already close to optimal
# too high a learning rate would destroy the pretrained features
# our custom model can use a higher rate since its heads start from random
LEARNING_RATE = 0.0001 if USE_PRETRAINED_SSD else 0.001
NUM_EPOCHS = 50
