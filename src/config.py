# config.py
# all the settings for the waldo detector live here
# instead of hunting through every file to change a number, just change it here
# every other file imports from this one

# --- paths ---
# where our raw data and processed tiles live
# PROJECT_ROOT is computed so paths work no matter where you run from
# TODO: add path constants here

# --- data pipeline ---
# tile sizes, overlap threshold, train/val/test split ratios
# these control how we chop up images and divide them into sets
# TODO: add data pipeline constants here

# --- imagenet normalization ---
# mobilenet was trained on imagenet so we need to normalize our images the same way
# these exact numbers are used in data_pipeline.py and pre_processing.py
# if they dont match between files the model will get confused
# TODO: add IMAGENET_MEAN and IMAGENET_STD here

# --- pre-processing thresholds ---
# these control what counts as "red" and "white" for waldos stripes
# the HSV mask uses these to highlight pixels that look like waldo
# might need tuning if detection is poor
# TODO: add HSV and white mask thresholds here

# --- model architecture ---
# backbone layers, number of SSD heads, anchors per head, etc
# changing BACKBONE_LAYERS changes the output channels and spatial size
# so be careful with that one
# TODO: add model architecture constants here

# --- training ---
# learning rate, epochs, etc
# will be used by train.py once we build it
# TODO: add training constants here
