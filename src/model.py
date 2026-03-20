# model.py
# this is where we wire everything together into one full model
# the idea is: image tile goes in, waldo predictions come out
#
# the flow goes like this:
#   1. tile image [N, 3, 256, 256] comes from data_pipeline
#   2. pre_processing runs HSV mask + sobel edge detection on it
#   3. MobileV2Backbone extracts features [N, 64, 16, 16]
#   4. 6 SSD heads run on the features at different scales, each one predicts bounding boxes + class scores
#   5. ConcatenateAnchors collects all predictions from all 6 heads
#   6. NMS (non-maximum suppression) filters out overlapping boxes, keeps the best ones
#
# right now we have the individual pieces built:
#   data_pipeline.py (done): loads tiles and labels
#   pre_processing.py (done): HSV mask and sobel edges
#   MobileV2Backbone.py (done): feature extraction
#   ssd.py (done): single SSD block
#   ConcatenateAnchors.py (needs work): just a stub right now
#
# TODO: wire backbone -> ssd heads -> anchor concat -> nms into one nn.Module
# TODO: figure out how pre_processing fits in (extra input channels? or separate scoring?)
# TODO: handle the channel mismatch (backbone outputs 64 channels, need to decide SSD input channels)
