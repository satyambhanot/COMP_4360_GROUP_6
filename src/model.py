# model.py
# this is the main model file where we connect all the pieces together
# a tile image goes in and waldo predictions come out
#
# i was confused about how many SSD heads to use because config said 6
# but when i traced through the sizes it only works with 5
# the backbone gives us 16x16 and each head cuts it in half
# 16 -> 8 -> 4 -> 2 -> 1, thats 5 heads
# after that its 1x1 and theres nothing left to halve
#
# i also wasnt sure if we should reuse MobileV2Backbone.py or load
# mobilenet directly here. the problem with MobileV2Backbone.py is that
# it loads the entire model fresh every time you call it which is slow
# so in this file we load it once in __init__ and reuse it for every batch
#
# the full flow through the model:
#
#   1. images come in as [N, 3, 256, 256] from the dataloader
#
#   2. preprocess_batch adds the HSV mask and sobel edges
#      now we have [N, 5, 256, 256] (3 RGB + 1 HSV + 1 Sobel)
#
#   3. but mobilenet expects 3 channels not 5
#      so we use a 1x1 conv to squish 5 channels back down to 3
#      this is a learnable layer so the model figures out
#      the best way to combine those 5 channels
#      now we have [N, 3, 256, 256] again
#
#   4. run it through the mobilenet backbone (first 11 layers)
#      this extracts features and shrinks the spatial size
#      now we have [N, 64, 16, 16]
#
#   5. pass the features through 5 SSD heads one after another
#      each head does two things:
#        predicts bounding box offsets (loc) for its anchors
#        predicts class scores (cls) for waldo vs background
#      then it pools the feature map to half the size for the next head
#
#      head 1: 16x16 -> predicts 1024 anchors -> pools to 8x8
#      head 2: 8x8   -> predicts 256 anchors  -> pools to 4x4
#      head 3: 4x4   -> predicts 64 anchors   -> pools to 2x2
#      head 4: 2x2   -> predicts 16 anchors   -> pools to 1x1
#      head 5: 1x1   -> predicts 4 anchors    -> pools to 0x0 (we ignore this)
#
#   6. concatenate_all_heads takes all 5 heads worth of predictions
#      and stacks them into one big tensor
#      total anchors: 1024 + 256 + 64 + 16 + 4 = 1364
#      loc output: [N, 1364, 4]
#      cls output: [N, 1364, 2]
#
# things i need to be careful about:
#   the backbone weights are frozen so we dont mess up what mobilenet learned
#   the ssd heads go in nn.ModuleList not a regular python list
#     otherwise pytorch wont track them and they wont save in checkpoints
#   the last head pools to 0x0 but we dont care because we already got
#     the loc and cls predictions before the pooling happens
#
# whats trainable and whats frozen:
#   frozen (we dont touch these weights):
#     the mobilenet backbone, it already knows how to extract features
#   trainable (these learn during training):
#     the 1x1 conv that merges 5 channels to 3
#     all 5 SSD heads (the loc and cls conv layers inside each one)
#   not a layer at all (just a function, no weights):
#     preprocess_batch, it just does math on the pixels
#     concatenate_all_heads, it just reshapes tensors

# TODO: add imports
# TODO: build the WaldoDetector class with __init__ and forward
# TODO: add a test at the bottom
