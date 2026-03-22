# nms.py
# non maximum suppression
# after the model makes predictions, we end up with 1364 anchor predictions
# most of them will say "no waldo here" with low confidence
# but for the spots where waldo actually is, multiple nearby anchors
# will all say "waldo is here!" with high confidence
# we dont want 10 boxes around the same spot, we just want the best one
#
# how it works step by step:
#
#   1. first we throw away any prediction with a confidence score
#      below a threshold (like 0.5). most of the 1364 anchors will have
#      very low confidence so this gets rid of the obvious noise
#
#   2. for the remaining predictions, we sort them by confidence
#      highest confidence first
#
#   3. we take the top prediction (highest confidence) and keep it
#      then we check all the other remaining predictions to see
#      if they overlap with it too much (using IoU, intersection over union)
#      if they overlap more than our threshold (like 0.5) we throw them away
#      because they are probably detecting the same waldo
#
#   4. then we move to the next highest confidence prediction that
#      wasnt thrown away, and repeat step 3
#
#   5. keep going until we have checked everything
#      what we have left is one clean box per detected waldo
#
# this only runs during inference (when we are actually looking for waldo)
# during training we need all 1364 predictions to compute the loss
# so we skip nms during training
#
# we will use torchvision.ops.nms which handles the sorting and overlap
# checking for us (steps 2 through 5). i found this in the pytorch docs:
# https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html
# it takes boxes in [x1, y1, x2, y2] format, confidence scores, and
# an iou threshold, and returns the indices of the boxes to keep
#
# we just need to handle step 1 (confidence threshold) ourselves
# and convert our boxes to the right format before calling it
#
# inputs to our nms function:
#   loc predictions [N, 1364, 4] from the model (these are offsets from anchors)
#   cls predictions [N, 1364, 2] from the model (waldo vs background scores)
#   anchors [1364, 4] the anchor positions from ConcatenateAnchors
#   confidence_threshold (how sure the model needs to be to keep a prediction)
#   iou_threshold (how much two boxes can overlap before we call them duplicates)
#
# output:
#   for each image in the batch, a list of filtered boxes and their scores
#   only the best non overlapping predictions survive
#
# things to be careful about:
#
#   box format conversion:
#     our anchors use [cx, cy, w, h] (center x, center y, width, height)
#     but torchvision.ops.nms expects [x1, y1, x2, y2] (top left, bottom right)
#     so we need a helper function to convert between these two formats
#     i figured this out from looking at how other SSD implementations do it
#
#   decoding predictions:
#     the SSD heads dont predict actual box positions
#     they predict offsets from the anchor positions
#     so before we can do anything useful we need to "decode" the predictions
#     by adding the predicted offsets to the anchor centers
#     and scaling the predicted width/height by the anchor sizes
#     this gives us the actual box coordinates on the image
#
#   per image processing:
#     nms runs on one image at a time, not the whole batch at once
#     you cant mix predictions from different images together
#     so we need to loop through the batch and run nms on each image separately
#
#   softmax for class scores:
#     the model outputs raw scores (logits) not probabilities
#     we need to run softmax on the class scores first
#     so they become actual probabilities between 0 and 1
#     then we use the waldo probability (index 1) as the confidence score

# TODO: write a helper to convert [cx, cy, w, h] to [x1, y1, x2, y2]
# TODO: write a function to decode the predicted offsets into actual box positions
# TODO: write the main nms function that filters predictions for each image
# TODO: add a test at the bottom with fake predictions to make sure shapes are right
