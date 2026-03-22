# evaluate.py
# this is where we test our trained model and see how well it finds waldo
# run from project root: python evaluate.py
#
# what it needs to do:
#   1. load a saved model checkpoint from checkpoints/
#   2. run it on the test split (tiles the model has never seen)
#   3. for each tile, get the model's prediction (waldo or not, and where)
#   4. calculate metrics:
#      accuracy (what % of tiles did we classify correctly)
#      precision (when we say waldo, how often are we right)
#      recall (of all the tiles with waldo, how many did we find)
#   5. visualize results, draw predicted bounding boxes on tiles
#      and maybe stitch them back onto the full image to show
#      where the model thinks waldo is
#
# this is also where we'd do the final demo:
#   take a full wheres waldo image
#   tile it up
#   run each tile through the model
#   combine predictions back onto the original image
#   draw a box around where waldo is
#
# TODO: implement once the model is trained
