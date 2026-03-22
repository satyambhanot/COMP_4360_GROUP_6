# train.py
# this is the main training script, you run this from the project root
# like: python train.py
#
# what it needs to do:
#   1. load training and validation data using data_pipeline.get_dataloader()
#   2. create the full model (from src/model.py once its built)
#   3. set up the loss function
#      SSD uses two losses combined:
#        smooth L1 loss for bounding box predictions (how far off are our boxes?)
#        cross entropy loss for classification (is it waldo or not?)
#   4. handle class imbalance, we have way more negative tiles than positive ones
#      (like 14000 negative vs 56 positive) so we need to weight the loss or
#      use hard negative mining so the model doesnt just learn to always say "no waldo"
#   5. training loop: forward pass -> compute loss -> backward pass -> update weights
#   6. save checkpoints to checkpoints/ folder after each epoch
#   7. print training and validation loss so we can see if its learning
#
# TODO: implement once model.py is wired up
