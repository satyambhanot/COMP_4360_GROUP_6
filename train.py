# train.py
# this is the main training script, you run it from the project root
# like: python train.py
#
# what this file does:
#   loads tiles from the data pipeline, feeds them through the model,
#   checks how wrong the predictions are (the loss), uses that to update
#   the model weights, and saves the best version to checkpoints
#
# the training flow:
#
#   1. setup
#      load the WaldoDetector model
#      create an Adam optimizer with LEARNING_RATE (0.001) from config
#      we picked Adam because it adjusts the learning rate automatically
#      for each parameter which usually works well out of the box
#      load train and validation dataloaders with BATCH_SIZE (16) from config
#      check if we have a GPU available (we are using a GTX 1080 Ti with 11GB VRAM)
#      move the model to GPU if we can, otherwise it runs on CPU which is way slower
#
#   2. loss function
#      SSD uses two losses combined:
#
#      smooth L1 loss for bounding boxes:
#        measures how far off our predicted box is from the actual waldo box
#        its like absolute error but smoother near zero so it doesnt
#        freak out over tiny differences
#        only computed for positive tiles (tiles with waldo in them)
#        theres no point penalizing box predictions where waldo isnt there
#
#      cross entropy loss for classification:
#        measures how confident the model is about waldo vs not waldo
#        if the model says 90% waldo and its actually waldo, small loss
#        if the model says 90% waldo and its actually not waldo, big loss
#        computed for all tiles, both positive and negative
#
#      we combine them: total_loss = cls_loss + loc_loss
#
#   3. class imbalance
#      this is probably the trickiest part
#      we have about 14000 negative tiles but only 56 positive ones
#      if we dont handle this the model will just learn to always say "no waldo"
#      and get 99.6% accuracy while being completely useless
#      we will use weighted loss to give positive tiles much higher importance
#      so when the model gets a waldo tile wrong it hurts a lot more
#      we might also try hard negative mining later (only training on the
#      hardest negatives instead of all 14000 of them)
#
#   4. training loop (runs once per epoch, NUM_EPOCHS (50) total)
#      loop through all batches in the training dataloader
#      for each batch:
#        move images, labels, and bboxes to GPU
#        forward pass through the model to get predictions
#        compute the combined loss
#        backward pass to calculate gradients
#        optimizer step to update the weights
#      track the running loss so we can print it at the end of the epoch
#
#   5. validation (runs after each training epoch)
#      run through the validation set with torch.no_grad() to save memory
#      compute the validation loss but dont update any weights
#      this tells us if the model is overfitting
#      (train loss going down but val loss going up means overfitting)
#
#   6. checkpoint saving
#      after each epoch compare the val loss to the best we have seen so far
#      if its the best, save the model weights to checkpoints/best_model.pth
#      also save the last model as checkpoints/last_model.pth
#      we save the optimizer state too so we can resume training if it crashes
#
#   7. print progress
#      after each epoch print the epoch number, train loss, val loss
#      and whether we saved a new best model
#      also print how many positive tiles the model got right as a sanity check
#      at the end print which epoch had the best validation loss
#
# things to keep in mind:
#
#   we are training on 256x256 tiles to start with
#   we can try other tile sizes later in ablation studies
#
#   model.train() for training, model.eval() for validation
#   this matters because it changes how some layers behave
#
#   if the val loss stops improving for many epochs in a row
#   we might want to add early stopping so we dont waste time
#
#   we could also add a learning rate scheduler that reduces the
#   learning rate if val loss plateaus, this sometimes helps the
#   model find a better solution
#
#   for now we start simple and add these things if we need them

# TODO: implement once model.py is done
