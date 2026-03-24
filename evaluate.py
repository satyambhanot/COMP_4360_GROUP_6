# evaluate.py
# this is where we find out if our model actually works
# run from project root: python evaluate.py
#
# what this file does:
#   loads the best trained model from checkpoints, runs it on test tiles
#   the model has never seen before, measures how well it finds waldo
#   using real metrics, and draws the results so we can see them
#
# the evaluation flow:
#
#   1. setup
#      load the WaldoDetector model
#      load the best checkpoint from checkpoints/best_model.pth
#      move the model to GPU if available (GTX 1080 Ti)
#      put the model in eval mode (this is important, it changes how some layers behave)
#      load the test dataloader (these are tiles the model never saw during training)
#
#   2. run predictions on every test tile
#      loop through all test batches with torch.no_grad() to save memory
#      for each batch run the model to get loc and cls predictions
#      run NMS to filter out overlapping boxes and keep the best ones
#      collect all predictions and ground truth labels so we can compare them
#
#   3. compute metrics
#      first we build a confusion matrix which counts four things:
#        true positives: model said waldo and waldo was actually there (good)
#        false positives: model said waldo but waldo wasnt there (bad, false alarm)
#        true negatives: model said no waldo and there was no waldo (good)
#        false negatives: model said no waldo but waldo was there (worst, we missed him)
#
#      from those four numbers we compute:
#        accuracy: what percentage of tiles did we get right overall
#          not super useful on its own because of class imbalance
#          (saying "no waldo" every time gives 99.6% accuracy)
#        precision: when the model says waldo, how often is it actually right
#          high precision means few false alarms
#        recall: of all the tiles that actually have waldo, how many did we find
#          high recall means we rarely miss waldo
#        f1 score: combines precision and recall into one number
#          useful when you want a single number to compare models
#
#   4. visualize individual tiles
#      pick some test tiles and draw the predicted bounding boxes on them
#      green box if the prediction was correct
#      red box if the model got it wrong
#      save these as images we can put in the report
#      this helps us understand what the model is actually doing
#      not just whether the numbers look good
#
#   5. full image demo
#      this is the cool part for the presentation
#      take one of the original full wheres waldo images
#      tile it up the same way data_pipeline does
#      run every tile through the model
#      for each tile where the model thinks waldo is,
#      map the bounding box back to the original image coordinates
#      (the box is relative to the tile so we add the tiles x,y offset
#      from splits.json to get the position on the full image)
#      draw a big box on the full image showing where the model found waldo
#      save it as the final "look it actually works" image
#
# things to be careful about:
#
#   model.eval() is required, without it the results will be wrong
#
#   NMS only runs here during evaluation, not during training
#   this is where we actually call the nms function to filter predictions
#
#   the test set is split by image id not by tile
#   so entire images are held out, not just random tiles
#   this means the model cant cheat by memorizing parts of an image
#   it saw during training
#
#   we need PIL for drawing boxes and saving visualization images
#   its already in requirements.txt so that should be fine
#
#   when mapping tile predictions back to the full image
#   the bounding box from the model is in tile coordinates
#   we need to add the tiles position offset to convert to image coordinates
#   splits.json has this info stored for every tile
#
#   if the checkpoint file doesnt exist (nobody trained the model yet)
#   we should print a helpful message telling the user to run train.py first
#   instead of just crashing with a file not found error
#
#   when stitching predictions back onto the full image
#   multiple tiles might think they found waldo in nearby spots
#   thats actually fine, it means the model is confident about that area
#   we can either show all of them or run another round of NMS
#   across tile boundaries to merge them into one final prediction

# TODO: implement once the model is trained

from PIL import Image, ImageDraw


def draw_boxes(image, boxes, labels=None, scores=None, color="red", width=2):
    # takes a PIL image and draws bounding boxes on it
    # used for visualizing what the model predicted
    # green boxes for correct predictions, red for wrong ones
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box.tolist()

        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)

        text = ""
        if labels is not None:
            text += f"{labels[i]}"
        if scores is not None:
            text += f": {scores[i]:.2f}"

        if text:
            draw.text((x_min, y_min), text, fill=color)

    return image


def create_tiles_from_image(image, tile_size, overlap):
    # takes a full wheres waldo image and chops it into tiles
    # this is for the demo where we run the trained model on a new image
    # we keep track of each tiles position so we can map predictions
    # back to the original image later
    tiles = []
    tile_coords = []

    width, height = image.size

    for h in range(0, height, tile_size - overlap):
        for w in range(0, width, tile_size - overlap):
            tile = image.crop((w, h, w + tile_size, h + tile_size))
            tiles.append(tile)
            tile_coords.append((w, h))

    return tiles, tile_coords
