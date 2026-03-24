# evaluate_ablation.py
# this is where we figure out if each piece of our pipeline actually helps
# we train and test multiple versions of the model with different parts turned off
# then compare the results to see what matters and what doesnt
#
# this file is really important for the report because it shows
# we understand why each component is there, not just that the whole thing works
# the professor wants to see that we can justify our design choices with data
#
# how it works:
#   for each experiment we create a slightly different version of the model
#   train it for the same number of epochs on the same data
#   evaluate it on the same test set
#   then compare all the results side by side
#   every experiment uses the same random seed so the data splits are identical
#   this makes the comparison fair
#
#
# study 1: sobel edge detector bypass
#
#   what we test:
#     run the full model with sobel edges vs without sobel edges
#     with sobel: model gets 5 channels (RGB + HSV mask + Sobel)
#     without sobel: model gets 4 channels (RGB + HSV mask only)
#
#   why we test this:
#     the sobel edge detector is supposed to help the model find waldos stripes
#     because the red and white stripes create strong horizontal edges
#     but maybe the backbone can figure out edges on its own
#     and the sobel channel is just adding noise
#
#   what we expect:
#     performance should drop without sobel because edge information
#     helps distinguish waldos stripe pattern from other red/white objects
#     if it doesnt drop then sobel isnt helping and we can simplify the model
#
#   what we need to change in the code:
#     add a use_sobel flag to preprocess_batch and model.py
#     when False, skip the sobel channel and adjust the 1x1 conv to take 4 channels
#
#
# study 2: model size reduction
#
#   what we test:
#     run the model with different numbers of SSD heads
#     full model uses 5 heads (16x16, 8x8, 4x4, 2x2, 1x1)
#     reduced model uses 3 heads (16x16, 8x8, 4x4)
#     we could also try 2 heads or 4 heads
#
#   why we test this:
#     more heads means the model can detect waldo at more scales
#     but it also means more parameters and slower training
#     the later heads (2x2, 1x1) have very few anchors so they might not be
#     contributing much. if 3 heads works just as well as 5 thats a win
#     because training is faster and the model is simpler
#
#   what we expect:
#     some drop in performance with fewer heads but maybe not a lot
#     waldo is roughly the same size in all tiles so detecting at one
#     or two scales might be enough
#     the paper by Liu et al (SSD: Single Shot MultiBox Detector, ECCV 2016)
#     uses 6 heads but they were detecting objects of many different sizes
#     we are only looking for one character who is always roughly the same size
#
#   what we need to change in the code:
#     make num_heads a parameter in model.py instead of hardcoding 5
#     pass a different number when creating the model for this experiment
#
#
# study 3: image resolution
#
#   what we test:
#     train separate models on each tile size: 64x64, 128x128, 256x256
#     our data pipeline already generates tiles at all three sizes
#     so we just need to train three models with different tile_size values
#
#   why we test this:
#     waldo is about 30 pixels tall in the original images
#     at 256x256 tiles waldo takes up maybe 12% of the tile (lots of clutter)
#     at 128x128 waldo takes up about 23% (less clutter)
#     at 64x64 waldo could take up almost half the tile (very focused)
#     but at 64x64 we have 11,193 tiles vs only 608 at 256x256
#     so theres a tradeoff between how focused each tile is and
#     how many tiles the model has to search through
#     also at 64x64 waldo might get split across tile boundaries
#
#   what we expect:
#     128x128 might be the sweet spot
#     64x64 could have issues with waldo being cut in half at tile edges
#     256x256 has a lot of clutter making it harder for the model
#     we should report both accuracy and total tiles processed
#     to show the full tradeoff
#
#   what we need to change in the code:
#     nothing, just call get_dataloader with different tile_size values
#     the dataset already supports this
#
#
# study 4: color information
#
#   what we test:
#     train the model on three versions of the same images:
#       full color (RGB, what we normally use)
#       grayscale (convert to gray, duplicate to 3 channels for mobilenet)
#       black and white (threshold grayscale to pure black or white pixels)
#
#   why we test this:
#     waldos most distinctive feature is his red and white striped shirt
#     if we remove color the HSV mask becomes meaningless because
#     theres no color information to detect red and white
#     but the sobel edges should still work because edges exist in grayscale
#     this study shows how much our model depends on color vs structure
#     the original Hey Waldo dataset (github.com/vc1492a/Hey-Waldo)
#     actually includes grayscale and black/white versions of the tiles
#     which is what inspired this study
#
#   what we expect:
#     big performance drop when color is removed
#     because waldos red and white stripes are his most distinctive feature
#     grayscale should perform better than black/white because
#     grayscale keeps the brightness information (waldo is lighter than background)
#     black/white throws that away too
#     sobel should partially compensate because stripe edges still exist
#
#   what we need to change in the code:
#     add a color_mode parameter to the dataset or preprocessing
#     when "grayscale": convert RGB to gray, duplicate to 3 channels
#     when "bw": convert to grayscale, threshold, duplicate to 3 channels
#     the imagenet normalization might need adjusting for grayscale
#
#
# study 5: pretrained vs random backbone
#
#   what we test:
#     train the model with mobilenet backbone pretrained on imagenet
#     vs mobilenet backbone with completely random weights
#
#   why we test this:
#     our whole approach assumes that features learned from real photos
#     (imagenet has dogs, cars, buildings etc) transfer to cartoon illustrations
#     (wheres waldo drawings). but thats not obvious because cartoons look
#     very different from photos. if the pretrained backbone doesnt help much
#     then transfer learning isnt useful for this problem and we could
#     potentially use a simpler custom backbone instead
#     the concept of transfer learning and how it works across different
#     visual domains is discussed in the MobileNetV3 paper
#     (Howard et al, Searching for MobileNetV3, ICCV 2019)
#
#   what we expect:
#     pretrained should win because even though cartoons look different
#     from photos, the low level features (edges, corners, textures)
#     are similar. mobilenet learned these in its early layers and those
#     features are useful for any visual task
#     the interesting question is by how much. if pretrained wins by
#     a huge margin it means the model heavily relies on transfer learning
#     if its close it means the task is simple enough to learn from scratch
#
#   what we need to change in the code:
#     add a use_pretrained flag to model.py
#     when True: load mobilenet with weights='DEFAULT' (pretrained on imagenet)
#     when False: load mobilenet with weights=None (random initialization)
#     thats literally one line change, everything else stays the same
#
#
# what we collect for every experiment:
#   accuracy, precision, recall, f1 score
#   confusion matrix (true positives, false positives, true negatives, false negatives)
#   training loss curve (loss per epoch)
#   validation loss curve (to check for overfitting)
#   how many epochs it took to reach the best val loss
#   total training time on the GTX 1080 Ti
#
# what this file should output for the report:
#   a comparison table with one row per experiment and columns for all metrics
#   training curves overlaid on one chart so you can see which variants learn faster
#   bar charts comparing accuracy/recall/f1 across experiments
#   a few example tiles where different variants disagree
#     (one found waldo, the other didnt) to visualize what each component helps with
#
# things to be careful about:
#   use the same random seed for every experiment so data splits are identical
#   use the same number of epochs for fair comparison
#   save each experiments checkpoint with a different name
#     like checkpoints/ablation_no_sobel.pth, checkpoints/ablation_grayscale.pth
#   training 5 experiments x 50 epochs each could take a while on the GTX 1080 Ti
#   we might want to use fewer epochs for ablation (like 20) to save time
#   some experiments change the number of input channels to the model
#     so the 1x1 conv layer needs to match (5 channels, 4 channels, or skip it)

# TODO: implement once train.py and evaluate.py are working
