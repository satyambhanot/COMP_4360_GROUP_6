# evaluate_ablation.py
# this is where we figure out if each piece of our pipeline actually helps
# we train and test multiple versions of the model with different parts turned off
# then compare the results to see what matters and what doesnt
#
# the ablation studies we want to run:
#   1. full model (everything on) as the baseline
#   2. no HSV mask (remove the red/white cheat sheet, keep sobel)
#   3. no Sobel edges (remove the edge cheat sheet, keep HSV)
#   4. no pre processing at all (just raw RGB into the backbone)
#   5. fewer SSD heads (try 3 instead of 5, see if we lose accuracy)
#   6. different tile sizes (only 256, only 128, etc)
#
# for each experiment we compare accuracy, precision, and recall
# this shows the professor we understand why each component matters
# not just that the whole thing works together
#
# TODO: implement once train.py and evaluate.py are working
# TODO: print a comparison table at the end
