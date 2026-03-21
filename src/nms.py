# nms.py
# non maximum suppression
# after the model makes predictions, we end up with a lot of overlapping boxes
# that all think they found waldo in roughly the same spot
# nms looks at all those overlapping boxes, picks the one with the highest
# confidence score, and throws away the rest
# this way we get one clean prediction per location instead of a pile of duplicates
#
# TODO: implement the filtering logic
# TODO: add a confidence threshold so we ignore low confidence predictions
# TODO: add an overlap threshold so we know when two boxes are "too close"
