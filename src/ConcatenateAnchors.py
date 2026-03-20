
# Anchors will be center points on the original image that will be constant (there will be 4 of them). Input will be a struct from Gabe containing the following for each anchor: 1) offset from anchor 2) probability for whether waldo is in that box/anchor 

# My job is to add 4 boxes on the original image for each head (so if there's 6 heads, then that's 6 * 4 = 24 boxes that will be calculated but not drawn yet)


class ConcatenateAnchors:


    def __init__(self):
        self.headsOutput = []

    def ConcatenateAnchorOfHead(self,SSDHeadOutput):
        #SSDHeadsOutput should contain 1) Position of anchors 2) offsets from each anchor 3) probabilities for waldo being contained in the box centered around the offset
        self.headsOutput.append(SSDHeadOutput)

    def GetHeadsOutput(self):
        return self.headsOutput
         