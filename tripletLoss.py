import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        
        ap_dist = F.pairwise_distance(anchor, positive, p=2)
        an_dist = F.pairwise_distance(anchor, negative, p=2)

        losses = F.relu(ap_dist - an_dist + self.margin)

        return losses.mean()
