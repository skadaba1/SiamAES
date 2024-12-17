import torch
import torch.nn as nn
class CrossEntropyLoss(torch.nn.Module):
  def __init__(self, num_labels):
    super(CrossEntropyLoss, self).__init__()  # pre 3.3 syntax
    self.num_labels = num_labels
    self.softmax = nn.Softmax(dim=1)
    
  def forward(self, x):
    loss_fct = nn.CrossEntropyLoss()
    logits = x[0]; target = x[1]; outputs = x[2]
    #print(self.softmax(logits), target)
    soft = self.softmax(logits)
    loss = loss_fct(logits, target)
    entropy =  soft@torch.log(soft).T
    gamma = 0.3
    return gamma*entropy + loss

class ContrastiveLoss(torch.nn.Module):
  def __init__(self, m=1):
    super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
    self.m = m  # margin or radius

  def forward(self, y1, y2, flag):
    # flag = 1 means y1 and y2 are supposed to be same
    # flag = 0 means y1 and y2 are supposed to be different
    pdist = nn.PairwiseDistance(p=2)
    self.m = (torch.norm(y1, 1) + torch.norm(y2, 1)) / 2
    #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    euc_dist = pdist(y1, y2)
    if(flag != None):
      loss = torch.mean(flag * torch.pow(euc_dist, 2) + (1-flag) * torch.pow(torch.clamp(self.m - euc_dist, min=0.0), 2))
    else:
      loss = euc_dist
    return loss 
    
class TripletLoss(torch.nn.Module):
    def __init__(self, m=1.0):
      super(TripletLoss, self).__init__()  # pre 3.3 syntax
      self.m = m  # margin or radius
    def forward(self, anchor, positive, negative):
      triplet_loss = nn.TripletMarginLoss(margin=self.m, p=2)
      loss = triplet_loss(anchor, positive, negative)
      return loss 