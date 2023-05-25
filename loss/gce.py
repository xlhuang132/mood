
# add by 陈奕州, imboaba

import torch
from torch import nn
from torch.nn import functional as F

# from https://zhuanlan.zhihu.com/p/420913134
class GeneralizedCELoss(nn.Module):
    def __init__(self, num_classes=10, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes
        self._eps = 10-6 # maybe float? (10-7)

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=self._eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()
    
