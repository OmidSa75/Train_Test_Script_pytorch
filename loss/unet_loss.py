import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

        self.smooth = 1.0

    def forward(self, y_pred: torch.Tensor, y_org: torch.Tensor):
        assert y_pred.size() == y_org.size(), 'y_pred and y_org must be same size'
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_org = y_org[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_org).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_org.sum() + self.smooth)
        return 1. - dsc
