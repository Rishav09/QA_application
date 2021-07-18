"""Author: Rishav Sapahia."""

import torch.nn as nn
import torch


class WeightedFocalLoss(nn.Module):
    """For classification in imabalanced classes."""

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss(reduce='none')

    def forward(self, inputs, targets):
        CE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss

        return F_loss
