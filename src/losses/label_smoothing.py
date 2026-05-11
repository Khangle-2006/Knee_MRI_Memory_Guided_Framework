"""
Binary cross-entropy with label smoothing.
"""
import torch.nn as nn


class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce(logits, targets_smooth)
        return loss.mean()
