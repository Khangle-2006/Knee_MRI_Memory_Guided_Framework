"""
SimCLR-style projection wrapper around a backbone for contrastive pretraining.
"""
import torch.nn as nn
import torch.nn.functional as F


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim=128):
        super(ResNetSimCLR, self).__init__()
        self.encoder = base_model
        self.feature_dim = 512
        self.encoder.fc = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, F.normalize(z, dim=1)
