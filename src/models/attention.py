"""
Attention modules used by the backbone.

- `CBAM`: Convolutional Block Attention Module (channel + spatial gating).
- `DepthAttention`: learnable depth-wise aggregation, replacing rigid MaxPool3d.
"""
import torch
import torch.nn as nn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018).

    Args:
        gate_channels: input channel count.
        reduction_ratio: bottleneck reduction for the channel-gating MLP.
    """

    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # 1. Channel attention
        avg_out = self.shared_mlp(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.shared_mlp(self.max_pool(x).view(x.size(0), -1))
        channel_att = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        x = x * channel_att

        # 2. Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = self.sigmoid(self.conv_spatial(spatial_in))
        x = x * spatial_att
        return x


class DepthAttention(nn.Module):
    """
    Learnable depth aggregation that replaces MaxPool3d.

    Lets the model decide which slices matter most (e.g. those containing
    small tears) instead of bluntly pooling along the depth axis.
    """

    def __init__(self, channels=512, depth=32):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                             # (B, C, D, 7, 7) -> (B, C, D, 1, 1)
            nn.Conv3d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input: (B, C, D, H, W)
        weights = self.attn(x)            # (B, C, D, 1, 1)
        out = (x * weights).sum(dim=2)    # weighted depth aggregation
        return out
