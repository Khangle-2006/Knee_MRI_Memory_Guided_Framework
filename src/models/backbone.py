"""
Hybrid backbone: ResNet18 with a stride-1 final block, CBAM channel/spatial
attention, and DepthAttention to collapse 3D into a 2D feature map.

Input  : (B, 3, D, H, W) volumes
Output : (B, 512, 14, 14)
"""
import torch
import torch.nn as nn
import torchvision.models as models

from .attention import CBAM, DepthAttention


class HybridBackbone(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # Load standard ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # IMPORTANT: keep the spatial size at 14x14 by removing layer4 stride
        resnet.layer4[0].conv1.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        # Drop the final AvgPool + FC layers
        self.features_2d = nn.Sequential(*list(resnet.children())[:-2])

        # 2D channel/spatial gating
        self.cbam = CBAM(gate_channels=512, reduction_ratio=16)

        # 3D depth aggregation; input will be (B, 512, 32, 14, 14)
        self.adapter_3d = DepthAttention(channels=512, depth=32)

    def forward(self, x):
        # Input: (B, 3, 32, 224, 224)
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

        features = self.features_2d(x)        # (B*D, 512, 14, 14)
        features = self.cbam(features)

        # Re-fold the depth dimension
        _, C_feat, H_feat, W_feat = features.shape
        features = features.view(B, D, C_feat, H_feat, W_feat).permute(0, 2, 1, 3, 4)

        # Aggregate over depth
        out = self.adapter_3d(features)       # (B, 512, 14, 14)
        return out
