"""
Top-level classifier: backbone -> task-aware memory -> MLP head.
"""
import torch.nn as nn

from .backbone import HybridBackbone
from .memory import TaskAwareMemoryModule


class SingleViewMemoryNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        # 1. Backbone
        self.backbone = HybridBackbone()

        # 2. Task-aware memory
        self.memory = TaskAwareMemoryModule(feature_dim=512, out_dim=128, top_k=5)

        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(3 * 128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        # x: single-view input (B, 3, 32, 224, 224)

        # Feature extraction
        feats_2d = self.backbone(x)              # (B, 512, 14, 14)

        # Memory lookup
        mem_feats, maps = self.memory(feats_2d)

        # Classification
        flat = mem_feats.flatten(1)              # (B, 3*128)
        logits = self.classifier(flat)

        return logits, maps
