"""
Task-Aware Memory Module.

Strategy: SHARED SPATIAL GATE
- One small shared encoder produces a 64-channel feature map.
- Three lightweight 1x1 heads turn that into per-task spatial masks
  (abnormal / ACL / meniscus).
- Each task has its own learnable memory bank queried via Top-K
  attention, fused with the visual query through a gating MLP.

This cuts the parameter count from ~1.7M to ~0.3M vs the naive per-task
encoder design.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskAwareMemoryModule(nn.Module):
    def __init__(self, feature_dim=512, memory_slots=64, out_dim=128, top_k=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.memory_slots = memory_slots
        self.top_k = top_k

        # 1. Shared spatial encoder (512 -> 64 channels)
        self.shared_spatial_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 2. Task-specific 1x1 heads producing per-task spatial masks
        self.abnormal_head = nn.Conv2d(64, 1, kernel_size=1)
        self.acl_head      = nn.Conv2d(64, 1, kernel_size=1)
        self.meniscus_head = nn.Conv2d(64, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        # 3. Per-task query projections (512 -> 128)
        self.query_projs = nn.ModuleList([
            nn.Conv2d(feature_dim, out_dim, kernel_size=1) for _ in range(3)
        ])

        # 4. Three independent memory banks
        self.memory_banks = nn.Parameter(torch.empty(3, memory_slots, out_dim))
        for i in range(3):
            nn.init.orthogonal_(self.memory_banks[i])

        # 5. Gating MLP for visual <-> memory fusion
        self.gate_fc = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, feature_map):
        # feature_map: (B, 512, H, W)
        B, C, H, W = feature_map.shape

        refined_feats = []
        attention_maps = []

        # Compute the shared spatial features once
        shared_spatial_feat = self.shared_spatial_encoder(feature_map)  # (B, 64, H, W)

        task_heads = [self.abnormal_head, self.acl_head, self.meniscus_head]

        for i in range(3):
            # 1. Per-task spatial mask
            raw_mask = task_heads[i](shared_spatial_feat)               # (B, 1, H, W)
            spatial_mask = self.sigmoid(raw_mask)
            attention_maps.append(spatial_mask.squeeze(1))

            # 2. Apply mask to the original 512-channel feature map
            task_specific_feat = feature_map * spatial_mask

            # 3. Project to query
            query = self.query_projs[i](task_specific_feat).flatten(2).permute(0, 2, 1)
            bank = self.memory_banks[i]

            # 4. Top-K masked attention over memory
            attn_logits = torch.matmul(query, bank.t())
            top_val, top_idx = torch.topk(attn_logits, k=self.top_k, dim=-1)
            mask_scores = torch.full_like(attn_logits, float('-inf'))
            mask_scores.scatter_(-1, top_idx, top_val)
            attn_prob = F.softmax(mask_scores, dim=-1)

            # 5. Retrieve and gate
            retrieved = torch.matmul(attn_prob, bank)
            concat_info = torch.cat([query, retrieved], dim=-1)
            gate = self.gate_fc(concat_info)
            fused = gate * retrieved + (1 - gate) * query

            # 6. Spatial pooling
            task_feat = fused.mean(dim=1)
            refined_feats.append(task_feat)

        final_features = torch.stack(refined_feats, dim=1)              # (B, 3, 128)
        final_features = self.norm(final_features)

        final_maps = torch.stack(attention_maps, dim=1)                  # (B, 3, H, W)
        return final_features, final_maps
