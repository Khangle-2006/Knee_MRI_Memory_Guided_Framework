"""
ModAn-MulSupCon: Modality- and Anatomy-aware Multi-Label Supervised
Contrastive loss.

Positives are pairs whose Jaccard similarity (over multi-hot labels) exceeds
a threshold; the loss weights each positive by that similarity.
"""
import torch
import torch.nn as nn


class ModAnMulSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, threshold=0.3):
        super(ModAnMulSupConLoss, self).__init__()
        self.temperature = temperature
        self.threshold = threshold

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        sim_matrix = torch.matmul(features, features.T) / self.temperature

        intersection = torch.matmul(labels, labels.T)
        labels_sum = labels.sum(dim=1, keepdim=True)
        union = labels_sum + labels_sum.T - intersection
        jaccard_sim = intersection / (union + 1e-8)

        mask_threshold = (jaccard_sim >= self.threshold).float()
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_threshold * (1 - mask_self)

        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        weighted_log_prob = jaccard_sim * log_prob * mask_pos
        num_positives = mask_pos.sum(1)

        loss = - (weighted_log_prob.sum(1) / (num_positives + 1e-6))
        loss = loss[num_positives > 0].mean()

        return loss
