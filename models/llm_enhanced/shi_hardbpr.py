"""Shi Hard-BPR: 钟形梯度 BPR (Shi et al., 2024, arXiv:2403.19276).

将标准 sigmoid 替换为分段钟形梯度函数：
- 中等难度的负样本梯度最大
- 过难的负样本（可能是 false negatives）梯度被抑制
- 过易的负样本梯度自然小

与 HardBPR 的区别：同用 BPR loss，但 Shi 改 sigmoid 形状（模型内部打分驱动），
HardBPR 用外部 LLM 语义相似度加权（保留标准 sigmoid）。两者正交。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baselines.lightgcn import LightGCN
from losses.traditional import l2_reg_loss


class Shi_HardBPR(LightGCN):
    """Shi et al. 钟形梯度 BPR。"""

    def __init__(self, num_users, num_items, config, norm_adj):
        super().__init__(num_users, num_items, config, norm_adj)
        self.bell_a = config.get("bell_a", 1.0)
        self.bell_b = config.get("bell_b", 0.5)

    def _bell_sigmoid(self, x):
        """钟形梯度的 sigmoid 替代：grad = bell_a * x * exp(-bell_b * x^2/2) / sqrt(2*pi)"""
        scaled = self.bell_a * x
        return torch.sigmoid(scaled) - 0.5 * torch.tanh(self.bell_b * x * x) * torch.sigmoid(scaled)

    def _safe_log_sigmoid(self, x):
        """数值稳定的 log_sigmoid 替代。"""
        return F.logsigmoid(x)

    def compute_loss(self, batch, **kwargs):
        users, pos_items, neg_items = batch
        all_u, all_i = self._propagate()
        pos = (all_u[users] * all_i[pos_items]).sum(dim=1)
        neg = (all_u[users] * all_i[neg_items]).sum(dim=1)
        diff = pos - neg

        # Shi Hard-BPR: 钟形梯度
        scaled_diff = self.bell_a * diff
        # 核心公式：-(1 + bell_b * diff) * exp(-bell_b * diff^2/2) * log_sigmoid
        # 简化：用标准 log_sigmoid 但加 bell-shaped weight
        bell_weight = torch.exp(-self.bell_b * diff * diff / 2)
        ranking_loss = -(bell_weight * F.logsigmoid(scaled_diff)).mean()

        reg = l2_reg_loss([self.user_embedding(users),
                           self.item_embedding(pos_items),
                           self.item_embedding(neg_items)], reduction="mean")
        reg_w = kwargs.get("reg_weight", self.reg_weight)
        return ranking_loss + reg_w * reg

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
