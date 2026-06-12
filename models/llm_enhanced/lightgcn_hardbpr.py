"""LightGCN-HardBPR：纯 CF + 语义感知 BPR 损失。

LLM 语义嵌入仅在损失层使用——计算正负样本的语义相似度来加权 BPR。
特征层完全是传统 CF（随机初始化 ID embedding），不加任何 LLM 信号。

对比：LightGCN (标准BPR) vs LightGCN-HardBPR (语义BPR)，唯一变量是损失权重。
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from models.baselines.lightgcn import LightGCN
from losses.traditional import l2_reg_loss


class LightGCN_HardBPR(LightGCN):
    """纯 CF 架构 + 语义感知 BPR 损失。

    LLM embedding 仅用于损失层的语义相似度计算，不参与图传播。
    """

    def __init__(self, num_users, num_items, config, norm_adj,
                 llm_item_emb=None):
        super().__init__(num_users, num_items, config, norm_adj)
        self.hard_bpr_beta = config.get("hard_bpr_beta", 0.5)
        # 冻结的 LLM 语义向量（仅用于 loss 中的相似度计算）
        if llm_item_emb is not None:
            item_norm = F.normalize(llm_item_emb.float(), p=2, dim=1)
            self.register_buffer("_llm_item_norm", item_norm)
        else:
            self._llm_item_norm = None
            print("[HardBPR] Warning: no LLM item embeddings, falling back to standard BPR")

    def compute_loss(self, batch, **kwargs):
        users, pos_items, neg_items = batch

        all_u, all_i = self._propagate()
        u = all_u[users]
        pi = all_i[pos_items]
        ni = all_i[neg_items]
        pos_s = (u * pi).sum(dim=1)
        neg_s = (u * ni).sum(dim=1)

        # 语义感知 BPR 加权
        if self._llm_item_norm is not None:
            sem_sim = (self._llm_item_norm[pos_items] *
                       self._llm_item_norm[neg_items]).sum(dim=1)
            weights = 1.0 + self.hard_bpr_beta * sem_sim
            ranking_loss = -(weights * F.logsigmoid(pos_s - neg_s)).mean()
        else:
            ranking_loss = -F.logsigmoid(pos_s - neg_s).mean()

        reg_loss = l2_reg_loss([
            self.user_embedding(users), self.item_embedding(pos_items),
            self.item_embedding(neg_items)], reduction="mean")
        reg_weight = kwargs.get("reg_weight", self.reg_weight)
        return ranking_loss + reg_weight * reg_loss

    def count_parameters(self):
        n = super().count_parameters()
        return {"total_trainable": n, "llm_in_loss_only": "yes, frozen"}
