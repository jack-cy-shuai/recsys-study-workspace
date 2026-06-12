"""MF: 矩阵分解 (Matrix Factorization)
以及 MF-LLM 和 MF-LLM-HardBPR 变体。

MF 不使用图传播，仅做 user_embedding @ item_embedding.T 内积预测。
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseRecommender
from losses.traditional import bpr_loss, l2_reg_loss


class MF(BaseRecommender):
    """标准矩阵分解，BPR 训练。"""

    def __init__(self, num_users, num_items, config, norm_adj=None):
        super().__init__(num_users, num_items, config)
        self.reg_weight = config.get("reg_weight", 1e-4)
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def get_user_embeddings(self):
        return self.user_embedding.weight

    def get_item_embeddings(self):
        return self.item_embedding.weight

    def invalidate_cache(self):
        pass

    def forward(self, users, items):
        u = self.user_embedding(users)
        i = self.item_embedding(items)
        return (u * i).sum(dim=1)

    def compute_loss(self, batch, **kwargs):
        users, pos_items, neg_items = batch
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        ranking_loss = bpr_loss(pos_scores, neg_scores)
        reg_loss = l2_reg_loss([
            self.user_embedding(users),
            self.item_embedding(pos_items),
            self.item_embedding(neg_items),
        ], reduction="mean")
        return ranking_loss + self.reg_weight * reg_loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MF_LLM(MF):
    """MF + LLM 语义相加融合。"""

    def __init__(self, num_users, num_items, config, norm_adj=None,
                 llm_user_emb=None, llm_item_emb=None):
        super().__init__(num_users, num_items, config, norm_adj)
        llm_dim = llm_user_emb.shape[1]
        self.llm_proj_user = nn.Linear(llm_dim, self.embedding_dim, bias=False)
        self.llm_proj_item = nn.Linear(llm_dim, self.embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.llm_proj_user.weight, gain=0.1)
        nn.init.xavier_uniform_(self.llm_proj_item.weight, gain=0.1)
        self.register_buffer("llm_user_raw", llm_user_emb.float())
        self.register_buffer("llm_item_raw", llm_item_emb.float())

    def get_user_embeddings(self):
        return self.user_embedding.weight + self.llm_proj_user(self.llm_user_raw)

    def get_item_embeddings(self):
        return self.item_embedding.weight + self.llm_proj_item(self.llm_item_raw)

    def count_parameters(self):
        d = {"total_trainable": super().count_parameters(),
             "llm_embedding (frozen)": self.llm_user_raw.numel() + self.llm_item_raw.numel()}
        return d


class MF_LLM_HardBPR(MF_LLM):
    """MF + LLM + 语义感知 BPR。"""

    def __init__(self, num_users, num_items, config, norm_adj=None,
                 llm_user_emb=None, llm_item_emb=None):
        super().__init__(num_users, num_items, config, norm_adj, llm_user_emb, llm_item_emb)
        self.hard_bpr_mode = config.get("hard_bpr_mode", "weight")
        self.hard_bpr_beta = config.get("hard_bpr_beta", 0.5)
        llm_norm = F.normalize(self.llm_item_raw, p=2, dim=1)
        self.register_buffer("_llm_item_norm", llm_norm)

    def compute_loss(self, batch, **kwargs):
        users, pos_items, neg_items = batch
        u = self.get_user_embeddings()[users]
        pi = self.get_item_embeddings()[pos_items]
        ni = self.get_item_embeddings()[neg_items]
        pos_scores = (u * pi).sum(dim=1)
        neg_scores = (u * ni).sum(dim=1)

        if self.hard_bpr_mode == "weight":
            sem_sim = (self._llm_item_norm[pos_items] * self._llm_item_norm[neg_items]).sum(dim=1)
            weights = 1.0 + self.hard_bpr_beta * sem_sim
            ranking_loss = -(weights * F.logsigmoid(pos_scores - neg_scores)).mean()
        else:
            ranking_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        reg_loss = l2_reg_loss([
            self.user_embedding(users),
            self.item_embedding(pos_items),
            self.item_embedding(neg_items),
        ], reduction="mean")
        return ranking_loss + self.reg_weight * reg_loss

    def count_parameters(self):
        d = super().count_parameters()
        d["hard_bpr_mode"] = self.hard_bpr_mode
        return d
