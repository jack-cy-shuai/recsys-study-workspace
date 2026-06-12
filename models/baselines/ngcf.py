"""NGCF: Neural Graph Collaborative Filtering + HardBPR variant.

LightGCN 的前身，在每层图传播中加入特征变换矩阵 W 和 LeakyReLU 激活。
输出是各层拼接（非平均），使用 BPR 损失训练。
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseRecommender
from losses.traditional import l2_reg_loss


class NGCF(BaseRecommender):
    """Neural Graph Collaborative Filtering (SIGIR 2019).

    每层: E^(k+1) = LeakyReLU( norm_adj @ E^(k) @ W^(k) + E^(k) @ W_self^(k) )
    输出: concat(E^(0), E^(1), ..., E^(K)) → 最终维度 = (K+1) * d
    """

    def __init__(self, num_users, num_items, config, norm_adj):
        super().__init__(num_users, num_items, config)
        self.num_layers = config["num_layers"]
        self.reg_weight = config.get("reg_weight", 1e-4)
        self.register_buffer("norm_adj", norm_adj.coalesce())

        d = self.embedding_dim
        self.user_embedding = nn.Embedding(num_users, d)
        self.item_embedding = nn.Embedding(num_items, d)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        # 每层的变换矩阵
        self.W_list = nn.ModuleList()
        self.W_self_list = nn.ModuleList()
        for _ in range(self.num_layers):
            self.W_list.append(nn.Linear(d, d, bias=False))
            self.W_self_list.append(nn.Linear(d, d, bias=False))
            nn.init.xavier_uniform_(self.W_list[-1].weight)
            nn.init.xavier_uniform_(self.W_self_list[-1].weight)

        self._cached_user_emb = None
        self._cached_item_emb = None

    def _propagate(self):
        ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layers = [ego]
        x = ego
        for k in range(self.num_layers):
            neighbor = torch.sparse.mm(self.norm_adj, x)
            x_next = self.W_list[k](neighbor) + self.W_self_list[k](x)
            x = F.leaky_relu(x_next, 0.2)
            layers.append(x)
        final = torch.cat(layers, dim=1)  # concat along feature dim
        u, i = torch.split(final, [self.num_users, self.num_items], dim=0)
        return u, i

    def get_user_embeddings(self):
        if self.training: return self._propagate()[0]
        if self._cached_user_emb is None:
            self._cached_user_emb, self._cached_item_emb = self._propagate()
        return self._cached_user_emb

    def get_item_embeddings(self):
        if self.training: return self._propagate()[1]
        if self._cached_item_emb is None:
            self._cached_user_emb, self._cached_item_emb = self._propagate()
        return self._cached_item_emb

    def invalidate_cache(self):
        self._cached_user_emb = None
        self._cached_item_emb = None

    def forward(self, users, items):
        u = self.get_user_embeddings()[users]
        i = self.get_item_embeddings()[items]
        return (u * i).sum(dim=1)

    def compute_loss(self, batch, **kwargs):
        users, pos_items, neg_items = batch
        p = self.forward(users, pos_items)
        n = self.forward(users, neg_items)
        r = -F.logsigmoid(p - n).mean()
        reg = l2_reg_loss([self.user_embedding(users),
                           self.item_embedding(pos_items),
                           self.item_embedding(neg_items)], reduction="mean")
        reg_w = kwargs.get("reg_weight", self.reg_weight)
        return r + reg_w * reg

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NGCF_HardBPR(NGCF):
    """NGCF + 语义感知 BPR（LLM 仅用于 loss 加权）。"""

    def __init__(self, num_users, num_items, config, norm_adj,
                 llm_item_emb=None):
        super().__init__(num_users, num_items, config, norm_adj)
        self.hard_bpr_beta = config.get("hard_bpr_beta", 0.5)
        if llm_item_emb is not None:
            self.register_buffer("_llm_item_norm",
                                 F.normalize(llm_item_emb.float(), p=2, dim=1))
        else:
            self._llm_item_norm = None

    def compute_loss(self, batch, **kwargs):
        users, pos_items, neg_items = batch
        all_u, all_i = self._propagate()
        p = (all_u[users] * all_i[pos_items]).sum(dim=1)
        n = (all_u[users] * all_i[neg_items]).sum(dim=1)

        if self._llm_item_norm is not None:
            sem = (self._llm_item_norm[pos_items] *
                   self._llm_item_norm[neg_items]).sum(dim=1)
            w = 1.0 + self.hard_bpr_beta * sem
            r = -(w * F.logsigmoid(p - n)).mean()
        else:
            r = -F.logsigmoid(p - n).mean()

        reg = l2_reg_loss([self.user_embedding(users),
                           self.item_embedding(pos_items),
                           self.item_embedding(neg_items)], reduction="mean")
        return r + self.reg_weight * reg

    def count_parameters(self):
        return {"total_trainable": super().count_parameters(),
                "llm_in_loss_only": "yes, frozen"}
