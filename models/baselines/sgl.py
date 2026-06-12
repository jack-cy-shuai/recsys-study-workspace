"""SGL: Simplified Graph Contrastive Learning + 变体

在 LightGCN 基础上增加边 Dropout 的对比学习。每个 batch:
1. 两次传播: 每次随机 drop 一部分边 → 两个增广视图
2. InfoNCE 损失: 同节点的两个表示互相为正，batch 内其他为负
3. 联合训练: BPR + λ * InfoNCE
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baselines.lightgcn import LightGCN
from losses.traditional import bpr_loss, l2_reg_loss


class SGL(LightGCN):
    """SGL: 边 Dropout 图对比学习。"""

    def __init__(self, num_users, num_items, config, norm_adj):
        super().__init__(num_users, num_items, config, norm_adj)
        self.drop_rate = config.get("drop_rate", 0.1)
        self.cl_weight = config.get("cl_weight", 0.05)
        self.cl_tau = config.get("cl_tau", 0.2)

    def _propagate_with_dropout(self):
        """带边 dropout 的图传播，返回 user/item 嵌入。"""
        # 随机 drop 边
        adj = self.norm_adj.coalesce()
        indices = adj.indices()
        nnz = indices.shape[1]
        keep = torch.rand(nnz, device=indices.device) > self.drop_rate
        kept_indices = indices[:, keep]
        kept_values = adj.values()[keep]

        dropped_adj = torch.sparse_coo_tensor(
            kept_indices, kept_values, adj.shape, device=adj.device
        ).coalesce()

        ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layers = [ego]
        x = ego
        for _ in range(self.num_layers):
            x = torch.sparse.mm(dropped_adj, x)
            layers.append(x)
        final = torch.stack(layers, dim=0).mean(dim=0)
        u, i = torch.split(final, [self.num_users, self.num_items], dim=0)
        return u, i

    def _contrastive_loss(self, emb1, emb2):
        """InfoNCE: emb1[i] 和 emb2[i] 是正对，其余是负对。"""
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        # 取 subset 做 batch 内对比
        n = min(emb1.shape[0], 2048)
        idx = torch.randperm(emb1.shape[0], device=emb1.device)[:n]
        e1, e2 = emb1[idx], emb2[idx]
        sim = (e1 @ e2.T) / self.cl_tau  # [n, n]
        labels = torch.arange(n, device=sim.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

    def compute_loss(self, batch, **kwargs):
        users, pos_items, neg_items = batch
        # BPR loss (standard LightGCN propagation)
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        ranking_loss = bpr_loss(pos_scores, neg_scores)
        reg_loss = l2_reg_loss([
            self.user_embedding(users), self.item_embedding(pos_items),
            self.item_embedding(neg_items),
        ], reduction="mean")
        reg_weight = kwargs.get("reg_weight", self.reg_weight)
        bpr_total = ranking_loss + reg_weight * reg_loss

        # Contrastive loss (two dropped views)
        u1, i1 = self._propagate_with_dropout()
        u2, i2 = self._propagate_with_dropout()
        cl_user = self._contrastive_loss(u1, u2)
        cl_item = self._contrastive_loss(i1, i2)
        cl_loss = cl_user + cl_item

        return bpr_total + self.cl_weight * cl_loss


class SGL_LLM(SGL):
    """SGL + LLM 相加融合。"""

    def __init__(self, num_users, num_items, config, norm_adj,
                 llm_user_emb=None, llm_item_emb=None):
        super().__init__(num_users, num_items, config, norm_adj)
        llm_dim = llm_user_emb.shape[1]
        self.llm_proj_user = nn.Linear(llm_dim, self.embedding_dim, bias=False)
        self.llm_proj_item = nn.Linear(llm_dim, self.embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.llm_proj_user.weight, gain=0.1)
        nn.init.xavier_uniform_(self.llm_proj_item.weight, gain=0.1)
        self.register_buffer("llm_user_raw", llm_user_emb.float())
        self.register_buffer("llm_item_raw", llm_item_emb.float())

    def _propagate(self):
        llm_u = self.llm_proj_user(self.llm_user_raw)
        llm_i = self.llm_proj_item(self.llm_item_raw)
        ego = torch.cat([
            self.user_embedding.weight + llm_u,
            self.item_embedding.weight + llm_i,
        ], dim=0)
        layers = [ego]
        x = ego
        for _ in range(self.num_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            layers.append(x)
        final = torch.stack(layers, dim=0).mean(dim=0)
        u, i = torch.split(final, [self.num_users, self.num_items], dim=0)
        return u, i

    def _propagate_with_dropout(self):
        llm_u = self.llm_proj_user(self.llm_user_raw)
        llm_i = self.llm_proj_item(self.llm_item_raw)
        ego = torch.cat([
            self.user_embedding.weight + llm_u,
            self.item_embedding.weight + llm_i,
        ], dim=0)
        adj = self.norm_adj.coalesce()
        idx = adj.indices()
        keep = torch.rand(idx.shape[1], device=idx.device) > self.drop_rate
        d_adj = torch.sparse_coo_tensor(idx[:, keep], adj.values()[keep], adj.shape).coalesce()
        layers = [ego]
        x = ego
        for _ in range(self.num_layers):
            x = torch.sparse.mm(d_adj, x)
            layers.append(x)
        final = torch.stack(layers, dim=0).mean(dim=0)
        u, i = torch.split(final, [self.num_users, self.num_items], dim=0)
        return u, i

    def count_parameters(self):
        return {"total_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "llm_frozen": self.llm_user_raw.numel() + self.llm_item_raw.numel()}


class SGL_LLM_HardBPR(SGL_LLM):
    """SGL + LLM + 语义感知 BPR。"""

    def __init__(self, num_users, num_items, config, norm_adj,
                 llm_user_emb=None, llm_item_emb=None):
        super().__init__(num_users, num_items, config, norm_adj, llm_user_emb, llm_item_emb)
        self.hard_bpr_beta = config.get("hard_bpr_beta", 0.5)
        i_norm = F.normalize(self.llm_item_raw, p=2, dim=1)
        self.register_buffer("_llm_item_norm", i_norm)

    def compute_loss(self, batch, **kwargs):
        users, pos_items, neg_items = batch
        # 一次传播
        all_u, all_i = self._propagate()
        u, pi, ni = all_u[users], all_i[pos_items], all_i[neg_items]
        pos_s = (u * pi).sum(dim=1)
        neg_s = (u * ni).sum(dim=1)

        # 语义加权 BPR
        sem_sim = (self._llm_item_norm[pos_items] * self._llm_item_norm[neg_items]).sum(dim=1)
        weights = 1.0 + self.hard_bpr_beta * sem_sim
        ranking_loss = -(weights * F.logsigmoid(pos_s - neg_s)).mean()

        reg_loss = l2_reg_loss([self.user_embedding(users), self.item_embedding(pos_items),
                                self.item_embedding(neg_items)], reduction="mean")
        reg_weight = kwargs.get("reg_weight", self.reg_weight)

        # Contrastive
        u1, i1 = self._propagate_with_dropout()
        u2, i2 = self._propagate_with_dropout()
        cl = self._contrastive_loss(u1, u2) + self._contrastive_loss(i1, i2)

        return ranking_loss + reg_weight * reg_loss + self.cl_weight * cl
