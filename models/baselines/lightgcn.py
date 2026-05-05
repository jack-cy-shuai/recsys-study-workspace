"""LightGCN：轻量图卷积协同过滤模型。

原论文：He et al., "LightGCN: Simplifying and Powering Graph Convolution Network
for Recommendation", SIGIR 2020.

核心思想：
- 去除 NGCF 中的特征变换矩阵和非线性激活函数
- 仅保留图上的邻居消息传播（多层线性聚合）
- 最终将各层 embedding 取平均作为用户/物品表示
- 训练使用 BPR pairwise ranking loss

数学推导（单层传播）：
    E^{(k+1)} = (D^{-1/2} A D^{-1/2}) E^{(k)}
    最终表示 = 1/(K+1) * sum_{k=0}^{K} E^{(k)}
    其中 K 为传播层数，E^{(0)} 为可学习的 ID embedding。
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from models.base_model import BaseRecommender
from losses.traditional import bpr_loss, l2_reg_loss


class LightGCN(BaseRecommender):
    """LightGCN 模型。

    使用多层图传播来聚合邻居信息，最终通过平均各层嵌入得到用户和物品的表示。

    Parameters
    ----------
    num_users : int
        用户数量。
    num_items : int
        物品数量。
    config : dict
        配置字典，必须包含 "embedding_dim" 和 "num_layers"，
        可选包含 "reg_weight" 等。
    norm_adj : torch.sparse.FloatTensor
        归一化后的稀疏邻接矩阵，形状 [num_users + num_items, num_users + num_items]。
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        config: Dict[str, Any],
        norm_adj: torch.Tensor,
    ) -> None:
        super().__init__(num_users, num_items, config)
        self.num_layers = config["num_layers"]
        self.reg_weight = config.get("reg_weight", 1e-4)

        # ── 注册归一化邻接矩阵 ──
        # 用 register_buffer 确保它随模型一起移动到 GPU
        self.register_buffer("norm_adj", norm_adj.coalesce())

        # ── ID Embedding ──
        # 使用均值为 0、标准差 0.1 的正态分布初始化，
        # 这是 LightGCN 论文中的标准做法。
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)
        self._init_embeddings()

        # 缓存上一次计算得到的嵌入（用于避免同一个 epoch 内重复图传播）
        self._cached_user_emb: torch.Tensor | None = None
        self._cached_item_emb: torch.Tensor | None = None

    def _init_embeddings(self) -> None:
        """用正态分布初始化 embedding。"""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """执行 LightGCN 的多层图传播，返回 (user_emb, item_emb)。

        这是 LightGCN 的核心计算：
        1. 将用户和物品 embedding 拼接
        2. 进行 K 轮稀疏矩阵乘法（图传播）
        3. 将所有层的输出取平均
        4. 拆分回用户和物品部分
        """
        # 拼接初始 embedding：前 num_users 行是用户，后 num_items 行是物品
        ego_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0,
        )  # shape: [N+M, d]

        layer_embeddings = [ego_embeddings]
        x = ego_embeddings

        # LightGCN 的关键设计：每层只做稀疏乘法，不加变换矩阵和激活函数
        for _ in range(self.num_layers):
            x = torch.sparse.mm(self.norm_adj, x)  # [N+M, d] — 图消息传播
            layer_embeddings.append(x)

        # 所有层取平均：E_final = 1/(K+1) * sum_{k=0}^{K} E^{(k)}
        # stack → mean 比循环累加再除法更 Pythonic
        final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)

        # 拆分为用户和物品部分
        user_emb, item_emb = torch.split(
            final_embeddings, [self.num_users, self.num_items], dim=0
        )
        return user_emb, item_emb

    def get_user_embeddings(self) -> torch.Tensor:
        """获取最终的用户嵌入矩阵。使用缓存避免重复计算。"""
        if self._cached_user_emb is None:
            self._cached_user_emb, self._cached_item_emb = self._propagate()
        return self._cached_user_emb

    def get_item_embeddings(self) -> torch.Tensor:
        """获取最终的物品嵌入矩阵。"""
        if self._cached_item_emb is None:
            self._cached_user_emb, self._cached_item_emb = self._propagate()
        return self._cached_item_emb

    def invalidate_cache(self) -> None:
        """清除嵌入缓存。

        每个 epoch 开始时应调用此方法，以确保使用最新更新的参数计算嵌入。
        """
        self._cached_user_emb = None
        self._cached_item_emb = None

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """计算指定用户-物品对的预测分数。

        分数 = <user_emb[u], item_emb[i]>（内积）
        """
        user_emb = self.get_user_embeddings()[users]  # [B, d]
        item_emb = self.get_item_embeddings()[items]  # [B, d]
        return (user_emb * item_emb).sum(dim=1)  # [B]

    def compute_loss(self, batch, **kwargs) -> torch.Tensor:
        """计算 BPR pairwise ranking loss + L2 正则化。

        接收一个 batch = (users, pos_items, neg_items)，
        返回标量损失张量。

        BPR 损失的正则化使用原始的第 0 层 embedding（而非传播后的），
        这与 LightGCN 论文中的实现一致（见论文 Section 3.3）。
        """
        users, pos_items, neg_items = batch

        # 1. BPR pairwise loss — 核心排序损失
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        ranking_loss = bpr_loss(pos_scores, neg_scores)  # 默认 mean reduction

        # 2. L2 正则化 — 使用原始的第 0 层 embedding
        # 每个 batch 只对参与计算的 embedding 做正则化
        reg_loss = l2_reg_loss(
            [
                self.user_embedding(users),
                self.item_embedding(pos_items),
                self.item_embedding(neg_items),
            ],
            reduction="mean",
        )
        reg_weight = kwargs.get("reg_weight", self.reg_weight)
        return ranking_loss + reg_weight * reg_loss
