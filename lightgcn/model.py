from __future__ import annotations

import torch
from torch import nn


class LightGCN(nn.Module):
    """一个最小可运行版本的 LightGCN。

    LightGCN 的核心思想是：
    - 不使用复杂的特征变换层
    - 不使用非线性激活
    - 只保留图上传播邻居信息这一步
    - 最终把每一层传播得到的 embedding 做平均

    这也是它相对 NGCF 等更复杂图推荐模型更轻量的原因。
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int,
        norm_adj: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.norm_adj = norm_adj

        # 用户和物品都只保留一个可学习的 ID embedding。
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # 用较小方差的正态分布初始化，是推荐系统 embedding 的常见做法。
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def compute_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """执行 LightGCN 的多层图传播，并返回最终用户/物品表示。"""

        # 先把用户 embedding 和物品 embedding 拼在一起，
        # 因为在图里它们本来就是同一张二部图上的两类节点。
        all_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0,
        )

        # 第 0 层就是原始 embedding，本身也会参与最终平均。
        layer_outputs = [all_embeddings]

        x = all_embeddings
        for _ in range(self.num_layers):
            # 每做一次稀疏矩阵乘法，相当于做一轮图消息传播。
            x = torch.sparse.mm(self.norm_adj, x)
            layer_outputs.append(x)

        # LightGCN 最终把每一层的结果求平均，而不是只取最后一层。
        final_embeddings = torch.stack(layer_outputs, dim=0).mean(dim=0)
        user_emb, item_emb = torch.split(
            final_embeddings,
            [self.num_users, self.num_items],
            dim=0,
        )
        return user_emb, item_emb

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        reg_weight: float,
    ) -> torch.Tensor:
        """计算 BPR pairwise ranking loss。

        目标是让：
        `score(user, pos_item) > score(user, neg_item)`

        同时加入 embedding 的 L2 正则，避免向量无限增大。
        """

        user_emb, item_emb = self.compute_embeddings()
        u = user_emb[users]
        pos = item_emb[pos_items]
        neg = item_emb[neg_items]

        # 内积作为用户-物品匹配分数。
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        ranking_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        # 这里的正则使用的是“原始可学习 embedding”，
        # 而不是传播后的 embedding，这也是 BPR / MF 里常见的处理方式。
        reg_loss = (
            self.user_embedding(users).pow(2).sum(dim=1)
            + self.item_embedding(pos_items).pow(2).sum(dim=1)
            + self.item_embedding(neg_items).pow(2).sum(dim=1)
        ).mean()
        return ranking_loss + reg_weight * reg_loss

    @torch.no_grad()
    def predict(self) -> torch.Tensor:
        """一次性计算所有用户对所有物品的打分矩阵。

        返回形状是：
        `[num_users, num_items]`

        这样评估阶段就可以直接按用户取 Top-K。
        """

        user_emb, item_emb = self.compute_embeddings()
        return user_emb @ item_emb.t()
