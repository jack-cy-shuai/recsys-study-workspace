"""推荐模型统一抽象基类。

定义所有推荐模型必须实现的核心接口，以及通用的训练/评估逻辑。
所有基线模型和创新模型都必须继承此类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch import nn


class BaseRecommender(nn.Module, ABC):
    """推荐系统模型的抽象基类。

    提供了统一的接口约定：
    - 模型构建：__init__ 中接收 num_users, num_items, config
    - 前向传播：forward(user, item) 返回指定用户-物品对的分数
    - 嵌入获取：get_user_embeddings(), get_item_embeddings()
    - 全量预测：predict_all() 返回完整打分矩阵
    - 损失计算：compute_loss(batch) 返回训练损失
    - 优化器配置：configure_optimizers()

    子类至少需要实现：
    - forward()
    - compute_loss()
    - get_user_embeddings()
    - get_item_embeddings()
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.config = config
        self.embedding_dim = config.get("embedding_dim", 64)
        self._device = config.get("device", "cpu")

    @abstractmethod
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """计算指定用户-物品对的预测分数。

        Parameters
        ----------
        users : LongTensor, shape [batch_size]
            用户索引。
        items : LongTensor, shape [batch_size]
            物品索引。

        Returns
        -------
        scores : Tensor, shape [batch_size]
            每个 (user, item) 对的预测分数。
        """
        ...

    @abstractmethod
    def get_user_embeddings(self) -> torch.Tensor:
        """获取最终的用户嵌入矩阵。

        Returns
        -------
        user_emb : Tensor, shape [num_users, embedding_dim]
        """
        ...

    @abstractmethod
    def get_item_embeddings(self) -> torch.Tensor:
        """获取最终的物品嵌入矩阵。

        Returns
        -------
        item_emb : Tensor, shape [num_items, embedding_dim]
        """
        ...

    @abstractmethod
    def compute_loss(self, batch, **kwargs) -> torch.Tensor:
        """从 batch 计算训练损失。

        子类根据自身训练范式（BPR / BCE / MSE）实现具体逻辑。
        """
        ...

    @torch.no_grad()
    def predict_all(self) -> torch.Tensor:
        """返回完整的用户-物品打分矩阵。

        通过对用户嵌入和物品嵌入做内积得到：
        score_matrix[u, i] = <user_emb[u], item_emb[i]>

        Returns
        -------
        score_matrix : Tensor, shape [num_users, num_items]
        """
        user_emb = self.get_user_embeddings()
        item_emb = self.get_item_embeddings()
        return user_emb @ item_emb.t()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """返回优化器实例。

        默认使用 Adam，子类可按需重写。
        """
        lr = self.config.get("lr", 1e-3)
        weight_decay = self.config.get("weight_decay", 1e-4)
        return torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

    @property
    def device(self) -> torch.device:
        """返回模型参数所在的设备。"""
        return next(self.parameters()).device

    def count_parameters(self) -> int:
        """统计可训练参数总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
