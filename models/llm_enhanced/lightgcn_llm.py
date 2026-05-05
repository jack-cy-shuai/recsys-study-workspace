"""LightGCN-LLM：融合 LLM 语义 Embedding 的增强版 LightGCN。

实现思路（相加融合）：
1. 将预计算的 LLM embedding（768维）通过线性层投影到 CF embedding 空间（64维）
2. 在每一层图传播前，将投影后的 LLM embedding 与可学习的 ID embedding 相加
3. 后续图传播和 BPR 训练与原始 LightGCN 完全一致

对比实验设计：
- LightGCN（纯 CF）：随机初始化 ID embedding → 图传播 → BPR
- LightGCN-LLM（本文）：ID emb + Proj(LLM emb) → 图传播 → BPR

唯一变量是初始化信息来源，架构完全相同，确保公平对比。
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from models.baselines.lightgcn import LightGCN


class LightGCN_LLM(LightGCN):
    """融合 LLM 语义 Embedding 的 LightGCN。

    在标准 LightGCN 基础上增加：
    - 线性投影层：将 LLM embedding（d_llm）映射到 CF embedding 维度（d_cf）
    - 相加融合：ID embedding + 投影后的 LLM embedding 作为节点初始特征

    Parameters
    ----------
    num_users, num_items, config, norm_adj : 与 LightGCN 相同
    llm_user_emb : Tensor [num_users, llm_dim]
        预计算的用户 LLM 语义向量（来自 RLMRec 或其他来源）。
    llm_item_emb : Tensor [num_items, llm_dim]
        预计算的商品 LLM 语义向量。
    freeze_llm : bool
        是否冻结 LLM embedding（默认 True，仅训练投影层）。
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        config: Dict[str, Any],
        norm_adj: torch.Tensor,
        llm_user_emb: torch.Tensor,
        llm_item_emb: torch.Tensor,
        freeze_llm: bool = True,
    ) -> None:
        super().__init__(num_users, num_items, config, norm_adj)

        llm_dim = llm_user_emb.shape[1]
        self.llm_dim = llm_dim

        # ── 线性投影层：d_llm → d_cf ──
        # 使用无偏置线性层，确保投影是纯线性变换
        self.llm_proj_user = nn.Linear(llm_dim, self.embedding_dim, bias=False)
        self.llm_proj_item = nn.Linear(llm_dim, self.embedding_dim, bias=False)

        # 小权重初始化：让模型逐步学会利用语义信息
        nn.init.xavier_uniform_(self.llm_proj_user.weight, gain=0.1)
        nn.init.xavier_uniform_(self.llm_proj_item.weight, gain=0.1)

        # ── 注册 LLM embedding（不可训练） ──
        self.register_buffer("llm_user_raw", llm_user_emb.float())
        self.register_buffer("llm_item_raw", llm_item_emb.float())
        self._freeze_llm = freeze_llm

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """覆盖父类的图传播：在初始特征中融入 LLM 语义 embedding。

        关键改动仅在第 0 层：ego_embeddings = ID_emb + Proj(LLM_emb)
        后续 K 层图传播与 LightGCN 完全相同。
        """

        # 投影 LLM embedding 到 CF 维度
        llm_user = self.llm_proj_user(self.llm_user_raw)  # [N, d_cf]
        llm_item = self.llm_proj_item(self.llm_item_raw)  # [M, d_cf]

        # ── 相加融合：ID embedding + 投影后的 LLM embedding ──
        user_init = self.user_embedding.weight + llm_user
        item_init = self.item_embedding.weight + llm_item

        # 拼接为统一的节点特征矩阵（与父类格式一致）
        ego_embeddings = torch.cat([user_init, item_init], dim=0)  # [N+M, d_cf]

        # ── 图传播：与 LightGCN 完全相同的逻辑 ──
        layer_embeddings = [ego_embeddings]
        x = ego_embeddings
        for _ in range(self.num_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            layer_embeddings.append(x)

        final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        user_emb, item_emb = torch.split(
            final_embeddings, [self.num_users, self.num_items], dim=0
        )
        return user_emb, item_emb

    def count_parameters(self) -> dict:
        """返回各组件参数量的明细。"""
        base_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        # 分别统计
        id_params = (
            sum(p.numel() for p in self.user_embedding.parameters())
            + sum(p.numel() for p in self.item_embedding.parameters())
        )
        proj_params = (
            sum(p.numel() for p in self.llm_proj_user.parameters())
            + sum(p.numel() for p in self.llm_proj_item.parameters())
        )
        return {
            "total_trainable": base_params,
            "id_embedding": id_params,
            "llm_projection": proj_params,
            "llm_embedding (frozen)": (
                self.llm_user_raw.numel() + self.llm_item_raw.numel()
            ),
        }

    def get_llm_fusion_weight(self) -> dict:
        """返回 LLM 投影层的权重统计，用于分析模型对语义信息的利用程度。"""
        return {
            "user_proj_norm": self.llm_proj_user.weight.norm().item(),
            "item_proj_norm": self.llm_proj_item.weight.norm().item(),
            "user_id_norm": self.user_embedding.weight.norm().item(),
            "item_id_norm": self.item_embedding.weight.norm().item(),
        }
