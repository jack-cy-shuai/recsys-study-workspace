"""LightGCN-LLM-HardBPR：语义感知的 BPR 损失。

核心思路：BPR 随机负采样产生大量"容易"的负样本（正样本 vs 完全无关物品），
训练信号浪费在区分"三体 vs 菜谱"上。真正需要区分的是"三体 vs 球状闪电"——
语义相近的同类物品之间的细微差异。

方法：用 LLM frozen embedding 的余弦相似度增强 BPR：
  - margin = beta * cosine_sim(pos_llm, neg_llm)
  - loss = -log(sigmoid(pos_score - neg_score - margin))
  语义越近的负样本，需要的 margin 越大 → 模型被迫学更精细的语义区分。

对比：
  - LightGCN：随机初始化 + 标准 BPR
  - LightGCN-LLM：相加融合 + 标准 BPR
  - LightGCN-LLM-HardBPR：相加融合 + 语义感知 BPR (本文)
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from models.llm_enhanced.lightgcn_llm import LightGCN_LLM
from losses.traditional import l2_reg_loss


class LightGCN_LLM_HardBPR(LightGCN_LLM):
    """语义感知 BPR 的 LightGCN-LLM。

    Parameters
    ----------
    与 LightGCN_LLM 相同。
    hard_bpr_mode : str
        "margin" — 语义 margin 增强 BPR
        "weight" — 语义加权 BPR
    hard_bpr_beta : float
        语义信号的强度系数。
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
        super().__init__(
            num_users, num_items, config, norm_adj,
            llm_user_emb, llm_item_emb, freeze_llm,
        )

        self.hard_bpr_mode = config.get("hard_bpr_mode", "margin")
        self.hard_bpr_beta = config.get("hard_bpr_beta", 0.5)

        # 缓存归一化的 LLM 物品 embedding 用于快速 cosine 计算
        llm_norm = F.normalize(self.llm_item_raw.float(), p=2, dim=1)
        self.register_buffer("_llm_item_norm", llm_norm)

    def compute_loss(self, batch, **kwargs) -> torch.Tensor:
        """语义感知 BPR + L2 正则。"""
        users, pos_items, neg_items = batch

        # ── 一次图传播 ──
        all_user_emb, all_item_emb = self._propagate()
        user_emb = all_user_emb[users]
        pos_emb = all_item_emb[pos_items]
        neg_emb = all_item_emb[neg_items]

        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)

        # ── 语义感知的 BPR ──
        if self.hard_bpr_mode == "margin":
            # 语义越近 → 需要的 margin 越大
            pos_llm = self._llm_item_norm[pos_items]
            neg_llm = self._llm_item_norm[neg_items]
            sem_sim = (pos_llm * neg_llm).sum(dim=1)  # [B], cosine similarity
            margin = self.hard_bpr_beta * sem_sim
            diff = pos_scores - neg_scores - margin
            ranking_loss = -F.logsigmoid(diff).mean()

        elif self.hard_bpr_mode == "weight":
            # 语义越近 → loss 权重越大（hard negative 更重要）
            pos_llm = self._llm_item_norm[pos_items]
            neg_llm = self._llm_item_norm[neg_items]
            sem_sim = (pos_llm * neg_llm).sum(dim=1)  # [B]
            weights = 1.0 + self.hard_bpr_beta * sem_sim  # [B]
            diff = pos_scores - neg_scores
            per_sample = -F.logsigmoid(diff)
            ranking_loss = (weights * per_sample).mean()

        else:
            # 标准 BPR
            ranking_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # ── L2 正则 ──
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

    def count_parameters(self) -> dict:
        base = super().count_parameters()
        base["hard_bpr_mode"] = self.hard_bpr_mode
        base["hard_bpr_beta"] = self.hard_bpr_beta
        return base
