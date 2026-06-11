"""LightGCN-LLM-Distill：热→冷协同知识蒸馏。

在 LightGCN-LLM（相加融合）基础上增加蒸馏损失：
- 对交互稀疏的冷门物品，用 LLM 语义相似度找到热门邻居
- 让冷门物品的 CF 表示逼近热门邻居的加权平均
- 热门物品的协同信号通过 LLM 语义桥梁传递给冷门物品

对比实验：
- LightGCN（纯CF）：随机初始化
- LightGCN-LLM（相加）：ID + Proj(LLM)
- LightGCN-LLM-Distill（本文）：ID + Proj(LLM) + 热→冷蒸馏
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.traditional import bpr_loss, l2_reg_loss
from models.llm_enhanced.lightgcn_llm import LightGCN_LLM


class LightGCN_LLM_Distill(LightGCN_LLM):
    """带热→冷蒸馏的 LightGCN-LLM。

    Parameters
    ----------
    num_users, num_items, config, norm_adj, llm_user_emb, llm_item_emb, freeze_llm :
        与 LightGCN_LLM 相同。
    neighbors_path : str, optional
        semantic_neighbors.pkl 文件路径。
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
        neighbors_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            num_users, num_items, config, norm_adj,
            llm_user_emb, llm_item_emb, freeze_llm,
        )

        self.distill_weight = config.get("distill_weight", 0.1)
        self._cold_indices = None
        self._neighbor_indices = None
        self._neighbor_weights = None

        if neighbors_path is not None:
            self._load_neighbors(neighbors_path, num_items)

    def _load_neighbors(self, path: str, num_items: int) -> None:
        """加载预计算的语义邻居映射并转换为 GPU 张量。"""
        p = Path(path)
        if not p.exists():
            print(f"[Distill] Neighbor file not found: {p}, disabling distillation.")
            return

        with open(p, "rb") as f:
            data = pickle.load(f)

        neighbors_dict = data["neighbors"]
        k = data["k"]
        num_cold = len(neighbors_dict)

        cold_list: List[int] = []
        neigh_list: List[List[int]] = []
        weight_list: List[List[float]] = []

        for cold_id, info in sorted(neighbors_dict.items()):
            cold_list.append(cold_id)
            neigh_list.append(info["neighbors"])
            # 用余弦相似度做加权平均的权重
            sims = info["sims"]
            w = torch.tensor(sims, dtype=torch.float32).softmax(dim=0).tolist()
            weight_list.append(w)

        self._cold_indices = torch.tensor(cold_list, dtype=torch.long)
        self._neighbor_indices = torch.tensor(neigh_list, dtype=torch.long)  # [C, K]
        self._neighbor_weights = torch.tensor(weight_list, dtype=torch.float32)  # [C, K]

        print(
            f"[Distill] Loaded {num_cold} cold items"
            f" (threshold={data['cold_threshold']}, k={k}),"
            f" distill_weight={self.distill_weight}"
        )

    def compute_loss(self, batch, **kwargs) -> torch.Tensor:
        """BPR loss + L2 reg + 热→冷蒸馏损失。

        只做一次图传播，同时用于 BPR 和蒸馏，避免重复计算。
        """
        users, pos_items, neg_items = batch

        # ── 一次图传播，拿到全部嵌入 ──
        all_user_emb, all_item_emb = self._propagate()  # [N,d], [M,d]

        # ── BPR loss ──
        user_emb = all_user_emb[users]
        pos_emb = all_item_emb[pos_items]
        neg_emb = all_item_emb[neg_items]
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        ranking_loss = bpr_loss(pos_scores, neg_scores)

        reg_loss = l2_reg_loss(
            [
                self.user_embedding(users),
                self.item_embedding(pos_items),
                self.item_embedding(neg_items),
            ],
            reduction="mean",
        )
        reg_weight = kwargs.get("reg_weight", self.reg_weight)
        bpr_total = ranking_loss + reg_weight * reg_loss

        # ── 蒸馏 loss：冷门物品 → 热门邻居加权平均 ──
        if self._cold_indices is not None and len(self._cold_indices) > 0:
            # 把冷门索引和邻居索引移到当前设备
            cold_idx = self._cold_indices.to(all_item_emb.device)
            neigh_idx = self._neighbor_indices.to(all_item_emb.device)
            neigh_w = self._neighbor_weights.to(all_item_emb.device)

            cold_embs = all_item_emb[cold_idx]                    # [C, d]
            neighbor_embs = all_item_emb[neigh_idx]               # [C, K, d]
            # detach: 冷门追热门，热门不被冷门拉偏
            target = (neighbor_embs * neigh_w.unsqueeze(-1)).sum(dim=1).detach()
            distill_loss = F.mse_loss(cold_embs, target)

            return bpr_total + self.distill_weight * distill_loss

        return bpr_total

    def count_parameters(self) -> dict:
        """返回参数明细（蒸馏不增加参数）。"""
        base = super().count_parameters()
        base["distill_cold_items"] = (
            len(self._cold_indices) if self._cold_indices is not None else 0
        )
        return base
