"""推荐系统标准评估指标：Recall@K, NDCG@K, MRR@K, Precision@K, Hit@K。

所有指标均采用按用户平均（macro-averaged）的方式计算：
对每个用户单独计算指标值，然后取所有用户的算术平均。
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Set, Tuple

import torch


def recall_at_k(pred_items: List[int], true_items: Set[int]) -> float:
    """Recall@K = |预测命中| / |真实物品|。

    对每个用户：在 Top-K 中命中的正例数 / 该用户真实交互数。
    """
    if len(true_items) == 0:
        return 0.0
    hits = sum(1 for item in pred_items if item in true_items)
    return hits / len(true_items)


def ndcg_at_k(pred_items: List[int], true_items: Set[int]) -> float:
    """NDCG@K = DCG@K / IDCG@K。

    DCG 使用标准对数折现：rel_i / log2(rank + 1)，rel_i ∈ {0, 1}。
    IDCG 假设所有正例排在最前面。
    """
    k = len(pred_items)
    dcg = 0.0
    for rank, item in enumerate(pred_items, start=1):
        if item in true_items:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(true_items), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(pred_items: List[int], true_items: Set[int]) -> float:
    """MRR@K = 1 / rank_of_first_hit。

    若 Top-K 中无命中则返回 0。rank 从 1 开始计数。
    """
    for rank, item in enumerate(pred_items, start=1):
        if item in true_items:
            return 1.0 / rank
    return 0.0


def precision_at_k(pred_items: List[int], true_items: Set[int]) -> float:
    """Precision@K = |预测命中| / K。"""
    if len(pred_items) == 0:
        return 0.0
    hits = sum(1 for item in pred_items if item in true_items)
    return hits / len(pred_items)


def hit_at_k(pred_items: List[int], true_items: Set[int]) -> float:
    """Hit@K：Top-K 中至少有一个命中则返回 1，否则返回 0。"""
    for item in pred_items:
        if item in true_items:
            return 1.0
    return 0.0


# ── 统一评估入口 ──────────────────────────────────────────


@torch.no_grad()
def evaluate_model(
    score_matrix: torch.Tensor,
    target_pairs: List[Tuple[int, int]],
    train_user_items: Dict[int, set],
    ks: Iterable[int],
) -> Dict[str, float]:
    """对所有目标用户计算全部标准 Top-K 指标。

    Parameters
    ----------
    score_matrix : Tensor, shape [num_users, num_items]
        模型预测的完整打分矩阵。
    target_pairs : list of (user, item)
        评估集（验证或测试）的交互对。
    train_user_items : dict, user → set of items
        用户在训练集中已交互的物品，用于评估时屏蔽（避免推荐已见过的物品）。
    ks : iterable of int
        需要计算的 K 值列表。

    Returns
    -------
    metrics : dict
        以 "指标名@K" 为键的字典，所有值都是按用户平均的标量。
        包含：Recall@{k}, NDCG@{k}, MRR@{k}, Precision@{k}, Hit@{k}
    """
    ks = sorted(set(int(k) for k in ks))
    max_k = ks[-1]

    # 初始化累加器：每个指标都是按用户求和，最后再除以用户数
    accumulators: Dict[str, float] = {}
    for k in ks:
        for prefix in ["Recall", "NDCG", "MRR", "Precision", "Hit"]:
            accumulators[f"{prefix}@{k}"] = 0.0

    # 将目标交互按用户分组
    user_targets: Dict[int, Set[int]] = {}
    for user, item in target_pairs:
        user_targets.setdefault(user, set()).add(item)

    for user, true_items in user_targets.items():
        # 获取该用户对所有物品的打分
        scores = score_matrix[user].clone()

        # 屏蔽训练集已见物品
        seen = train_user_items.get(user, set())
        if seen:
            seen_tensor = torch.tensor(list(seen), device=scores.device, dtype=torch.long)
            scores[seen_tensor] = -float("inf")

        # 取 Top-max_k 物品索引
        top_items = torch.topk(scores, k=max_k).indices.tolist()

        # 对每个 k 计算所有指标
        for k in ks:
            top_k = top_items[:k]
            accumulators[f"Recall@{k}"] += recall_at_k(top_k, true_items)
            accumulators[f"NDCG@{k}"] += ndcg_at_k(top_k, true_items)
            accumulators[f"MRR@{k}"] += mrr_at_k(top_k, true_items)
            accumulators[f"Precision@{k}"] += precision_at_k(top_k, true_items)
            accumulators[f"Hit@{k}"] += hit_at_k(top_k, true_items)

    # 按用户数平均
    num_users = max(len(user_targets), 1)
    return {name: value / num_users for name, value in accumulators.items()}
