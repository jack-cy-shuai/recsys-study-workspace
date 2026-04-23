from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

Interaction = Tuple[int, int]


@torch.no_grad()
def evaluate_model(
    score_matrix: torch.Tensor,
    target_pairs: Sequence[Interaction],
    train_user_items: Dict[int, set],
    ks: Iterable[int],
) -> Dict[str, float]:
    """在给定目标集上计算 Recall@K 和 NDCG@K。

    参数说明：
    - `score_matrix[user, item]`：模型预测分数
    - `target_pairs`：当前评估目标，通常是 val 或 test
    - `train_user_items`：训练集中已见物品，用于评估时屏蔽
    - `ks`：需要统计的 Top-K 列表

    这里采用的是按用户平均的评估方式。
    """

    ks = sorted(set(int(k) for k in ks))
    metrics = {f"Recall@{k}": 0.0 for k in ks}
    metrics.update({f"NDCG@{k}": 0.0 for k in ks})

    # 先把目标交互按用户聚合，方便后面逐个用户评估。
    user_targets: Dict[int, List[int]] = {}
    for user, item in target_pairs:
        user_targets.setdefault(user, []).append(item)

    for user, items in user_targets.items():
        scores = score_matrix[user].clone()

        # 推荐评估时通常不把训练集里已经见过的物品算进候选集，
        # 否则模型可能只是把历史物品排得很高，看起来指标虚高。
        if train_user_items[user]:
            seen_items = torch.tensor(list(train_user_items[user]), device=scores.device)
            scores[seen_items] = -1e9

        max_k = ks[-1]
        top_items = torch.topk(scores, k=max_k).indices.tolist()
        target_set = set(items)

        for k in ks:
            top_k = top_items[:k]
            hits = [1 if item in target_set else 0 for item in top_k]
            num_hits = sum(hits)

            # Recall = 在 Top-K 里命中的目标物品数 / 该用户真实目标物品数
            metrics[f"Recall@{k}"] += num_hits / len(target_set)

            dcg = 0.0
            for rank, hit in enumerate(hits, start=1):
                if hit:
                    # 越靠前命中的物品，贡献越大。
                    dcg += 1.0 / math.log2(rank + 1)
            ideal_hits = min(len(target_set), k)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            metrics[f"NDCG@{k}"] += dcg / idcg if idcg > 0 else 0.0

    # 最终按用户数求平均，得到整体指标。
    num_users = max(len(user_targets), 1)
    return {name: value / num_users for name, value in metrics.items()}
