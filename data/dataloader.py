"""PyTorch DataLoader 实现：支持 BPR 训练所需的负采样。

提供两种训练模式：
1. BPRBatchLoader：无限循环随机采样，每个 batch 内完成负采样
2. PointwiseDataset：标准 PyTorch Dataset，用于 BCE/MSE 训练
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

Interaction = Tuple[int, int]


# ── BPR 训练用的批量加载器 ────────────────────────────────


class BPRBatchLoader:
    """BPR 训练专用的批量数据加载器。

    每步从训练集中随机采样一批 (user, pos_item) 对，
    并为每个正样本随机采样一个用户未交互过的负样本物品。
    这是一个无限迭代器，适合 BPR 的训练模式。

    Parameters
    ----------
    train_pairs : list of (user, item)
        训练集交互对。
    train_user_items : dict, user → set of items
        训练集中每个用户的所有交互物品（用于避免负采样命中正样本）。
    num_items : int
        物品总数。
    batch_size : int
        批量大小。
    seed : int
        随机种子。
    device : str
        计算设备。
    """

    def __init__(
        self,
        train_pairs: List[Interaction],
        train_user_items: Dict[int, set],
        num_items: int,
        batch_size: int,
        seed: int = 2024,
        device: str = "cpu",
    ) -> None:
        self.train_pairs = train_pairs
        self.train_user_items = train_user_items
        self.num_items = num_items
        self.batch_size = batch_size
        self.device = device
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self._num_pairs = len(train_pairs)

    def __len__(self) -> int:
        """近似的每 epoch 步数（用于进度显示）。"""
        return max(1, self._num_pairs // self.batch_size)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """无限迭代器，每次 yield (users, pos_items, neg_items)。"""
        while True:
            yield self._sample_batch()

    def _sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机采样一个 BPR 三元组 batch。"""
        # 随机选择 batch_size 个训练交互
        indices = self.np_rng.randint(0, self._num_pairs, self.batch_size)
        users = []
        pos_items = []
        neg_items = []

        for idx in indices:
            user, pos_item = self.train_pairs[idx]
            # 负采样：随机选一个用户未交互过的物品
            neg_item = self.np_rng.randint(0, self.num_items)
            user_seen = self.train_user_items.get(user, set())
            while neg_item in user_seen:
                neg_item = self.np_rng.randint(0, self.num_items)
            users.append(user)
            pos_items.append(pos_item)
            neg_items.append(neg_item)

        # 直接转为张量并移到目标设备
        return (
            torch.tensor(users, dtype=torch.long, device=self.device),
            torch.tensor(pos_items, dtype=torch.long, device=self.device),
            torch.tensor(neg_items, dtype=torch.long, device=self.device),
        )

    def set_seed(self, seed: int) -> None:
        """重新设置随机种子（用于多次实验时确保可复现）。"""
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)


# ── Pointwise 训练用的 Dataset ────────────────────────────


class PointwiseTrainDataset(Dataset):
    """标准 PyTorch Dataset，用于 BCE/MSE 等 pointwise 训练。

    每个样本是一个 (user, item, label) 三元组，
    其中 label=1 表示正样本（训练集交互），label=0 表示负样本。
    负样本通过在线采样生成。
    """

    def __init__(
        self,
        train_pairs: List[Interaction],
        train_user_items: Dict[int, set],
        num_items: int,
        num_negatives: int = 1,
        seed: int = 2024,
    ) -> None:
        self.train_pairs = train_pairs
        self.train_user_items = train_user_items
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.train_pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        """返回 (user, item, label)。"""
        user, pos_item = self.train_pairs[idx]
        # 正样本
        samples = [(user, pos_item, 1)]

        # 负样本
        user_seen = self.train_user_items.get(user, set())
        for _ in range(self.num_negatives):
            neg_item = self.rng.randint(0, self.num_items)
            while neg_item in user_seen:
                neg_item = self.rng.randint(0, self.num_items)
            samples.append((user, neg_item, 0))

        # 为简单起见，返回单个样本（如果 num_negatives > 1，
        # 外层 DataLoader 的 collate_fn 可以聚合多个样本）
        # 这里返回第一个正样本，负样本也在每个 epoch 中轮换
        idx_in_group = self.rng.randint(0, len(samples))
        return samples[idx_in_group]


# ── 语义硬负采样 BPR 加载器 ────────────────────────────────


class HardBPRBatchLoader(BPRBatchLoader):
    """带语义硬负采样的 BPR 加载器。

    在随机负采样的基础上，以一定概率从物品的 LLM 语义近邻中选择负样本，
    让模型在训练时面对更困难的区分任务。

    Parameters
    ----------
    semantic_topk : ndarray [num_items, top_k]
        每个物品的 top-K 语义近邻索引（预计算）。
    hard_neg_prob : float
        使用语义硬负采样的概率（0~1）。
    """

    def __init__(
        self,
        train_pairs: List[Interaction],
        train_user_items: Dict[int, set],
        num_items: int,
        batch_size: int,
        seed: int = 2024,
        device: str = "cpu",
        semantic_topk: np.ndarray | None = None,
        hard_neg_prob: float = 0.5,
    ) -> None:
        super().__init__(train_pairs, train_user_items, num_items, batch_size, seed, device)
        self.semantic_topk = semantic_topk
        self.hard_neg_prob = hard_neg_prob
        self._topk_size = semantic_topk.shape[1] if semantic_topk is not None else 0

    def _sample_hard_neg(self, pos_item: int, user_seen: set) -> int:
        """从 pos_item 的语义近邻中选一个用户未交互的物品作为硬负样本。"""
        candidates = self.semantic_topk[pos_item]  # [K]
        # 随机打乱候选顺序
        shuffled = self.np_rng.permutation(candidates)
        for cand in shuffled:
            if cand not in user_seen:
                return int(cand)
        # 所有 top-k 邻居都已被交互 → 回退到随机
        return None

    def _sample_batch(self):
        indices = self.np_rng.randint(0, self._num_pairs, self.batch_size)
        users = []
        pos_items = []
        neg_items = []

        for idx in indices:
            user, pos_item = self.train_pairs[idx]
            user_seen = self.train_user_items.get(user, set())

            # 以 hard_neg_prob 的概率使用语义硬负采样
            if self.semantic_topk is not None and self.np_rng.random() < self.hard_neg_prob:
                hard_neg = self._sample_hard_neg(pos_item, user_seen)
                if hard_neg is not None:
                    neg_item = hard_neg
                else:
                    # 回退到随机
                    neg_item = self.np_rng.randint(0, self.num_items)
                    while neg_item in user_seen:
                        neg_item = self.np_rng.randint(0, self.num_items)
            else:
                neg_item = self.np_rng.randint(0, self.num_items)
                while neg_item in user_seen:
                    neg_item = self.np_rng.randint(0, self.num_items)

            users.append(user)
            pos_items.append(pos_item)
            neg_items.append(neg_item)

        return (
            torch.tensor(users, dtype=torch.long, device=self.device),
            torch.tensor(pos_items, dtype=torch.long, device=self.device),
            torch.tensor(neg_items, dtype=torch.long, device=self.device),
        )


# ── 便捷函数 ─────────────────────────────────────────────


def create_bpr_loader(
    train_pairs: List[Interaction],
    train_user_items: Dict[int, set],
    num_items: int,
    batch_size: int,
    seed: int = 2024,
    device: str = "cpu",
) -> BPRBatchLoader:
    """快速创建 BPR 训练的批量加载器。"""
    return BPRBatchLoader(
        train_pairs=train_pairs,
        train_user_items=train_user_items,
        num_items=num_items,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )


# ── 图结构构建（供 LightGCN 等图模型使用）───────────────────


def build_normalized_adj(
    num_users: int,
    num_items: int,
    interactions: Sequence[Interaction],
    device: str = "cpu",
) -> torch.Tensor:
    """构建 LightGCN 使用的对称归一化邻接矩阵 D^{-1/2} A D^{-1/2}。

    图结构为 user-item 二部图：
    - 用户节点编号：[0, num_users)
    - 物品节点编号：[num_users, num_users + num_items)
    - 每条交互 (u, i) 添加无向边 u ↔ i

    归一化方式：对每条边 (u, v)，权重 = 1 / sqrt(deg(u) * deg(v))

    Parameters
    ----------
    num_users : int
        用户数。
    num_items : int
        物品数。
    interactions : sequence of (user, item)
        用于构建图的交互对（通常仅使用训练集）。
    device : str
        计算设备。

    Returns
    -------
    norm_adj : sparse FloatTensor, shape [N+M, N+M]
        归一化后的稀疏邻接矩阵。
    """
    total_nodes = num_users + num_items
    row_indices: List[int] = []
    col_indices: List[int] = []

    for user, item in interactions:
        item_node = num_users + item  # 物品节点整体平移
        # 无向边：user ↔ item_node
        row_indices.extend([user, item_node])
        col_indices.extend([item_node, user])

    indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
    values = torch.ones(len(row_indices), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(
        indices, values, (total_nodes, total_nodes)
    ).coalesce()

    # 计算每个节点的度数 → D^{-1/2}
    degrees = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = degrees.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0  # 孤立节点 → 权重为 0

    # 对每条边做 D^{-1/2} A D^{-1/2} 归一化
    row, col = adj.indices()
    norm_values = deg_inv_sqrt[row] * adj.values() * deg_inv_sqrt[col]
    norm_adj = torch.sparse_coo_tensor(
        adj.indices(), norm_values, adj.shape,
        dtype=torch.float32, device=device,
    )
    return norm_adj.coalesce()
