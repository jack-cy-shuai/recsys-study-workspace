from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


# 一条交互样本统一表示为 `(user_index, item_index)`。
# 注意这里保存的是重映射后的连续索引，不一定等于原始 CSV 里的 ID。
Interaction = Tuple[int, int]


@dataclass
class DatasetBundle:
    """训练与评估阶段需要反复使用的数据集合。

    之所以封装成 dataclass，是为了让主训练脚本里只保留一份 `dataset` 对象，
    访问起来更清晰，也方便后续继续扩展更多字段。
    """

    num_users: int
    num_items: int
    train_pairs: List[Interaction]
    val_pairs: List[Interaction]
    test_pairs: List[Interaction]
    train_user_items: Dict[int, set]
    all_user_items: Dict[int, set]
    norm_adj: torch.Tensor


def load_interactions(csv_path: str | Path) -> Dict[int, List[Tuple[int, int, str]]]:
    """从 CSV 读取原始交互，并按用户分组保存。

    返回结构：
    `user_id -> [(timestamp, item_id, split), ...]`

    这里先保留原始 user/item ID，不急着映射成连续索引；
    这样做是为了让读取阶段更直观，也便于替换成你自己的数据文件。
    """

    user_rows: Dict[int, List[Tuple[int, int, str]]] = {}
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user = int(row["user_id"])
            item = int(row["item_id"])
            split = row["split"].strip().lower()
            timestamp = int(row["timestamp"])
            user_rows.setdefault(user, []).append((timestamp, item, split))

    # 为了让数据顺序稳定，我们按时间戳排序。
    # 虽然当前训练逻辑不直接依赖顺序，但这样更符合推荐系统原始数据的习惯，
    # 后续如果要做时序切分或者会话建模，也更自然。
    for rows in user_rows.values():
        rows.sort(key=lambda x: x[0])
    return user_rows


def build_dataset(csv_path: str | Path, device: str = "cpu") -> DatasetBundle:
    """构建训练所需的完整数据对象。

    这一步会完成几件事：
    1. 读取 CSV
    2. 把原始 user/item ID 映射成从 0 开始的连续索引
    3. 分离 train / val / test 样本
    4. 构建只基于训练边的归一化图邻接矩阵

    注意：图只用训练集构建，这是推荐系统评估里常见的设定，
    可以避免把验证集和测试集信息泄露给模型。
    """

    user_rows = load_interactions(csv_path)
    item_ids = sorted({item for rows in user_rows.values() for _, item, _ in rows})

    # 把稀疏、不连续的原始 ID 映射成连续索引。
    # Embedding 层要求索引落在 `[0, n)` 范围内，因此这是标准预处理步骤。
    user_id_map = {raw_user: idx for idx, raw_user in enumerate(sorted(user_rows))}
    item_id_map = {raw_item: idx for idx, raw_item in enumerate(item_ids)}

    train_pairs: List[Interaction] = []
    val_pairs: List[Interaction] = []
    test_pairs: List[Interaction] = []

    # `train_user_items[user]` 只保存训练集中用户已经交互过的物品。
    # 这会在两个地方用到：
    # 1. 负采样时，避免把正样本错当成负样本
    # 2. 评估时，屏蔽训练集中已经看过的物品
    train_user_items: Dict[int, set] = {uid: set() for uid in range(len(user_id_map))}

    # `all_user_items` 保留用户所有 split 的交互，后续如果你要做数据分析、
    # 统计覆盖率、或者自定义评估过滤规则，这个字段会有帮助。
    all_user_items: Dict[int, set] = {uid: set() for uid in range(len(user_id_map))}

    for raw_user, rows in user_rows.items():
        user = user_id_map[raw_user]
        for _, raw_item, split in rows:
            item = item_id_map[raw_item]
            pair = (user, item)
            all_user_items[user].add(item)
            if split == "train":
                train_pairs.append(pair)
                train_user_items[user].add(item)
            elif split == "val":
                val_pairs.append(pair)
            elif split == "test":
                test_pairs.append(pair)
            else:
                raise ValueError(f"Unsupported split: {split}")

    # LightGCN 的图结构只使用训练交互边。
    # 这样模型的消息传播过程不会提前“看到”验证和测试集的目标物品。
    norm_adj = build_normalized_adj(
        num_users=len(user_id_map),
        num_items=len(item_id_map),
        interactions=train_pairs,
        device=device,
    )
    return DatasetBundle(
        num_users=len(user_id_map),
        num_items=len(item_id_map),
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_user_items=train_user_items,
        all_user_items=all_user_items,
        norm_adj=norm_adj,
    )


def build_normalized_adj(
    num_users: int,
    num_items: int,
    interactions: Sequence[Interaction],
    device: str = "cpu",
) -> torch.Tensor:
    """构建 LightGCN 使用的对称归一化邻接矩阵。

    图结构是标准的 user-item 二部图：
    - 用户节点编号范围：`[0, num_users)`
    - 物品节点编号范围：`[num_users, num_users + num_items)`

    对于每条交互 `(u, i)`，我们会添加两条边：
    - `u -> i`
    - `i -> u`

    随后使用 LightGCN 论文里的常见归一化形式：
    `D^{-1/2} A D^{-1/2}`
    """

    total_nodes = num_users + num_items
    row_indices: List[int] = []
    col_indices: List[int] = []

    for user, item in interactions:
        # 因为用户节点和物品节点共用同一张图，所以物品索引需要整体平移。
        item_node = num_users + item
        row_indices.extend((user, item_node))
        col_indices.extend((item_node, user))

    indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
    values = torch.ones(len(row_indices), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (total_nodes, total_nodes)).coalesce()

    # 稀疏图的每个节点度数，用来做对称归一化。
    degrees = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = degrees.pow(-0.5)

    # 对于孤立点，度数可能是 0，此时会出现 inf。
    # 直接把它们置 0，避免数值问题。
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    row, col = adj.indices()
    norm_values = deg_inv_sqrt[row] * adj.values() * deg_inv_sqrt[col]
    norm_adj = torch.sparse_coo_tensor(
        adj.indices(),
        norm_values,
        adj.shape,
        dtype=torch.float32,
        device=device,
    )
    return norm_adj.coalesce()


def sample_batch(
    train_pairs: Sequence[Interaction],
    train_user_items: Dict[int, set],
    num_items: int,
    batch_size: int,
    rng: random.Random,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从训练样本里采一个 BPR batch。

    BPR 训练需要三元组 `(user, positive_item, negative_item)`：
    - 正样本来自真实交互
    - 负样本从该用户未交互过的物品里随机采样
    """

    users: List[int] = []
    pos_items: List[int] = []
    neg_items: List[int] = []

    for _ in range(batch_size):
        user, pos_item = train_pairs[rng.randrange(len(train_pairs))]
        neg_item = rng.randrange(num_items)

        # 负样本必须是“用户训练集中没见过的物品”。
        # 否则会把真阳性当成负例，直接破坏训练目标。
        while neg_item in train_user_items[user]:
            neg_item = rng.randrange(num_items)
        users.append(user)
        pos_items.append(pos_item)
        neg_items.append(neg_item)

    # 最后统一转成张量，方便直接送进模型和 loss。
    return (
        torch.tensor(users, dtype=torch.long, device=device),
        torch.tensor(pos_items, dtype=torch.long, device=device),
        torch.tensor(neg_items, dtype=torch.long, device=device),
    )
