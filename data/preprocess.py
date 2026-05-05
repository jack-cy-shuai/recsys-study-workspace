"""数据预处理：从原始数据集生成标准化的训练/验证/测试集。

支持的数据集：
- MovieLens-100K (ml-100k)
- MovieLens-1M (ml-1m)

数据划分策略：
- 留一法 (leave-one-out)：每个用户按时间排序，最后 1 条 → test，倒数第 2 条 → val，其余 → train。
  这是论文中最常用的隐式反馈评估设定。

输出格式：
    三个 CSV 文件：train.csv, val.csv, test.csv
    每行格式：user_id,item_id,timestamp
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ProcessedData:
    """预处理后的完整数据集合。

    Attributes
    ----------
    num_users : int
        去重后的用户数（连续 ID，0 到 num_users-1）。
    num_items : int
        去重后的物品数（连续 ID，0 到 num_items-1）。
    train_pairs : list of (user, item)
        训练集交互对。每个用户至少有 1 条训练交互。
    val_pairs : list of (user, item)
        验证集交互对。每个用户恰好 1 条（留一法）。
    test_pairs : list of (user, item)
        测试集交互对。每个用户恰好 1 条（留一法）。
    train_user_items : dict, user → set of items
        训练集中每个用户交互过的物品集合。
    """

    num_users: int
    num_items: int
    train_pairs: List[Tuple[int, int]]
    val_pairs: List[Tuple[int, int]]
    test_pairs: List[Tuple[int, int]]
    train_user_items: Dict[int, set]


# ── MovieLens 数据读取 ───────────────────────────────────


def load_movielens(data_dir: str | Path) -> List[Tuple[int, int, int, float]]:
    """读取 MovieLens u.data 文件。

    Movielens-100K/1M 使用统一的 u.data 格式：
    user_id \t item_id \t rating \t timestamp

    Returns
    -------
    interactions : list of (user_id, item_id, rating, timestamp)
        按时间戳排序后的全部交互记录。
    """
    data_dir = Path(data_dir)

    # 尝试常见的文件名和位置
    candidates = [
        data_dir / "u.data",
        data_dir / "ml-100k" / "u.data",
        data_dir / "ml-1m" / "ratings.dat",
    ]
    data_path = None
    for c in candidates:
        if c.exists():
            data_path = c
            break

    if data_path is None:
        raise FileNotFoundError(
            f"Could not find MovieLens data file. "
            f"Checked: {[str(c) for c in candidates]}\n"
            f"Please download from https://grouplens.org/datasets/movielens/ "
            f"and place it in {data_dir.resolve()}/"
        )

    interactions = []
    if data_path.suffix == ".dat":
        # ml-1m 格式：user_id::item_id::rating::timestamp
        sep = "::"
    else:
        sep = "\t"

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(sep)
            user_id = int(parts[0])
            item_id = int(parts[1])
            rating = float(parts[2])
            timestamp = int(parts[3])
            interactions.append((user_id, item_id, rating, timestamp))

    # 按时间戳排序，确保留一法划分的确定性
    interactions.sort(key=lambda x: x[3])
    return interactions


# ── 留一法数据划分 ───────────────────────────────────────


def leave_one_out_split(
    interactions: List[Tuple[int, int, int, float]],
    min_rating: float = 0.0,
    min_interactions: int = 5,
) -> ProcessedData:
    """对原始交互记录执行留一法划分。

    处理流程：
    1. 过滤低评分（可选，min_rating=0 表示保留全部）
    2. 过滤交互数过少的用户（min_interactions）
    3. 按用户分组，时间排序
    4. 每个用户最后 1 条 → test，倒数第 2 条 → val，其余 → train
    5. 将原始 ID 映射为连续索引

    Parameters
    ----------
    interactions : list of (user_id, item_id, rating, timestamp)
        按时间排序后的交互记录。
    min_rating : float
        最低评分阈值。0 表示保留所有交互（隐式反馈），4 表示仅保留评分 ≥ 4 的。
        论文中 BPR-MF 使用全部交互，LightGCN 也使用全部。
    min_interactions : int
        用户最少交互数。交互不足的用户会被过滤。
        留一法至少需要每用户 3 条交互（train + val + test 各至少 1 条）。

    Returns
    -------
    ProcessedData
    """
    # 步骤 1：按用户分组，过滤评分和时间排序
    user_data: Dict[int, List[Tuple[int, int]]] = {}  # user_id -> [(item_id, timestamp), ...]

    for user_id, item_id, rating, timestamp in interactions:
        if rating < min_rating:
            continue
        user_data.setdefault(user_id, []).append((item_id, timestamp))

    # 步骤 2：过滤交互数不足的用户 & 按时间排序
    filtered_data = {}
    for uid, items in user_data.items():
        items.sort(key=lambda x: x[1])  # 按时间排序
        if len(items) >= min_interactions:
            filtered_data[uid] = items

    if len(filtered_data) == 0:
        raise ValueError(
            f"No users with >= {min_interactions} interactions after filtering "
            f"(min_rating={min_rating}). Try lowering the thresholds."
        )

    # 步骤 3：留一法划分
    # 每个用户：... train倒数多项 ..., val倒数第二, test倒数第一
    raw_train: Dict[int, List[int]] = {}
    raw_val: List[Tuple[int, int]] = []
    raw_test: List[Tuple[int, int]] = []

    for uid, items in filtered_data.items():
        # test = 最后一条
        test_item, _ = items[-1]
        raw_test.append((uid, test_item))

        # val = 倒数第二条（如果该用户至少有 2 条）
        if len(items) >= 2:
            val_item, _ = items[-2]
            raw_val.append((uid, val_item))
            train_items = items[:-2]
        else:
            train_items = items[:-1]

        raw_train[uid] = [item for item, _ in train_items]

    # 步骤 4：ID 映射（原始 → 连续 0-based）
    all_users = sorted(filtered_data.keys())
    user_id_map = {raw_uid: idx for idx, raw_uid in enumerate(all_users)}

    all_items_set = set()
    for items in raw_train.values():
        all_items_set.update(items)
    for uid, item in raw_val:
        all_items_set.add(item)
    for uid, item in raw_test:
        all_items_set.add(item)
    all_items = sorted(all_items_set)
    item_id_map = {raw_iid: idx for idx, raw_iid in enumerate(all_items)}

    # 步骤 5：映射并构造输出
    train_pairs = [
        (user_id_map[uid], item_id_map[iid])
        for uid, items in raw_train.items()
        for iid in items
    ]
    val_pairs = [
        (user_id_map[uid], item_id_map[iid]) for uid, iid in raw_val
    ]
    test_pairs = [
        (user_id_map[uid], item_id_map[iid]) for uid, iid in raw_test
    ]

    train_user_items: Dict[int, set] = {uid: set() for uid in range(len(all_users))}
    for uid, items in raw_train.items():
        mapped_uid = user_id_map[uid]
        for iid in items:
            mapped_iid = item_id_map[iid]
            train_user_items[mapped_uid].add(mapped_iid)

    return ProcessedData(
        num_users=len(user_id_map),
        num_items=len(item_id_map),
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_user_items=train_user_items,
    )


# ── 数据持久化 ───────────────────────────────────────────


def save_processed_data(
    data: ProcessedData, output_dir: str | Path
) -> None:
    """将预处理后的数据保存为 CSV 文件。

    生成 train.csv, val.csv, test.csv 三个文件，
    格式：user_id,item_id
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, pairs in [
        ("train", data.train_pairs),
        ("val", data.val_pairs),
        ("test", data.test_pairs),
    ]:
        path = output_dir / f"{split}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "item_id"])
            writer.writerows(pairs)

    # 同时保存元信息
    meta_path = output_dir / "meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"num_users={data.num_users}\n")
        f.write(f"num_items={data.num_items}\n")
        f.write(f"num_train={len(data.train_pairs)}\n")
        f.write(f"num_val={len(data.val_pairs)}\n")
        f.write(f"num_test={len(data.test_pairs)}\n")
        if data.train_pairs:
            f.write(f"density={len(data.train_pairs) / (data.num_users * data.num_items):.6f}\n")

    print(f"Saved processed data to {output_dir.resolve()}")
    print(
        f"  Users: {data.num_users}, Items: {data.num_items}\n"
        f"  Train: {len(data.train_pairs)}, "
        f"Val: {len(data.val_pairs)}, "
        f"Test: {len(data.test_pairs)}"
    )


# ── 命令行入口 ───────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess MovieLens dataset with leave-one-out split."
    )
    parser.add_argument(
        "--dataset", type=str, default="ml-100k",
        help="Dataset name: ml-100k or ml-1m."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/ml-100k",
        help="Directory containing the raw dataset files."
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed_ml100k",
        help="Output directory for processed files."
    )
    parser.add_argument(
        "--min-rating", type=float, default=0.0,
        help="Minimum rating to consider as positive interaction (0=keep all)."
    )
    parser.add_argument(
        "--min-interactions", type=int, default=5,
        help="Minimum interactions per user."
    )
    args = parser.parse_args()

    print(f"Loading {args.dataset} from {args.data_dir} ...")
    interactions = load_movielens(args.data_dir)
    print(f"Loaded {len(interactions)} raw interactions.")

    data = leave_one_out_split(
        interactions,
        min_rating=args.min_rating,
        min_interactions=args.min_interactions,
    )
    save_processed_data(data, args.output_dir)


if __name__ == "__main__":
    main()
