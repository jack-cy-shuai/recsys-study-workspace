"""RLMRec 数据集加载器：加载 pickle 格式的稀疏矩阵和预计算 LLM embedding。

RLMRec (WWW 2024) 提供了三个经过预处理的数据集：
- Amazon-book: 526k 用户, 91k 商品, 298万交互
- Yelp: 42k 用户, 86k 商品, 113万交互
- Steam: 61k 用户, 35k 商品, 35万交互

每个数据集包含：
- trn_mat.pkl / val_mat.pkl / tst_mat.pkl: scipy.sparse 训练/验证/测试矩阵
- usr_emb_np.pkl / itm_emb_np.pkl:   预计算的 LLM 语义向量 (Instructor/Contriever)
- usr_prf.pkl / itm_prf.pkl:         LLM 生成的文本画像

下载方式：见本文件末尾的 DOWNLOAD_INSTRUCTIONS。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

Interaction = Tuple[int, int]


@dataclass
class RLMRecData:
    """RLMRec 数据集完整内容。

    Attributes
    ----------
    num_users : int
    num_items : int
    train_pairs : 训练交互 (user, item) 列表
    val_pairs : 验证交互
    test_pairs : 测试交互
    train_user_items : 用户 → 训练集物品集合
    llm_user_emb : [num_users, llm_dim] LLM 用户语义向量
    llm_item_emb : [num_items, llm_dim] LLM 商品语义向量
    user_profiles : user原始ID → 文本画像 (可选)
    item_profiles : item原始ID → 文本画像 (可选)
    """

    num_users: int
    num_items: int
    train_pairs: List[Interaction]
    val_pairs: List[Interaction]
    test_pairs: List[Interaction]
    train_user_items: Dict[int, set]
    llm_user_emb: torch.Tensor  # [num_users, llm_dim]
    llm_item_emb: torch.Tensor  # [num_items, llm_dim]
    user_profiles: Optional[Dict[int, str]] = None
    item_profiles: Optional[Dict[int, str]] = None


def _load_pickle(path: Path):
    """安全加载 pickle 文件。"""
    with open(path, "rb") as f:
        return pickle.load(f)


def _sparse_to_pairs(sparse_mat) -> List[Interaction]:
    """将 scipy.sparse 矩阵转换为 (user, item) 对列表。

    假设矩阵已 0-indexed，形状为 [num_users, num_items]。
    """
    import scipy.sparse as sp

    if sp.issparse(sparse_mat):
        coo = sparse_mat.tocoo()
        return [(int(u), int(i)) for u, i in zip(coo.row, coo.col)]
    else:
        # 如果已是 numpy array
        rows, cols = np.where(sparse_mat > 0)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]


def load_rlmrec_data(data_dir: str | Path) -> RLMRecData:
    """加载 RLMRec 数据集。

    Parameters
    ----------
    data_dir : 包含 .pkl 文件的数据目录，如 data/rlmrec/amazon/

    Returns
    -------
    RLMRecData
    """
    data_dir = Path(data_dir)

    # 检查必要文件
    required = ["trn_mat.pkl", "val_mat.pkl", "tst_mat.pkl",
                 "usr_emb_np.pkl", "itm_emb_np.pkl"]
    for fname in required:
        if not (data_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing {fname} in {data_dir}.\n"
                f"Please download the RLMRec dataset first.\n"
                f"See DOWNLOAD_INSTRUCTIONS at the bottom of this file."
            )

    print(f"Loading RLMRec data from {data_dir} ...")

    # 1. 加载稀疏矩阵
    print("  Loading train/val/test sparse matrices ...")
    trn_mat = _load_pickle(data_dir / "trn_mat.pkl")
    val_mat = _load_pickle(data_dir / "val_mat.pkl")
    tst_mat = _load_pickle(data_dir / "tst_mat.pkl")

    num_users, num_items = trn_mat.shape
    print(f"  Matrix shape: {num_users} users × {num_items} items")

    # 2. 提取交互对
    train_pairs = _sparse_to_pairs(trn_mat)
    val_pairs = _sparse_to_pairs(val_mat)
    test_pairs = _sparse_to_pairs(tst_mat)
    print(f"  Interactions: {len(train_pairs)} train, "
          f"{len(val_pairs)} val, {len(test_pairs)} test")

    # 3. 构建 train_user_items
    train_user_items: Dict[int, set] = {}
    for u, i in train_pairs:
        train_user_items.setdefault(u, set()).add(i)

    # 4. 加载 LLM embedding
    print("  Loading LLM semantic embeddings ...")
    llm_user_emb = _load_pickle(data_dir / "usr_emb_np.pkl")
    llm_item_emb = _load_pickle(data_dir / "itm_emb_np.pkl")

    # 确保是 numpy array
    if not isinstance(llm_user_emb, np.ndarray):
        llm_user_emb = np.array(llm_user_emb)
    if not isinstance(llm_item_emb, np.ndarray):
        llm_item_emb = np.array(llm_item_emb)

    # 验证形状
    if llm_user_emb.shape[0] != num_users:
        print(f"  [WARNING] User emb shape {llm_user_emb.shape[0]} != {num_users}")
    if llm_item_emb.shape[0] != num_items:
        print(f"  [WARNING] Item emb shape {llm_item_emb.shape[0]} != {num_items}")

    print(f"  LLM embedding dim: user {llm_user_emb.shape}, item {llm_item_emb.shape}")

    # 5. （可选）加载文本画像
    user_profiles = None
    item_profiles = None
    if (data_dir / "usr_prf.pkl").exists():
        print("  Loading text profiles ...")
        user_profiles = _load_pickle(data_dir / "usr_prf.pkl")
        item_profiles = _load_pickle(data_dir / "itm_prf.pkl")

    print(f"Done. {num_users} users, {num_items} items, "
          f"LLM dim={llm_user_emb.shape[1]}")

    return RLMRecData(
        num_users=num_users,
        num_items=num_items,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_user_items=train_user_items,
        llm_user_emb=torch.from_numpy(llm_user_emb).float(),
        llm_item_emb=torch.from_numpy(llm_item_emb).float(),
        user_profiles=user_profiles,
        item_profiles=item_profiles,
    )
