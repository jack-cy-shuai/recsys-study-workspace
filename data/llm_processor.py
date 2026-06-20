"""LLM 语义 Embedding 生成模块。

从物品文本元数据（标题、类别等）生成 LLM 语义 embedding。
当前支持 sentence-transformers 模型。
"""

from pathlib import Path
import numpy as np


def generate_item_embeddings(
    item_texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """用 sentence-transformers 编码物品文本。

    Args:
        item_texts: 每个物品的文本描述列表
        model_name: sentence-transformers 模型名
        batch_size: 编码 batch size
        device: "cuda" 或 "cpu"

    Returns:
        item_emb: [num_items, embedding_dim] numpy float32 数组
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        item_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def read_ml1m_items(data_dir):
    """读取 ML-1M movies.dat，返回每部电影的文本表示。

    格式：MovieID::Title (Year)::Genres
    文本：Title. Genres: Genres
    """
    data_dir = Path(data_dir)
    movies_path = data_dir / "movies.dat"
    items_dict = {}
    with open(movies_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) < 3:
                continue
            mid, title, genres = int(parts[0]), parts[1], parts[2]
            text = f"{title}. Genres: {genres}"
            items_dict[mid] = text
    return [items_dict[i] for i in sorted(items_dict.keys())]


def compute_user_emb_from_items(
    item_emb: np.ndarray,
    user_items: list[list[int]],
) -> np.ndarray:
    """用户 embedding = 交互物品 embedding 的均值。

    Args:
        item_emb: [num_items, dim] 物品 embedding
        user_items: 每个用户的交互物品索引列表

    Returns:
        user_emb: [num_users, dim]
    """
    user_emb = np.zeros((len(user_items), item_emb.shape[1]), dtype=np.float32)
    for u, items in enumerate(user_items):
        if items:
            user_emb[u] = item_emb[items].mean(axis=0)
    return user_emb
