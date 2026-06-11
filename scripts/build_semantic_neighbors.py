"""预处理：用 LLM 语义相似度为冷门物品找热门邻居。

用于后续的热→冷知识蒸馏实验。
"""

import pickle
import numpy as np
from pathlib import Path

DATASETS = ['amazon', 'steam', 'yelp']
COLD_THRESHOLD = 5  # 交互数 < 5 = 冷门
K = 10  # 每个冷门物品找 10 个热门邻居


def build_neighbors(data_dir: Path):
    with open(data_dir / 'itm_emb_np.pkl', 'rb') as f:
        item_llm = pickle.load(f)
    with open(data_dir / 'trn_mat.pkl', 'rb') as f:
        trn_mat = pickle.load(f)

    item_counts = np.array(trn_mat.sum(axis=0)).flatten()
    cold_mask = item_counts < COLD_THRESHOLD
    hot_mask = ~cold_mask

    num_cold = cold_mask.sum()
    num_hot = hot_mask.sum()
    print(f'  Cold (<{COLD_THRESHOLD}): {num_cold}, Hot: {num_hot}')

    if num_cold == 0:
        print('  No cold items, skipping.')
        return

    item_llm = item_llm.astype(np.float32)
    norms = np.linalg.norm(item_llm, axis=1, keepdims=True) + 1e-10
    item_llm_norm = item_llm / norms

    cold_llm = item_llm_norm[cold_mask]
    hot_llm = item_llm_norm[hot_mask]
    cold_ids = np.where(cold_mask)[0]
    hot_ids = np.where(hot_mask)[0]

    sim = cold_llm @ hot_llm.T
    top_k = np.argsort(-sim, axis=1)[:, :K]

    neighbors = {}
    for i, cid in enumerate(cold_ids):
        neighbors[int(cid)] = {
            'neighbors': [int(hot_ids[j]) for j in top_k[i]],
            'sims': [float(sim[i, j]) for j in top_k[i]],
            'count': int(item_counts[cid]),
        }

    top1_sims = [v['sims'][0] for v in neighbors.values()]
    print(f'  Avg top-1 similarity: {np.mean(top1_sims):.4f}')
    print(f'  Median top-1 similarity: {np.median(top1_sims):.4f}')

    out = data_dir / 'semantic_neighbors.pkl'
    with open(out, 'wb') as f:
        pickle.dump({
            'neighbors': neighbors,
            'cold_threshold': COLD_THRESHOLD,
            'k': K,
        }, f)
    print(f'  Saved to {out}')


if __name__ == '__main__':
    base = Path('I:/claude_code文件/run_recmodels/data/rlmrec')
    for ds in DATASETS:
        print(f'\n=== {ds} ===')
        build_neighbors(base / ds)
    print('\nDone.')
