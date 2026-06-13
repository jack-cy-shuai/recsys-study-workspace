"""记录真实梯度分布：标准 BPR vs HardBPR。训练若干 batch 后画直方图。"""
import pickle, sys, json, torch, numpy as np
from pathlib import Path

PROJECT = Path('I:/claude_code文件/run_recmodels')
sys.path.insert(0, str(PROJECT))

from data.rlmrec_loader import load_rlmrec_data
from data.dataloader import build_normalized_adj, BPRBatchLoader
from models.baselines.lightgcn import LightGCN
from models.llm_enhanced.lightgcn_hardbpr import LightGCN_HardBPR

rlmrec = load_rlmrec_data(PROJECT / 'data/rlmrec/amazon')
device = torch.device('cuda')
norm_adj = build_normalized_adj(rlmrec.num_users, rlmrec.num_items, rlmrec.train_pairs, device=device)

config = {'embedding_dim': 64, 'num_layers': 3, 'lr': 0.001, 'weight_decay': 0.0001, 'reg_weight': 1e-4, 'hard_bpr_beta': 0.5}

results = {}

for label, model_cls, kwargs in [
    ('Standard BPR', LightGCN, {}),
    ('HardBPR', LightGCN_HardBPR, {'llm_item_emb': rlmrec.llm_item_emb}),
]:
    model = model_cls(rlmrec.num_users, rlmrec.num_items, config, norm_adj, **kwargs).to(device)
    loader = BPRBatchLoader(rlmrec.train_pairs, rlmrec.train_user_items, rlmrec.num_items,
                            batch_size=2048, seed=2024, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    all_grads = []
    model.train()
    for step in range(100):  # 100 batches
        batch = next(iter(loader))
        optimizer.zero_grad()

        if label == 'Standard BPR':
            loss = model.compute_loss(batch, reg_weight=1e-4)
        else:
            # HardBPR's compute_loss handles its own BPR internally
            loss = model.compute_loss(batch, reg_weight=1e-4)

        loss.backward()

        # 记录所有参数的梯度幅值
        batch_grads = []
        for p in model.parameters():
            if p.grad is not None:
                batch_grads.append(p.grad.abs().cpu().numpy().flatten())
        if batch_grads:
            all_grads.append(np.concatenate(batch_grads))

        optimizer.step()
        model.invalidate_cache()

    all_grads = np.concatenate(all_grads)
    # 采样到 10000 个点用于画图
    if len(all_grads) > 10000:
        rng = np.random.RandomState(42)
        all_grads = rng.choice(all_grads, 10000, replace=False)

    results[label] = all_grads
    del model
    torch.cuda.empty_cache()

# 画图
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, (label, grads) in zip(axes, results.items()):
    # log scale, clip extreme values
    g = np.clip(grads, 0, np.percentile(grads, 99))
    ax.hist(g, bins=50, color='#4472C4' if 'Standard' in label else '#ED7D31',
            alpha=0.8, edgecolor='white')
    ax.axvline(np.mean(g), color='red', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(g):.5f}')
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_xlabel('Gradient Magnitude', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(fontsize=10)

fig.suptitle('Real Gradient Distributions (100 batches, Amazon-Book)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(str(PROJECT / 'paper' / 'fig_gradient.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(str(PROJECT / 'paper' / 'fig_gradient.png'), dpi=150, bbox_inches='tight')
print('Real gradient figure saved.')
