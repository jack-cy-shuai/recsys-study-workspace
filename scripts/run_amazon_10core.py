"""Amazon 10-core 实验脚本（通宵运行）。
使用方法：
    conda activate recsys
    set PYTORCH_CUDA_ALLOC_CONF=
    python scripts/run_amazon_10core.py

预计时间：2-4 小时（160K 用户 × 131K 物品）
"""
import pickle, torch, numpy as np, sys, json, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from data.dataloader import build_normalized_adj, BPRBatchLoader
from models.baselines.lightgcn import LightGCN
from models.llm_enhanced.lightgcn_hardbpr import LightGCN_HardBPR
from utils.metrics import evaluate_model_batched
from utils.common import set_seed

CACHE = PROJECT / 'data/amazon_10core/processed_10core'
d = pickle.load(open(CACHE/'train_val_test.pkl','rb'))
nu, ni = d['num_users'], d['num_items']
print(f'Data: {nu:,} users × {ni:,} items × {len(d["train_pairs"]):,} train', flush=True)

set_seed(2024)

# Build adj on CPU
adj = build_normalized_adj(nu, ni, d['train_pairs'], device='cpu').cuda()
print(f'Adj: {adj._nnz()/1e6:.1f}M edges')

item_emb = torch.from_numpy(np.load(str(CACHE/'item_llm_emb.npy')))
config = {'embedding_dim':64, 'num_layers':3, 'lr':0.001, 'weight_decay':0.0001, 'reg_weight':1e-4, 'hard_bpr_beta':0.5}
bs = 512

def b_eval(model, pairs, ks):
    users = sorted({u for u,_ in pairs})
    def sf(uid): return model.get_user_embeddings()[uid] @ model.get_item_embeddings().T
    return evaluate_model_batched(sf, users, pairs, d['train_user_items'], ks, batch_size=bs, device='cuda')

results = {}
for label, cls, kwargs in [
    ('LightGCN', LightGCN, {}),
    ('HardBPR', LightGCN_HardBPR, {'llm_item_emb': item_emb}),
]:
    print(f'\n=== {label} ===', flush=True)
    model = cls(nu, ni, config, adj, **kwargs).cuda()
    loader = BPRBatchLoader(d['train_pairs'], d['train_user_items'], ni, batch_size=bs, seed=2024, device='cuda')
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_r20, patience, best_state = 0, 0, None
    t0 = time.time()
    for epoch in range(1, 200):
        model.train(); model.invalidate_cache()
        t1 = time.time()
        for _ in range(len(loader)):
            b = next(iter(loader)); opt.zero_grad()
            L = model.compute_loss(b, reg_weight=1e-4)
            L.backward(); opt.step()
        model.eval()
        vm = b_eval(model, d['val_pairs'], [20])
        r20 = vm['Recall@20']; t2 = time.time()
        if r20 > best_r20 + 1e-5:
            best_r20 = r20; patience = 0; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else: patience += 1
        if epoch==1 or epoch%5==0:
            print(f'  Epoch {epoch}: val R@20={r20:.4f} best={best_r20:.4f} ({t2-t1:.0f}s train + {t2-t2:.0f}s eval)', flush=True)
        if patience >= 20:
            print(f'  Early stop at epoch {epoch}', flush=True)
            break
    model.load_state_dict(best_state); model.eval()
    tm = b_eval(model, d['test_pairs'], [10,20,50])
    results[label] = tm
    dt = time.time() - t0
    print(f'  TEST: R@10={tm["Recall@10"]:.4f} R@20={tm["Recall@20"]:.4f} N@20={tm["NDCG@20"]:.4f} R@50={tm["Recall@50"]:.4f} ({dt/60:.0f}min)', flush=True)
    del model; torch.cuda.empty_cache()

# Save
out = PROJECT / 'artifacts/amazon_10core_results.json'
out.parent.mkdir(exist_ok=True)
json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, open(out,'w'), indent=2)
print(f'\nSaved to {out}', flush=True)
print('ALL DONE!')
