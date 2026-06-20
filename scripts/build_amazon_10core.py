"""完整 Amazon 10-core 数据处理 pipeline。
1. 解析 metadata → ASIN → text
2. 解析 reviews → 10-core filter
3. 生成 LLM embeddings
4. 创建 train/val/test + 保存
"""
import ast, gzip, pickle, numpy as np, torch, os, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer, AutoModel

DATA = Path('I:/claude_code文件/run_recmodels/data/amazon_10core')

# ==================== STEP 1: metadata ====================
print('=== Step 1: Metadata ===')
asin_text = {}
count = 0
with gzip.open(DATA / 'meta_Books.json.gz', 'rt', encoding='utf-8', errors='replace') as f:
    for line in f:
        try:
            item = ast.literal_eval(line.strip())
            asin = item.get('asin', '')
            title = item.get('title', '')
            desc = item.get('description', '')
            cats = item.get('categories', [])
            cat_str = ' > '.join(cats[0]) if cats and cats[0] else ''
            text = title or ''
            if cat_str: text += f'. Category: {cat_str}'
            if not text and desc: text = desc[:300]
            if text: asin_text[asin] = text[:300]
            count += 1
        except: pass
        if count % 2000000 == 0: print(f'  {count/1e6:.1f}M items, {len(asin_text)} with text')

print(f'Metadata done: {len(asin_text):,} items with text')

# ==================== STEP 2: Reviews + 10-core ====================
print('\n=== Step 2: Reviews + 10-core filter ===')
user_items = defaultdict(set)
item_users = defaultdict(set)
user_item_time = defaultdict(list)
rcount = 0

with gzip.open(DATA / 'reviews_Books_5.json.gz', 'rt', encoding='utf-8', errors='replace') as f:
    for line in f:
        try:
            r = ast.literal_eval(line.strip()) if '\"' not in line[:10] else __import__('json').loads(line.strip())
            if isinstance(r, dict):
                u, a, t = r.get('reviewerID',''), r.get('asin',''), r.get('unixReviewTime',0)
                if u and a:
                    user_items[u].add(a)
                    item_users[a].add(u)
                    user_item_time[u].append((a, t))
                    rcount += 1
        except: pass
        if rcount % 5000000 == 0: print(f'  {rcount/1e6:.1f}M reviews')

print(f'Reviews: {rcount:,} | users: {len(user_items):,} items: {len(item_users):,}')

# 10-core filter
for _ in range(5):
    vu = {u for u,i in user_items.items() if len(i)>=10}
    vi = {i for i,u in item_users.items() if len(u)>=10}
    user_items = {u:{i for i in items if i in vi} for u,items in user_items.items() if u in vu}
    item_users = {i:{u for u in users if u in vu} for i,users in item_users.items() if i in vi}

print(f'10-core: {len(user_items):,} users, {len(item_users):,} items')

# ID mapping + time-ordered interactions
uid_map = {u:i for i,u in enumerate(sorted(user_items.keys()))}
iid_map = {i:j for j,i in enumerate(sorted(item_users.keys()))}

all_int = []
for u, items_set in user_items.items():
    uid = uid_map[u]
    for a, t in user_item_time[u]:
        if a in iid_map:
            all_int.append((uid, iid_map[a], t))
all_int.sort(key=lambda x: x[2])

# Leave-one-out split
train_pairs, val_pairs, test_pairs = [], [], []
train_user_items = defaultdict(set)
for uid in range(len(uid_map)):
    user_ints = [(iid, ts) for uidx, iid, ts in all_int if uidx == uid]
    if len(user_ints) < 3: continue
    for iid, _ in user_ints[:-2]:
        train_pairs.append((uid, iid))
        train_user_items[uid].add(iid)
    val_pairs.append((uid, user_ints[-2][0]))
    test_pairs.append((uid, user_ints[-1][0]))

nu, ni = len(uid_map), len(iid_map)
print(f'Train: {len(train_pairs):,} Val: {len(val_pairs):,} Test: {len(test_pairs):,}')
print(f'Density: {len(train_pairs)/(nu*ni)*100:.3f}%')

# ==================== STEP 3: Generate embeddings ====================
print('\n=== Step 3: Generate LLM embeddings ===')
model_path = str(Path.home() / '.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/main')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).cuda()

# Build item texts in correct order
item_ids_list = sorted(iid_map.keys(), key=lambda x: iid_map[x])
item_texts = [asin_text.get(asin, f'Book {asin}') for asin in item_ids_list]
print(f'Item texts: {len(item_texts)}, example: {item_texts[0][:100]}')

# Encode in batches
all_emb = []
bs = 256
for start in range(0, len(item_texts), bs):
    batch = item_texts[start:start+bs]
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    mask = inputs['attention_mask'].unsqueeze(-1)
    emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
    emb = emb / emb.norm(dim=1, keepdim=True)
    all_emb.append(emb.cpu().numpy())
    if start % (bs*50) == 0: print(f'  {start}/{len(item_texts)}')

item_emb = np.concatenate(all_emb, axis=0)
print(f'Item embeddings: {item_emb.shape}')

# User embeddings
user_emb = np.zeros((nu, item_emb.shape[1]), dtype=np.float32)
for uid, items in train_user_items.items():
    if items:
        user_emb[uid] = item_emb[list(items)].mean(axis=0)

# ==================== STEP 4: Save ====================
print('\n=== Step 4: Save ===')
out_dir = DATA / 'processed_10core'
out_dir.mkdir(exist_ok=True)

with open(out_dir / 'data.pkl', 'wb') as f:
    pickle.dump({
        'num_users': nu, 'num_items': ni,
        'train_pairs': train_pairs, 'val_pairs': val_pairs, 'test_pairs': test_pairs,
        'train_user_items': dict(train_user_items),
        'user_llm_emb': user_emb, 'item_llm_emb': item_emb,
    }, f)

np.save(str(out_dir / 'item_llm_emb.npy'), item_emb)
np.save(str(out_dir / 'user_llm_emb.npy'), user_emb)
print(f'Saved to {out_dir}')
print(f'  Users: {nu:,}, Items: {ni:,}, Embed dim: {item_emb.shape[1]}')
print('ALL DONE!')
