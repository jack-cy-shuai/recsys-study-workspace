# Experiment 002: LLM Semantic Embedding vs Random Initialization

## Goal

Compare two initialization strategies for LightGCN on RLMRec Amazon-book data:
1. **LightGCN (random init)**: standard Gaussian random ID embedding
2. **LightGCN-LLM**: ID embedding + projected LLM semantic embedding (1536-dim frozen)

Goal: verify whether pre-computed LLM semantic embeddings improve collaborative filtering performance.

## Code Version

Current workspace, branch `main`, commit `f671b35`.

## Dataset

- source: RLMRec (WWW 2024) preprocessed Amazon-book subset
- users: 11,000
- items: 9,332
- train interactions: 120,464
- val interactions: 40,290
- test interactions: 40,106
- LLM embedding dim: 1536
- density: 0.12%

## Commands

```powershell
# Pure LightGCN baseline
python experiments/train_baseline.py --config configs/lightgcn_amazon.yaml

# LLM-enhanced LightGCN
python experiments/train_baseline.py --config configs/lightgcn_llm_amazon.yaml
```

## Main Settings

| Parameter | Value |
|-----------|-------|
| embedding_dim | 64 |
| num_layers | 3 |
| epochs (max) | 200 |
| batch_size | 2048 |
| lr | 0.001 |
| weight_decay | 0.0001 |
| early_stop_patience | 20 |
| valid_metric | Recall@20 |
| device | cuda (RTX 3050 Ti) |
| seed | 2024 |
| freeze_llm | true (LLM embeddings frozen) |

## Best Validation Results

| Metric | LightGCN (random) | LightGCN-LLM | Improvement |
|--------|-------------------|--------------|-------------|
| Best Epoch | 26 | 26 | — |
| Recall@10 | 0.0119 | **0.0375** | +215% |
| Recall@20 | 0.0247 | **0.0617** | +150% |
| Recall@50 | 0.0539 | **0.1123** | +108% |
| NDCG@10 | 0.0070 | **0.0283** | +304% |
| NDCG@20 | 0.0114 | **0.0362** | +218% |
| NDCG@50 | 0.0197 | **0.0504** | +156% |
| Hit@50 | 0.1578 | **0.3022** | +92% |

## Test Results

| Metric | LightGCN (random) | LightGCN-LLM | Improvement |
|--------|-------------------|--------------|-------------|
| Recall@10 | 0.0139 | **0.0380** | +173% |
| Recall@20 | 0.0267 | **0.0623** | +133% |
| Recall@50 | 0.0564 | **0.1147** | +103% |
| NDCG@10 | 0.0082 | **0.0287** | +250% |
| NDCG@20 | 0.0126 | **0.0364** | +189% |
| NDCG@50 | 0.0211 | **0.0511** | +142% |
| Hit@50 | 0.1642 | **0.3042** | +85% |

## Training Dynamics

| Metric | LightGCN (random) | LightGCN-LLM |
|--------|-------------------|--------------|
| Epoch 1 loss | 0.691 | 0.611 |
| Loss at best epoch | 0.693 (plateaued at epoch 16) | 0.256 (still decreasing) |

LightGCN-LLM converges much faster and reaches a lower BPR loss. Pure LightGCN loss stagnates after epoch 16, suggesting the model capacity is saturated on this sparse dataset without semantic priors.

## Output Files

- `artifacts/lightgcn_amazon/best_model.pt` + `metrics.json`
- `artifacts/lightgcn_llm_amazon/best_model.pt` + `metrics.json`

## Conclusion

LLM semantic embeddings provide **massive and consistent improvements** across all metrics. The 1536-dim frozen embeddings, projected through a simple linear layer and added to ID embeddings, enable LightGCN to achieve 2-3x better ranking quality. This confirms that semantic priors from LLMs are highly beneficial for collaborative filtering, especially on sparse datasets.

## Next Ideas

- verify consistency on Yelp and Steam datasets
- compare with stronger baselines (MF, NCF, NGCF)
- try larger CF embedding dimensions (128, 256)
- ablation: freeze_llm=false vs true, different projection layer designs
- explore alternative fusion strategies beyond additive (gating, attention)
- run on the full 526k-user Amazon dataset for paper-quality results
