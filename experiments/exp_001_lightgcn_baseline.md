# Experiment 001: LightGCN Baseline

## Goal

Run a minimal LightGCN baseline end to end on a small implicit-feedback dataset.

## Code Version

Current workspace baseline implementation.

## Dataset

- file: `data/basic_implicit/interactions.csv`
- users: 60
- items: 120
- interactions: 600

## Command

```powershell
conda run -n recsys python train.py
```

## Main Settings

- embedding dim: 32
- layers: 3
- epochs: 80
- batch size: 256
- learning rate: 1e-3
- regularization: 1e-4
- eval k: 10, 20
- device: cpu
- seed: 2024

## Best Validation Result

- best epoch: 61
- Recall@10: 0.5333
- Recall@20: 0.8000
- NDCG@10: 0.2490
- NDCG@20: 0.3177

## Test Result

- Recall@10: 0.5833
- Recall@20: 0.7667
- NDCG@10: 0.2792
- NDCG@20: 0.3274

## Output Files

- `artifacts/basic_run/best_model.pt`
- `artifacts/basic_run/metrics.json`

## Conclusion

The minimal LightGCN pipeline runs successfully and produces stable ranking metrics on the basic dataset.

## Next Ideas

- try larger embedding sizes
- compare different numbers of graph layers
- replace the basic dataset with MovieLens
- compare against MF or NCF baselines

