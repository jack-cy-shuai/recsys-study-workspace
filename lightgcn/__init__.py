"""DEPRECATED — 此包已被迁移到新项目结构。

旧代码保留仅作为历史参考。

新位置与映射：
  lightgcn/data.py 中的数据加载    → data/dataloader.py + data/preprocess.py
  lightgcn/model.py 中的模型定义   → models/baselines/lightgcn.py
  lightgcn/evaluate.py 中的指标   → utils/metrics.py

新的训练入口：
  python experiments/train_baseline.py --config configs/lightgcn_ml100k.yaml
"""
