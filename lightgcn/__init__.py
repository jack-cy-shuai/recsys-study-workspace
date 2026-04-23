"""Minimal LightGCN reproduction package.

这个包里放的是一个尽量精简、但训练流程完整的 LightGCN 实现。
拆成多个文件的目的，是为了便于分别调试：

- `data.py`：数据读取、ID 重映射、邻接矩阵构建、负采样
- `model.py`：LightGCN 模型本体和 BPR loss
- `evaluate.py`：Recall / NDCG 指标计算
"""
