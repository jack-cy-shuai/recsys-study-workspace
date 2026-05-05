"""基线模型模块。

所有基线模型的实现均遵循原论文的数学推导，不依赖任何第三方推荐系统库。

已实现：
- LightGCN：基于图卷积的轻量协同过滤模型

待实现：
- MF (Matrix Factorization)：矩阵分解
- BPR-MF：BPR 优化的矩阵分解
- NCF/NeuMF：神经协同过滤
- NGCF：神经图协同过滤（LightGCN 的前身）
"""
