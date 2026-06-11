"""LightGCN-LLM-Full：三层语义增强（特征 + 数据 + 损失）。

- 特征层：LLM embedding 相加融合（同 LightGCN_LLM）
- 损失层：语义感知 BPR 加权（同 LightGCN_LLM_HardBPR weight 模式）
- 数据层：语义硬负采样（在 train_baseline.py 中使用 HardBPRBatchLoader）

对比实验矩阵：
  LightGCN            : 随机初始化 + 随机负采样 + 标准 BPR
  LightGCN-LLM        : 相加融合 + 随机负采样 + 标准 BPR
  LightGCN-LLM-HardBPR: 相加融合 + 随机负采样 + 语义 BPR
  LightGCN-LLM-Full   : 相加融合 + 语义硬负 + 语义 BPR  (本文)
"""

from models.llm_enhanced.lightgcn_llm_hardbpr import LightGCN_LLM_HardBPR


class LightGCN_LLM_Full(LightGCN_LLM_HardBPR):
    """全栈语义增强 LightGCN。继承 HardBPR 的全部逻辑（特征+损失层）。"""
    pass
