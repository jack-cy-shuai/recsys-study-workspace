"""传统推荐系统损失函数：BPR pairwise loss、BCE pointwise loss、MSE regression loss。

所有损失函数返回标量 tensor，可直接用于 backward()。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """BPR (Bayesian Personalized Ranking) pairwise loss。

    目标：让每个正样本的得分高于对应负样本的得分。
    L = -log(sigmoid(pos_score - neg_score))

    这是协同过滤中最常用的 pairwise 排序损失，
    出自论文 "BPR: Bayesian Personalized Ranking from Implicit Feedback" (Rendle et al., 2009)。

    Parameters
    ----------
    pos_scores : Tensor, shape [batch_size]
        正样本的预测分数。
    neg_scores : Tensor, shape [batch_size]
        负样本的预测分数（每个正样本对应一个负样本）。
    reduction : str
        "mean" 返回均值，"sum" 返回总和，"none" 返回逐元素值。

    Returns
    -------
    loss : Tensor (scalar if reduction != "none")
    """
    diff = pos_scores - neg_scores
    # + 1e-8 防止 log(0) 的数值问题
    loss = -torch.log(torch.sigmoid(diff) + 1e-8)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def bce_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """二分类交叉熵损失 (Binary Cross Entropy)。

    适用于 pointwise 训练范式（如 NCF 中的二分类视角）：
    将交互视为正样本 (label=1)，未交互视为负样本 (label=0)。

    Parameters
    ----------
    scores : Tensor, shape [batch_size]
        模型输出的 logits（未经过 sigmoid）。
    labels : Tensor, shape [batch_size]
        真实标签，0 或 1。
    reduction : str
        "mean" / "sum" / "none"。

    Returns
    -------
    loss : Tensor
    """
    return F.binary_cross_entropy_with_logits(scores, labels.float(), reduction=reduction)


def mse_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """均方误差损失 (Mean Squared Error)。

    适用于显式反馈 / 评分预测任务（如传统 Matrix Factorization）。

    Parameters
    ----------
    scores : Tensor, shape [batch_size]
        模型预测评分。
    labels : Tensor, shape [batch_size]
        真实评分。
    reduction : str
        "mean" / "sum" / "none"。
    """
    return F.mse_loss(scores, labels.float(), reduction=reduction)


# ── 正则化工具 ─────────────────────────────────────────────


def l2_reg_loss(
    embeddings: List[torch.Tensor],
    reduction: str = "mean",
) -> torch.Tensor:
    """对一组 embedding 向量计算 L2 正则化损失。

    常用于 BPR-MF、LightGCN 等模型中，防止 embedding 范数无限增长。

    Parameters
    ----------
    embeddings : list of Tensor
        需要正则化的 embedding 张量列表，每个形状为 [batch_size, dim]。
    reduction : str
        "mean"：先对每个向量内部求和再对 batch 平均（推荐系统常用方式）
        "sum"：所有元素平方和

    Returns
    -------
    reg : scalar Tensor
    """
    if reduction == "mean":
        # 每个样本的 L2 范数平方（均值），再对 batch 平均
        total = 0.0
        for emb in embeddings:
            total += emb.pow(2).sum(dim=-1).mean()
        return total
    else:
        total = 0.0
        for emb in embeddings:
            total += emb.pow(2).sum()
        return total
