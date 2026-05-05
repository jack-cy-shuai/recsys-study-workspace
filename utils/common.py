"""通用工具函数：随机种子、配置读取、日志系统、模型保存加载、早停机制。"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml


# ── 随机种子 ──────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """固定所有随机种子，确保实验可复现。

    覆盖 Python random、NumPy、PyTorch CPU/GPU、cuDNN。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN 确定性模式：牺牲少量性能换取可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── 配置读取 ──────────────────────────────────────────────


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """从 YAML 文件加载实验配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config file {config_path} is empty or invalid.")
    return config


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    """将配置保存为 YAML 文件（用于实验复现记录）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


# ── 日志系统 ──────────────────────────────────────────────


def setup_logger(
    name: str,
    log_dir: str | Path,
    filename: str = "experiment.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """创建同时输出到文件和控制台的 logger。

    每次调用会清除已有 handler，避免重复日志。
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件输出
    fh = logging.FileHandler(log_dir / filename, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# ── 模型保存与加载 ────────────────────────────────────────


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """保存完整训练检查点（模型权重 + 优化器状态 + 元信息）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    if extra is not None:
        checkpoint["extra"] = extra
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """加载训练检查点，恢复模型权重和优化器状态。

    返回完整的 checkpoint 字典。
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# ── 早停机制 ──────────────────────────────────────────────


class EarlyStopping:
    """监控验证指标，在连续 patience 个 epoch 无改善时触发早停。

    Parameters
    ----------
    patience : int
        容忍的 epoch 数。达到此值后 `early_stop` 变为 True。
    mode : str
        "max" 表示指标越大越好，"min" 表示越小越好。
    min_delta : float
        被认为有改善的最小变化量。
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, score: float, epoch: int) -> bool:
        """检查是否应该早停。返回 True 表示停止。"""
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:  # "min"
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def is_best(self) -> bool:
        """当前 counter 为 0 意味着最近一次 call 发现了最佳指标。"""
        return self.counter == 0


# ── 设备工具 ──────────────────────────────────────────────


def get_device(device_str: str) -> torch.device:
    """解析设备字符串，自动处理 CUDA 可用性。"""
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)
