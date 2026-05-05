"""基线模型统一训练脚本。

支持的模型：LightGCN（后续扩展 MF, NCF 等）

用法：
    # 使用 YAML 配置文件训练
    python experiments/train_baseline.py --config configs/lightgcn_ml100k.yaml

    # 覆盖部分配置参数
    python experiments/train_baseline.py --config configs/lightgcn_ml100k.yaml \
        --override training.epochs=50 model.embedding_dim=128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataloader import BPRBatchLoader, build_normalized_adj
from data.preprocess import ProcessedData, load_movielens, leave_one_out_split, save_processed_data
from models.baselines.lightgcn import LightGCN
from utils.common import EarlyStopping, get_device, load_config, set_seed, setup_logger
from utils.metrics import evaluate_model


# ── 模型工厂 ─────────────────────────────────────────────


def build_model(
    model_name: str,
    num_users: int,
    num_items: int,
    config: Dict[str, Any],
    norm_adj: torch.Tensor,
) -> LightGCN:
    """根据模型名称构建模型实例。

    当前仅支持 LightGCN，后续扩展 MF、NCF 等时在此注册。
    """
    if model_name == "LightGCN":
        return LightGCN(num_users, num_items, config, norm_adj)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: LightGCN"
        )


# ── 训练循环 ────────────────────────────────────────────


def train_epoch(
    model: LightGCN,
    loader: BPRBatchLoader,
    optimizer: torch.optim.Optimizer,
    reg_weight: float,
    steps_per_epoch: int,
) -> float:
    """执行一个 epoch 的训练，返回平均损失。"""
    model.train()
    total_loss = 0.0

    for _ in range(steps_per_epoch):
        batch = next(iter(loader))
        optimizer.zero_grad()
        loss = model.compute_loss(batch, reg_weight=reg_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / steps_per_epoch


@torch.no_grad()
def evaluate(
    model: LightGCN,
    target_pairs,
    train_user_items,
    ks,
) -> Dict[str, float]:
    """评估模型并返回全部指标。"""
    model.eval()
    score_matrix = model.predict_all()
    return evaluate_model(score_matrix, target_pairs, train_user_items, ks)


# ── 主函数 ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a baseline recommendation model."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--override", type=str, nargs="*",
        help="Override config values in dot notation, e.g. training.epochs=50."
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    if args.override:
        for override in args.override:
            key, value = override.split("=")
            keys = key.split(".")
            target = config
            for k in keys[:-1]:
                target = target[k]
            # 自动类型转换
            v = value
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            target[keys[-1]] = v

    # 2. 设置随机种子和设备
    set_seed(config["seed"])
    device = get_device(config["device"])

    # 3. 日志系统
    logger = setup_logger(config["experiment_name"], config["log_dir"])
    logger.info(f"Experiment: {config['experiment_name']}")
    logger.info(f"Config:\n{json.dumps(config, indent=2, ensure_ascii=False)}")

    # 4. 数据准备
    processed_dir = Path(config["data"]["processed_dir"])
    if not processed_dir.exists() or not (processed_dir / "train.csv").exists():
        logger.info("Preprocessing data...")
        interactions = load_movielens(config["data"]["data_dir"])
        data: ProcessedData = leave_one_out_split(
            interactions,
            min_rating=config["data"]["min_rating"],
            min_interactions=config["data"]["min_interactions"],
        )
        save_processed_data(data, processed_dir)
    else:
        # 从已处理的 CSV 加载（简化版：假设已存在）
        logger.info(f"Loading preprocessed data from {processed_dir}")
        # 在实际使用中，这里应该实现 load_processed_data() 从 CSV 读取
        # 暂时重新处理以确保数据一致
        interactions = load_movielens(config["data"]["data_dir"])
        data = leave_one_out_split(
            interactions,
            min_rating=config["data"]["min_rating"],
            min_interactions=config["data"]["min_interactions"],
        )

    logger.info(
        f"Data: {data.num_users} users, {data.num_items} items, "
        f"{len(data.train_pairs)} train, {len(data.val_pairs)} val, "
        f"{len(data.test_pairs)} test"
    )

    # 5. 构建邻接矩阵（仅使用训练集）
    logger.info("Building normalized adjacency matrix...")
    norm_adj = build_normalized_adj(
        data.num_users, data.num_items, data.train_pairs, device=device
    )

    # 6. 构建模型
    model = build_model(
        config["model"]["name"],
        data.num_users,
        data.num_items,
        {**config["model"], **config["training"], "device": device},
        norm_adj,
    ).to(device)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Parameters: {model.count_parameters():,}")

    # 7. 构建数据加载器和优化器
    train_loader = BPRBatchLoader(
        train_pairs=data.train_pairs,
        train_user_items=data.train_user_items,
        num_items=data.num_items,
        batch_size=config["training"]["batch_size"],
        seed=config["seed"],
        device=device,
    )
    steps_per_epoch = len(train_loader)
    optimizer = model.configure_optimizers()
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stop_patience"],
        mode="max",
    )

    # 8. 训练循环
    ks = config["evaluation"]["ks"]
    valid_metric = config["evaluation"]["valid_metric"]
    best_val_score = float("-inf")
    best_state = None
    history = []

    logger.info(f"Starting training ({config['training']['epochs']} epochs max)...")

    for epoch in range(1, config["training"]["epochs"] + 1):
        # 每个 epoch 开始前清除嵌入缓存（确保使用最新参数）
        model.invalidate_cache()

        # 训练
        epoch_loss = train_epoch(
            model, train_loader, optimizer,
            config["training"]["weight_decay"], steps_per_epoch,
        )

        # 验证
        val_metrics = evaluate(
            model, data.val_pairs, data.train_user_items, ks,
        )
        val_score = val_metrics[valid_metric]

        # 记录
        history.append({
            "epoch": epoch,
            "loss": epoch_loss,
            **{k: float(v) for k, v in val_metrics.items()},
        })

        # 日志
        if epoch == 1 or epoch % config["training"]["log_interval"] == 0 or epoch == config["training"]["epochs"]:
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in val_metrics.items()
                if k.startswith("Recall") or k.startswith("NDCG")
            )
            logger.info(f"Epoch {epoch:03d} | loss={epoch_loss:.4f} | {metric_str}")

        # 保存最佳模型
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "val_metrics": val_metrics,
            }

        # 早停检查
        if early_stopping(val_score, epoch):
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(best: epoch {early_stopping.best_epoch}, "
                f"{valid_metric}={early_stopping.best_score:.4f})"
            )
            break

    # 9. 测试集评估
    if best_state is None:
        logger.error("No best model found. Training failed.")
        return

    model.load_state_dict(best_state["model"])
    model.invalidate_cache()
    model.eval()

    test_metrics = evaluate(
        model, data.test_pairs, data.train_user_items, ks,
    )

    logger.info(f"Best epoch: {best_state['epoch']}")
    logger.info("Test metrics:")
    for name, value in test_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # 10. 保存结果
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # 保存最佳模型
    torch.save(best_state, output_dir / "best_model.pt")
    # 保存指标
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": config,
            "best_epoch": best_state["epoch"],
            "best_val_metrics": {
                k: float(v) for k, v in best_state["val_metrics"].items()
            },
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "history": history,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
