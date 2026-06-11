"""基线模型统一训练脚本。

支持的模型：LightGCN, LightGCN_LLM
支持的数据集：MovieLens (ml-100k, ml-1m), RLMRec (amazon, yelp, steam)

用法：
    # LightGCN on MovieLens-100K
    python experiments/train_baseline.py --config configs/lightgcn_ml100k.yaml

    # LightGCN-LLM on RLMRec Amazon-book
    python experiments/train_baseline.py --config configs/lightgcn_llm_amazon.yaml

    # 覆盖参数
    python experiments/train_baseline.py --config configs/lightgcn_llm_amazon.yaml \
        --override training.epochs=50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# 修复 PyTorch 1.x 与 expandable_segments 不兼容的问题
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True":
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")

import torch

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataloader import BPRBatchLoader, HardBPRBatchLoader, build_normalized_adj
from data.preprocess import load_movielens, leave_one_out_split, save_processed_data
from data.rlmrec_loader import load_rlmrec_data, RLMRecData
from models.baselines.lightgcn import LightGCN
from models.llm_enhanced.lightgcn_llm import LightGCN_LLM
from models.llm_enhanced.lightgcn_llm_distill import LightGCN_LLM_Distill
from models.llm_enhanced.lightgcn_llm_hardbpr import LightGCN_LLM_HardBPR
from models.llm_enhanced.lightgcn_llm_full import LightGCN_LLM_Full
from utils.common import EarlyStopping, get_device, load_config, set_seed, setup_logger
from utils.metrics import evaluate_model, evaluate_model_batched


# ── 模型工厂 ─────────────────────────────────────────────


def build_model(
    model_name: str,
    num_users: int,
    num_items: int,
    config: Dict[str, Any],
    norm_adj: torch.Tensor,
    llm_data: RLMRecData | None = None,
):
    """根据模型名称构建模型实例。

    支持：
    - LightGCN：纯协同过滤
    - LightGCN_LLM：融合 LLM 语义 embedding（需传入 llm_data）
    """
    model_cfg = {**config.get("model", {}), **config.get("training", {}),
                 "device": config.get("device", "cpu")}

    if model_name == "LightGCN":
        return LightGCN(num_users, num_items, model_cfg, norm_adj)

    elif model_name == "LightGCN_LLM":
        if llm_data is None:
            raise ValueError(
                "LightGCN_LLM requires llm_data with pre-computed embeddings."
            )
        return LightGCN_LLM(
            num_users, num_items, model_cfg, norm_adj,
            llm_user_emb=llm_data.llm_user_emb,
            llm_item_emb=llm_data.llm_item_emb,
            freeze_llm=config.get("model", {}).get("freeze_llm", True),
        )

    elif model_name == "LightGCN_LLM_Distill":
        if llm_data is None:
            raise ValueError(
                "LightGCN_LLM_Distill requires llm_data."
            )
        neighbors_path = str(
            Path(config["data"]["data_dir"]) / "semantic_neighbors.pkl"
        )
        return LightGCN_LLM_Distill(
            num_users, num_items, model_cfg, norm_adj,
            llm_user_emb=llm_data.llm_user_emb,
            llm_item_emb=llm_data.llm_item_emb,
            freeze_llm=config.get("model", {}).get("freeze_llm", True),
            neighbors_path=neighbors_path,
        )

    elif model_name == "LightGCN_LLM_HardBPR":
        if llm_data is None:
            raise ValueError(
                "LightGCN_LLM_HardBPR requires llm_data."
            )
        return LightGCN_LLM_HardBPR(
            num_users, num_items, model_cfg, norm_adj,
            llm_user_emb=llm_data.llm_user_emb,
            llm_item_emb=llm_data.llm_item_emb,
            freeze_llm=config.get("model", {}).get("freeze_llm", True),
        )

    elif model_name == "LightGCN_LLM_Full":
        if llm_data is None:
            raise ValueError(
                "LightGCN_LLM_Full requires llm_data."
            )
        return LightGCN_LLM_Full(
            num_users, num_items, model_cfg, norm_adj,
            llm_user_emb=llm_data.llm_user_emb,
            llm_item_emb=llm_data.llm_item_emb,
            freeze_llm=config.get("model", {}).get("freeze_llm", True),
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}. Supported: LightGCN, LightGCN_LLM, LightGCN_LLM_Distill, LightGCN_LLM_HardBPR, LightGCN_LLM_Full"
        )


# ── 数据准备 ─────────────────────────────────────────────


def prepare_data(config: Dict[str, Any], logger) -> tuple:
    """根据配置加载或预处理数据。

    支持两种数据源：
    - movielens：通过 data/preprocess.py 加载和留一法划分
    - rlmrec：通过 data/rlmrec_loader.py 加载预处理好的 pickle 数据

    Returns
    -------
    (train_pairs, val_pairs, test_pairs, train_user_items,
     num_users, num_items, llm_data_or_None)
    """
    data_cfg = config["data"]
    data_type = data_cfg.get("type", "movielens")

    if data_type == "movielens":
        logger.info(f"Loading MovieLens from {data_cfg['data_dir']} ...")
        interactions = load_movielens(data_cfg["data_dir"])
        data = leave_one_out_split(
            interactions,
            min_rating=data_cfg.get("min_rating", 0),
            min_interactions=data_cfg.get("min_interactions", 5),
        )
        # 可选：保存处理后的数据
        processed_dir = data_cfg.get("processed_dir")
        if processed_dir:
            save_processed_data(data, processed_dir)

        return (data.train_pairs, data.val_pairs, data.test_pairs,
                data.train_user_items, data.num_users, data.num_items, None)

    elif data_type == "rlmrec":
        logger.info(f"Loading RLMRec data from {data_cfg['data_dir']} ...")
        llm_data = load_rlmrec_data(data_cfg["data_dir"])
        return (llm_data.train_pairs, llm_data.val_pairs, llm_data.test_pairs,
                llm_data.train_user_items, llm_data.num_users, llm_data.num_items,
                llm_data)

    else:
        raise ValueError(f"Unknown data type: {data_type}")


# ── 评估函数（自适应选择）─────────────────────────────────


@torch.no_grad()
def evaluate(
    model,
    target_pairs,
    train_user_items,
    ks,
    num_users: int = 0,
    threshold: int = 50000,
    eval_batch_size: int = 2048,
) -> Dict[str, float]:
    """自适应评估：小数据集用全量矩阵，大数据集用分批评估。"""
    model.eval()
    if num_users <= threshold:
        # 小数据集：一次性生成完整打分矩阵
        score_matrix = model.predict_all()
        return evaluate_model(score_matrix, target_pairs, train_user_items, ks)
    else:
        # 大数据集：分批评估
        return evaluate_model_batched(
            model, target_pairs, train_user_items, ks,
            eval_batch_size=eval_batch_size,
        )


# ── 训练循环 ────────────────────────────────────────────


def train_epoch(
    model,
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


# ── 主函数 ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a recommendation model (baseline or LLM-enhanced)."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--override", type=str, nargs="*",
        help="Override config values, e.g. training.epochs=50."
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
    (train_pairs, val_pairs, test_pairs,
     train_user_items, num_users, num_items, llm_data) = prepare_data(config, logger)

    logger.info(
        f"Data: {num_users} users, {num_items} items, "
        f"{len(train_pairs)} train, {len(val_pairs)} val, "
        f"{len(test_pairs)} test"
    )
    if llm_data is not None:
        logger.info(
            f"LLM embeddings loaded: "
            f"user {llm_data.llm_user_emb.shape}, "
            f"item {llm_data.llm_item_emb.shape}"
        )

    # 5. 构建邻接矩阵（仅使用训练集）
    logger.info("Building normalized adjacency matrix ...")
    norm_adj = build_normalized_adj(
        num_users, num_items, train_pairs, device=device
    )
    logger.info(f"Adjacency: {norm_adj.shape[0]}×{norm_adj.shape[1]}"
                 f" ({norm_adj._nnz()} edges)")

    # 6. 构建模型
    model = build_model(
        config["model"]["name"],
        num_users, num_items, config,
        norm_adj, llm_data=llm_data,
    ).to(device)

    # 打印参数信息
    param_info = model.count_parameters()
    if isinstance(param_info, dict):
        logger.info(f"Model: {config['model']['name']}")
        for k, v in param_info.items():
            if isinstance(v, int):
                logger.info(f"  {k}: {v:,}")
            else:
                logger.info(f"  {k}: {v}")
    else:
        logger.info(f"Model: {config['model']['name']}, params: {param_info:,}")

    # 7. 构建数据加载器和优化器
    # Full 模型使用语义硬负采样
    use_hard_neg = config["model"]["name"] == "LightGCN_LLM_Full"
    if use_hard_neg:
        import pickle, numpy as np
        topk_path = Path(config["data"]["data_dir"]) / "semantic_topk.pkl"
        try:
            with open(topk_path, "rb") as f:
                sem_topk = pickle.load(f)["indices"]
            logger.info(f"Loaded semantic top-k neighbors: {sem_topk.shape}")
        except FileNotFoundError:
            logger.warning(f"semantic_topk.pkl not found, falling back to random neg.")
            sem_topk = None

        train_loader = HardBPRBatchLoader(
            train_pairs=train_pairs,
            train_user_items=train_user_items,
            num_items=num_items,
            batch_size=config["training"]["batch_size"],
            seed=config["seed"],
            device=device,
            semantic_topk=np.array(sem_topk) if sem_topk is not None else None,
            hard_neg_prob=config["model"].get("hard_neg_prob", 0.5),
        )
    else:
        train_loader = BPRBatchLoader(
            train_pairs=train_pairs,
            train_user_items=train_user_items,
            num_items=num_items,
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
    eval_batch_size = config["evaluation"].get("eval_batch_size", 2048)
    best_val_score = float("-inf")
    best_state = None
    history = []

    logger.info(
        f"Starting training: {config['training']['epochs']} epochs max, "
        f"{steps_per_epoch} steps/epoch, "
        f"eval={'batched' if num_users > 50000 else 'full'}"
    )

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.invalidate_cache()

        # 训练
        epoch_loss = train_epoch(
            model, train_loader, optimizer,
            config["training"]["weight_decay"], steps_per_epoch,
        )

        # 验证
        val_metrics = evaluate(
            model, val_pairs, train_user_items, ks,
            num_users=num_users,
            eval_batch_size=eval_batch_size,
        )
        val_score = val_metrics[valid_metric]

        # 记录
        history.append({
            "epoch": epoch,
            "loss": epoch_loss,
            **{k: float(v) for k, v in val_metrics.items()},
        })

        # 日志
        if (epoch == 1 or epoch % config["training"]["log_interval"] == 0
                or epoch == config["training"]["epochs"]):
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in val_metrics.items()
                if k.startswith("Recall") or k.startswith("NDCG")
            )
            logger.info(
                f"Epoch {epoch:03d} | loss={epoch_loss:.4f} | {metric_str}"
            )

        # 保存最佳
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {
                "model": {k: v.cpu().clone()
                           for k, v in model.state_dict().items()},
                "epoch": epoch,
                "val_metrics": val_metrics,
            }

        # 早停
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
    test_metrics = evaluate(
        model, test_pairs, train_user_items, ks,
        num_users=num_users,
        eval_batch_size=eval_batch_size,
    )

    logger.info(f"Best epoch: {best_state['epoch']}")
    logger.info("Test metrics:")
    for name, value in test_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # 10. 保存结果
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_dir / "best_model.pt")
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
