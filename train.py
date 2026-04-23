from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from lightgcn.data import build_dataset, sample_batch
from lightgcn.evaluate import evaluate_model
from lightgcn.model import LightGCN


def parse_args() -> argparse.Namespace:
    """定义命令行参数，便于后续直接在终端或 PyCharm 里改实验配置。"""

    parser = argparse.ArgumentParser(description="Minimal LightGCN reproduction")
    parser.add_argument("--data-path", type=str, default="data/basic_implicit/interactions.csv")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--eval-k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="artifacts/basic_run")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机种子，尽量保证每次训练结果可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # 当前默认走 CPU。
    # 如果之后你想用 GPU，可以运行：
    # `python train.py --device cuda`
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    # 读取数据并提前把稀疏邻接矩阵放到目标设备上。
    dataset = build_dataset(args.data_path, device=device)

    # 构建 LightGCN 模型。
    model = LightGCN(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.layers,
        norm_adj=dataset.norm_adj,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed)

    # 每个 epoch 抽取多少个 batch。
    # 这里用训练交互数近似决定步数，逻辑简单，适合最小复现。
    steps_per_epoch = max(1, len(dataset.train_pairs) // args.batch_size)
    best_val = float("-inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            # 采样 BPR 三元组并执行一步梯度更新。
            users, pos_items, neg_items = sample_batch(
                train_pairs=dataset.train_pairs,
                train_user_items=dataset.train_user_items,
                num_items=dataset.num_items,
                batch_size=args.batch_size,
                rng=rng,
                device=device,
            )
            optimizer.zero_grad()
            loss = model.bpr_loss(users, pos_items, neg_items, reg_weight=args.reg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= steps_per_epoch
        model.eval()

        # 先得到完整的用户-物品打分矩阵，再在验证集上算 Top-K 指标。
        score_matrix = model.predict()
        val_metrics = evaluate_model(
            score_matrix=score_matrix,
            target_pairs=dataset.val_pairs,
            train_user_items=dataset.train_user_items,
            ks=args.eval_k,
        )

        # 这里用最大的 K 对应的 Recall 作为“最佳模型”选择标准。
        # 你后续也可以改成 NDCG@10 或者别的业务更关心的指标。
        val_key = f"Recall@{max(args.eval_k)}"
        history.append({"epoch": epoch, "loss": epoch_loss, **val_metrics})

        if val_metrics[val_key] > best_val:
            best_val = val_metrics[val_key]
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
            }

        # 控制日志频率，避免每个 epoch 都打印得太长。
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            metric_text = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"Epoch {epoch:03d} | loss={epoch_loss:.4f} | val {metric_text}")

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    # 回滚到验证集最优模型，再对测试集做一次最终评估。
    model.load_state_dict(best_state["model"])
    model.eval()
    test_metrics = evaluate_model(
        score_matrix=model.predict(),
        target_pairs=dataset.test_pairs,
        train_user_items=dataset.train_user_items,
        ks=args.eval_k,
    )

    # 保存模型权重和指标，方便后面复盘和继续实验。
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_dir / "best_model.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_state["epoch"],
                "best_val_metrics": best_state["metrics"],
                "test_metrics": test_metrics,
                "history": history,
            },
            f,
            indent=2,
        )

    print(f"Best epoch: {best_state['epoch']}")
    print("Test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"Saved artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
