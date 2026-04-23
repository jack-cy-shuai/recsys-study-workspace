from __future__ import annotations

import csv
import random
from pathlib import Path


def main() -> None:
    """生成一个可重复复现的小型隐式反馈数据集。

    这个数据集不是公开 benchmark，而是一个便于快速验证流程的基础样本：
    - 每个用户主要偏好某一类物品
    - 同时混入少量相邻类别物品
    - 每个用户固定留出 1 条验证、1 条测试

    这样模型可以学到一定规律，同时又足够小，适合先把训练和评估流程跑通。
    """

    rng = random.Random(2024)
    output_path = Path("data/basic_implicit/interactions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 这里故意把数据规模控制得比较小，方便 CPU 上也能快速跑通。
    num_users = 60
    num_items = 120
    num_groups = 6
    train_per_user = 8
    val_per_user = 1
    test_per_user = 1

    rows = []
    timestamp = 1

    # 把全部物品均匀分成若干组，每组可以理解为一个“兴趣主题”。
    items_by_group = {
        group: list(range(group * (num_items // num_groups), (group + 1) * (num_items // num_groups)))
        for group in range(num_groups)
    }

    for user in range(num_users):
        # 每个用户有一个主偏好组，并少量接触相邻组，
        # 这样数据里既有稳定偏好，也有一定噪声。
        primary_group = user % num_groups
        secondary_group = (primary_group + 1) % num_groups

        candidates = rng.sample(items_by_group[primary_group], train_per_user - 2)
        candidates += rng.sample(items_by_group[secondary_group], 2)
        held_out = rng.sample(
            [item for item in items_by_group[primary_group] if item not in candidates],
            val_per_user + test_per_user,
        )

        interactions = [(item, "train") for item in candidates]
        interactions.append((held_out[0], "val"))
        interactions.append((held_out[1], "test"))
        rng.shuffle(interactions)

        # 用 timestamp 只是为了保留一个类似真实日志的字段。
        # 当前模型不做时序建模，但保留时间列有利于以后换任务。
        for item, split in interactions:
            rows.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "split": split,
                    "timestamp": timestamp,
                }
            )
            timestamp += 1

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "item_id", "split", "timestamp"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved dataset to {output_path.resolve()}")
    print(f"Users: {num_users}, Items: {num_items}, Interactions: {len(rows)}")


if __name__ == "__main__":
    main()
