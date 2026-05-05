"""
DEPRECATED — 此脚本使用旧的 lightgcn/ 包。

新的训练入口（推荐）：
  python experiments/train_baseline.py --config configs/lightgcn_ml100k.yaml

如需在小的合成数据集上快速冒烟测试：
  python experiments/train_baseline.py --config configs/lightgcn_ml100k.yaml \
      --override data.dataset=basic data.data_dir=data/basic_implicit
"""
