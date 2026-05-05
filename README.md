# LLM增强推荐系统科研项目 (LLM-Enhanced-RecSys)

探索大语言模型与传统推荐系统的深度融合机制。

## 项目结构

```
run_recmodels/
├── data/                    # 数据处理模块
│   ├── preprocess.py        # 原始数据加载、留一法 train/val/test 划分
│   ├── dataloader.py        # PyTorch DataLoader + BPR 负采样 + 邻接矩阵构建
│   └── llm_processor.py     # LLM 相关数据处理（占位，待实现）
├── models/                  # 所有模型的实现
│   ├── base_model.py        # 推荐模型统一抽象基类
│   ├── baselines/           # 基线模型（全部自实现）
│   │   └── lightgcn.py      # LightGCN
│   └── llm_enhanced/        # LLM 增强模型（占位，待实现）
├── losses/                  # 损失函数模块
│   ├── traditional.py       # BPR, BCE, MSE
│   └── llm_enhanced.py      # LLM 相关损失（占位，待实现）
├── utils/                   # 通用工具函数
│   ├── common.py            # 随机种子、YAML 配置读取、日志、早停、模型保存
│   └── metrics.py           # Recall@K, NDCG@K, MRR@K, Precision@K, Hit@K
├── experiments/             # 实验脚本
│   ├── train_baseline.py    # 基线模型统一训练入口
│   ├── train_llm.py         # LLM 增强模型训练（占位）
│   └── ablation.py          # 消融实验（占位）
├── configs/                 # 实验配置（YAML）
│   └── lightgcn_ml100k.yaml # LightGCN × MovieLens-100K
├── logs/                    # 实验日志（自动生成）
├── saved_models/            # 训练好的模型权重
├── papers/                  # 论文 PDF
│   ├── traditional/         # 经典推荐论文（MF, BPR, NCF）
│   └── llmrec/              # LLM+Rec 论文（RGCF, RLMRec, TAGCF, TCA4Rec）
├── notes/                   # 论文阅读笔记
├── docs/                    # 项目文档
├── scripts/                 # 辅助脚本（数据集生成等）
├── artifacts/               # 历史实验结果
├── requirements.txt
└── README.md
```

## 环境配置

使用本地 Conda 环境：

```powershell
conda run -n recsys python --version
```

安装依赖：

```powershell
conda run -n recsys pip install -r requirements.txt
```

PyCharm 解释器路径：

```
I:\miniconda3\envs\recsys\python.exe
```

## 快速开始

### 1. 准备 MovieLens-100K 数据

从 [GroupLens](https://grouplens.org/datasets/movielens/100k/) 下载 ml-100k.zip，
解压到 `data/ml-100k/` 目录。

### 2. 预处理数据

```powershell
conda run -n recsys python data/preprocess.py \
    --dataset ml-100k \
    --data-dir data/ml-100k \
    --output-dir data/processed_ml100k
```

### 3. 训练 LightGCN 基线

```powershell
conda run -n recsys python experiments/train_baseline.py \
    --config configs/lightgcn_ml100k.yaml
```

### 4. 覆盖配置参数

```powershell
conda run -n recsys python experiments/train_baseline.py \
    --config configs/lightgcn_ml100k.yaml \
    --override training.epochs=100 model.embedding_dim=128
```

## 输出

训练结果保存在 `artifacts/{experiment_name}/`：

- `best_model.pt` — 最佳 epoch 的模型权重
- `metrics.json` — 完整指标（验证集 + 测试集 + 训练历史）

## 开发进度

| 模块 | 状态 |
|------|------|
| 项目结构 | ✅ 已建立 |
| LightGCN 自实现 | ✅ 已迁移到新结构 |
| 评估指标 (5种) | ✅ 已实现 |
| 损失函数 (BPR/BCE/MSE) | ✅ 已实现 |
| MovieLens 数据预处理 | ✅ 已实现 |
| YAML 配置系统 | ✅ 已实现 |
| 统一训练脚本 | ✅ 已实现 |
| MF/BPR-MF 自实现 | ⏳ 待实现 |
| NCF/NeuMF 自实现 | ⏳ 待实现 |
| LLM 数据处理 | ⏳ 待实现 |
| LLM 增强模型 | ⏳ 待实现 |

## 核心准则

- 所有模型完全自实现，不依赖 RecBole/TorchRec 等封装库
- 所有随机种子固定，确保实验可复现
- 每个实验自动记录完整日志和指标
- 代码模块化，LLM 组件可插拔到任意基线模型

## 旧代码说明

`lightgcn/` 目录和 `train.py` 中的代码已被迁出至新结构，保留仅作历史参考。
`run_recbole_bpr.py` 和 `compare_recbole_models.py` 中的 RecBole 结果保留作为参考基准。
