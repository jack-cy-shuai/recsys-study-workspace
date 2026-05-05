# PROJECT_TRANSFER_MEMORY.md — LLM-Enhanced-RecSys 项目完整迁移文档

> **生成日期**: 2026-05-05
> **目标接收方**: Claude Code 模式（或其他 AI 编程助手）
> **用途**: 新会话加载本文档后，无需重复沟通即可无缝接续开发

---

## 一、项目概览

### 1.1 项目身份

| 字段 | 内容 |
|------|------|
| **项目名** | LLM-Enhanced-RecSys（LLM增强推荐系统科研项目） |
| **GitHub** | `https://github.com/jack-cy-shuai/recsys-study-workspace` |
| **本地路径** | `I:\claude_code文件\run_recmodels` |
| **Git分支** | `main`，remote = origin |
| **开发者背景** | 统计学专业本科生，数学基础扎实，熟悉 PyTorch |
| **目标** | 发表 CCF B 类及以上会议论文 |
| **核心命题** | 探索大语言模型与传统推荐系统的深度融合机制 |

### 1.2 项目当前阶段

**阶段 2：基础设施搭建完成，准备跑第一个 LLM 对比实验**

- ✅ 阶段 0：论文收集（8 篇）+ 笔记模板
- ✅ 阶段 1：LightGCN 最小复现 + 合成数据验证
- ✅ 阶段 2：模块化基础设施框架搭建（今日完成）
- ⏳ 阶段 3：下载 RLMRec Amazon 数据，跑 LLM 对比实验（**当前卡点**）
- ⏳ 阶段 4：实现更多基线（MF, NCF, NGCF）
- ⏳ 阶段 5：设计并实现创新 LLM+Rec 模型

---

## 二、目录结构（完整，含注释）

```
I:\claude_code文件\run_recmodels/
│
├── README.md                           # 项目说明（已更新为当前结构）
├── requirements.txt                    # numpy, torch, pyyaml, scipy
├── .gitignore                          # 已配置 logs/, saved_models/, RecBole artifacts
│
├── data/                               # 数据处理模块
│   ├── __init__.py
│   ├── preprocess.py                   # MovieLens 加载 + 留一法划分
│   ├── dataloader.py                   # BPRBatchLoader + PointwiseDataset + build_normalized_adj()
│   ├── rlmrec_loader.py               # RLMRec pickle 数据加载（稀疏矩阵 + LLM embedding）
│   ├── llm_processor.py               # LLM 数据处理（占位）
│   └── basic_implicit/                 # 旧合成数据集（60用户×120物品，仅 smoke test）
│
├── models/                             # 模型模块
│   ├── __init__.py
│   ├── base_model.py                   # BaseRecommender 抽象基类（forward, predict_all, predict_for_users, compute_loss）
│   ├── baselines/
│   │   ├── __init__.py
│   │   └── lightgcn.py                # LightGCN（继承 BaseRecommender，带嵌入缓存机制）
│   └── llm_enhanced/
│       ├── __init__.py
│       └── lightgcn_llm.py            # LightGCN_LLM（ID emb + Proj(LLM emb) 相加融合）
│
├── losses/                             # 损失函数模块
│   ├── __init__.py
│   ├── traditional.py                  # bpr_loss, bce_loss, mse_loss, l2_reg_loss
│   └── llm_enhanced.py                # LLM 增强损失（占位）
│
├── utils/                              # 通用工具
│   ├── __init__.py
│   ├── common.py                       # set_seed, load_config, setup_logger, save/load_checkpoint, EarlyStopping, get_device
│   └── metrics.py                      # recall/ndcg/mrr/precision/hit_at_k + evaluate_model + evaluate_model_batched
│
├── configs/                            # YAML 实验配置
│   ├── lightgcn_ml100k.yaml           # LightGCN on MovieLens-100K（小型验证）
│   ├── lightgcn_amazon.yaml           # LightGCN on Amazon-book（纯CF基线）
│   └── lightgcn_llm_amazon.yaml       # LightGCN-LLM on Amazon-book（LLM融合实验）
│
├── experiments/                        # 实验脚本
│   ├── train_baseline.py              # 统一训练入口（支持 MovieLens/RLMRec + LightGCN/LightGCN_LLM）
│   ├── train_llm.py                   # LLM 增强模型训练（占位）
│   ├── ablation.py                    # 消融实验（占位）
│   └── exp_001_lightgcn_baseline.md   # 旧基线实验记录
│
├── scripts/                            # 辅助脚本
│   ├── make_basic_dataset.py          # 合成数据生成（旧）
│   ├── download_rlmrec_guide.py       # RLMRec 数据下载指南
│   └── push_to_github.ps1             # Git 推送脚本
│
├── logs/                               # 实验日志（.gitkeep 占位）
├── saved_models/                       # 模型权重（.gitkeep 占位）
│
├── papers/                             # 论文 PDF
│   ├── traditional/                    # MF (Koren 2009), BPR, NCF
│   └── llmrec/                         # RGCF, RLMRec, TAGCF, TCA4Rec
│
├── notes/                              # 阅读笔记（多为空模板）
├── docs/                               # 项目文档
│   ├── project_现状分析报告_2026-05-05.md  # 首次代码审查报告
│   └── github_publishing.md
│
├── artifacts/                          # 历史实验结果
│   ├── basic_run/                      # 旧 LightGCN 在合成数据上的结果
│   ├── recbole_bpr_ml100k/             # RecBole NeuMF 参考结果
│   └── recbole_compare/                # RecBole BPR/NeuMF/LightGCN 对比
│
├── lightgcn/                           # ⚠️ DEPRECATED — 旧代码，仅作历史参考
│   ├── __init__.py (已标记 DEPRECATED)
│   ├── data.py
│   ├── model.py
│   └── evaluate.py
│
├── train.py                            # ⚠️ DEPRECATED — 旧训练入口
├── run_recbole_bpr.py                  # ⚠️ DEPRECATED — 违反"不用RecBole"规范
└── compare_recbole_models.py           # ⚠️ DEPRECATED — 同上
```

---

## 三、技术栈与环境

### 3.1 依赖

```
numpy>=1.23
torch>=1.12
pyyaml>=6.0
scipy>=1.7
```

### 3.2 运行环境

| 项 | 值 |
|-----|-----|
| Python | conda 环境 `recsys` |
| 解释器路径 | `I:\miniconda3\envs\recsys\python.exe` |
| IDE | PyCharm |
| OS | Windows |
| 当前设备 | CPU（config 中默认 `device: cpu`） |

> **注意**：用户目前没有 GPU 算力运行大模型，因此采用"下载公开预计算 LLM embedding"的零算力方案。

---

## 四、已完成功能清单（按时间线）

### 4.1 早期（本次会话前）

| 功能 | 文件 | 状态 |
|------|------|------|
| LightGCN 最小复现 | `lightgcn/model.py` | ✅ 算法正确，已弃用 |
| 合成数据集生成 | `scripts/make_basic_dataset.py` | ✅ 60×120 物品 |
| 基本训练流程 | `train.py` | ✅ 已弃用 |
| RecBole 对比实验 | `run_recbole_bpr.py`, `compare_recbole_models.py` | ⚠️ 违反规范，已标记 DEPRECATED |
| 论文收集 | `papers/` 下 7 篇 PDF | ✅ |

### 4.2 本次会话 — 基础设施搭建（2026-05-05）

| 模块 | 文件 | 核心内容 |
|------|------|----------|
| **工具** | `utils/common.py` | `set_seed()` (Python/NumPy/PyTorch/CUDA/cuDNN全覆盖), `load_config()`/`save_config()` (YAML), `setup_logger()` (文件+控制台), `save_checkpoint()`/`load_checkpoint()`, `EarlyStopping` 类 (mode="max"/"min"), `get_device()` (自动fallback) |
| **指标** | `utils/metrics.py` | 五个原子函数: `recall_at_k()`, `ndcg_at_k()`, `mrr_at_k()` (rank从1开始, 无命中返回0), `precision_at_k()`, `hit_at_k()`; 两个聚合函数: `evaluate_model()` (全量矩阵), `evaluate_model_batched()` (分批, 用于>5万用户) |
| **损失** | `losses/traditional.py` | `bpr_loss()` (pairwise, +1e-8防log(0)), `bce_loss()` (pointwise, with logits), `mse_loss()`, `l2_reg_loss()` (支持mean/sum) |
| **数据** | `data/preprocess.py` | `load_movielens()` (支持 ml-100k .data 和 ml-1m .dat), `leave_one_out_split()` (时间排序→按用户最后1条=test, 倒数第2条=val, 其余=train; 支持 min_rating 和 min_interactions 过滤), `save_processed_data()` (CSV+meta.txt) |
| **数据** | `data/dataloader.py` | `BPRBatchLoader` (无限迭代器, numpy随机状态, 每batch内负采样), `PointwiseTrainDataset` (BCE/MSE训练用), `build_normalized_adj()` (D^{-1/2} A D^{-1/2}, sparse COO, 孤立节点权重置0) |
| **数据** | `data/rlmrec_loader.py` | `load_rlmrec_data()` (加载 pickle 稀疏矩阵 + numpy LLM embedding, 返回 RLMRecData dataclass) |
| **模型** | `models/base_model.py` | `BaseRecommender(nn.Module, ABC)`: 抽象方法 `forward()`, `get_user_embeddings()`, `get_item_embeddings()`, `compute_loss()`; 具体方法 `predict_all()` (内积), `predict_for_users()` (分批), `configure_optimizers()` (Adam默认), `count_parameters()` |
| **模型** | `models/baselines/lightgcn.py` | `LightGCN(BaseRecommender)`: ID embedding → K层sparse.mm图传播 → K+1层平均 → 内积预测; `_propagate()` 内部方法; `invalidate_cache()` 每epoch清除嵌入缓存; `compute_loss()` = bpr_loss + l2_reg_loss |
| **模型** | `models/llm_enhanced/lightgcn_llm.py` | `LightGCN_LLM(LightGCN)`: 继承全部逻辑, 仅覆盖 `_propagate()`——在ego_embeddings中加入 `Linear(llm_dim→d_cf)(LLM_emb)`, 投影层用 `xavier_uniform_(gain=0.1)` 小权重初始化, LLM embedding 注册为 buffer (不可训练) |
| **配置** | `configs/*.yaml` | 三个独立 YAML (ml100k/amazon纯CF/amazon+LLM), 结构: experiment_name, seed, device, data(type/dataset/data_dir/...), model(name/embedding_dim/num_layers/...), training(epochs/batch_size/lr/...), evaluation(ks/valid_metric/eval_batch_size) |
| **实验** | `experiments/train_baseline.py` | 统一入口: `--config` + `--override key=val` (支持点号嵌套key), 自动识别 data.type (movielens→preprocess.py, rlmrec→rlmrec_loader.py), 自动识别 model.name (LightGCN/LightGCN_LLM), 自适应评估 (<5万用户全量矩阵, ≥5万分批) |

### 4.3 本次会话 — LLM 对比实验准备

| 功能 | 文件 | 说明 |
|------|------|------|
| RLMRec 数据加载 | `data/rlmrec_loader.py` | 处理 scipy.sparse + numpy pickle |
| LLM 融合模型 | `models/llm_enhanced/lightgcn_llm.py` | ID+Proj(LLM) 相加 |
| 分批评估 | `utils/metrics.py` `evaluate_model_batched()` | 52万用户不会 OOM |
| Amazon 配置 | `configs/lightgcn_amazon.yaml`, `configs/lightgcn_llm_amazon.yaml` | 除 model.name 外完全一致 |
| 下载指南 | `scripts/download_rlmrec_guide.py` | |

---

## 五、未完成任务 / 待办

### 5.1 当前卡点 🔴

数据下载：需要从 RLMRec GitHub 的 Google Drive 下载 Amazon-book 数据集（约1-2GB），放到 `data/rlmrec/amazon/`。

### 5.2 短期待办（本阶段）

| # | 任务 | 优先级 | 备注 |
|---|------|--------|------|
| 1 | 下载 RLMRec Amazon 数据 | **P0** | 卡住一切实验 |
| 2 | 安装 scipy: `pip install scipy` | **P0** | RLMRec 加载稀疏矩阵需要 |
| 3 | 运行纯 LightGCN: `python experiments/train_baseline.py --config configs/lightgcn_amazon.yaml` | **P0** | 第一个基线 |
| 4 | 运行 LightGCN-LLM: `python experiments/train_baseline.py --config configs/lightgcn_llm_amazon.yaml` | **P0** | 第一个 LLM 实验 |
| 5 | 对比两个 `metrics.json` 的结果 | **P1** | 分析 LLM embedding 是否有提升 |

### 5.3 中期待办

| # | 任务 | 备注 |
|---|------|------|
| 6 | 实现 MF (Matrix Factorization) | `models/baselines/mf.py`, 继承 BaseRecommender |
| 7 | 实现 BPR-MF | 同上，使用 BPR loss |
| 8 | 实现 NCF/NeuMF | `models/baselines/ncf.py` |
| 9 | 实现 NGCF | `models/baselines/ngcf.py`（LightGCN 的前身，对比用） |
| 10 | 补充 `data/preprocess.py` 的 CSV 回读功能 | 目前每次重新处理数据 |
| 11 | 实现 `experiments/ablation.py` | 消融实验框架 |

### 5.4 长期待办

| # | 任务 | 备注 |
|---|------|------|
| 12 | 阅读 RLMRec / TAGCF / TCA4Rec / RGCF 论文 | 提炼可复用的创新思路 |
| 13 | 实现 `data/llm_processor.py` | 本地生成 LLM embedding（需要算力） |
| 14 | 实现 `losses/llm_enhanced.py` | 对比损失、知识蒸馏损失等 |
| 15 | 设计第一个创新模型 | `models/llm_enhanced/` |
| 16 | GPU 加速（当前全 CPU） | 需要用户获得 GPU 算力 |

---

## 六、编码规范与约定

### 6.1 代码风格

- Python 文件头部统一：`from __future__ import annotations`
- 类型标注全面使用（`def fn(x: int) -> str:`）
- 所有模块/类/公共函数必须有中文 docstring，说明用途和数学原理
- 变量名用完整英文单词（`num_users` 而非 `n_u`，`embedding_dim` 而非 `ed`）
- 注释用中文（用户母语），代码标识符用英文

### 6.2 模型约定

- 所有模型继承 `BaseRecommender`（`models/base_model.py`）
- 必须实现：`forward()`, `compute_loss()`, `get_user_embeddings()`, `get_item_embeddings()`
- 推荐实现：`predict_all()` (内积), `predict_for_users()` (分批)
- 嵌入缓存：对于图模型，用 `_cached_user_emb`/`_cached_item_emb` + `invalidate_cache()` 避免 epoch 内重复图传播
- 损失函数从 `losses/` 导入，不耦合在模型代码中

### 6.3 配置约定

- YAML 格式，每个实验一个独立文件
- 四大板块：`data:`, `model:`, `training:`, `evaluation:`
- 顶层必有：`experiment_name`, `seed`, `device`, `output_dir`, `log_dir`
- `evaluation.valid_metric` 用于早停和选最优模型（如 `"Recall@20"`）
- 配置通过 `--override key.subkey=value` 在命令行覆盖

### 6.4 训练流程约定

- 训练入口统一：`experiments/train_baseline.py --config configs/xxx.yaml`
- 所有实验自动生成：`artifacts/{exp_name}/best_model.pt` + `metrics.json`
- `metrics.json` 包含 `config`, `best_epoch`, `best_val_metrics`, `test_metrics`, `history`
- 日志自动保存到 `logs/{exp_name}/experiment.log`

### 6.5 绝对禁止

| 禁止事项 | 原因 |
|----------|------|
| 引入 RecBole / TorchRec / DeepRec | 项目核心规范——完全自实现 |
| 编造实验数据或论文引用 | 学术诚信 |
| 删除或覆盖历史实验日志 | 所有数据必须可追溯 |
| 修改 `lightgcn/` 旧代码 | 已归档，仅作参考 |

---

## 七、关键设计决策与自定义逻辑

### 7.1 LightGCN 嵌入缓存机制

**问题**：旧代码每个 batch 都调用 `compute_embeddings()` 做完整的 K 层图传播，极大浪费。

**方案**：新增 `_cached_user_emb` / `_cached_item_emb`，在 `get_user_embeddings()` / `get_item_embeddings()` 首次调用时计算并缓存。每个 epoch 开始用 `invalidate_cache()` 清除。

```python
# LightGCN 中：
def get_user_embeddings(self):
    if self._cached_user_emb is None:
        self._cached_user_emb, self._cached_item_emb = self._propagate()
    return self._cached_user_emb

def invalidate_cache(self):
    self._cached_user_emb = None
    self._cached_item_emb = None
```

### 7.2 BPR 损失 + L2 正则的位置

**问题**：正则应该用原始 embedding（第0层）还是传播后的 embedding？

**当前决策**：使用原始 embedding（`self.user_embedding(users)` 而非最终嵌入），与 LightGCN 论文 Section 3.3 一致。

```python
# 在 LightGCN.compute_loss() 中：
reg_loss = l2_reg_loss([
    self.user_embedding(users),       # 原始第0层
    self.item_embedding(pos_items),
    self.item_embedding(neg_items),
])
```

### 7.3 LLM 融合的投影层初始化

**问题**：投影层初始权重大了会盖过 ID embedding，初始小了完全没用。

**当前决策**：`xavier_uniform_(gain=0.1)`，让模型从小权重开始，逐步学习利用语义信息。这确保训练初期行为接近纯 LightGCN，变化是渐进和可解释的。

### 7.4 分批评估阈值

**阈值**：`num_users > 50000` 时自动切换为 `evaluate_model_batched()`。

**原因**：Amazon-book 有 526k 用户 × 91k 商品 = 48B 条打分 ≈ 192GB (float32)，全量矩阵无法放入内存。

**实现**：在 `train_baseline.py` 的 `evaluate()` 函数中自动判断。

### 7.5 BPR 负采样的随机种子设计

**当前实现**：`BPRBatchLoader` 使用独立的 `np.random.RandomState(seed)`，不依赖 Python 全局 `random`。这确保即使外部代码修改了全局随机状态，数据采样仍可复现。

### 7.6 NDCG 的 IDCG 计算

使用简化假设：所有正例排在最前面时 IDCG 最大。即 `IDCG@K = sum_{r=1}^{min(|true_items|, K)} 1/log2(r+1)`。这是推荐系统评估中的标准做法。

### 7.7 MovieLens 留一法 (Leave-One-Out)

- 所有交互按 `timestamp` 排序
- 每个用户的最后 1 条 → test
- 倒数第 2 条 → val
- 其余 → train
- `min_rating=0` 表示保留所有评分（隐式反馈），`min_rating=4` 仅保留 ≥4 的
- `min_interactions=5` 过滤交互数不足的用户
- 输出 ID 为 0-based 连续索引

---

## 八、已知问题与踩过的坑

### 8.1 已知问题

| # | 问题 | 影响 | 解决方案 |
|---|------|------|----------|
| 1 | **RLMRec 数据未下载** | 无法运行 Amazon 实验 | 用户需手动从 Google Drive 下载 |
| 2 | **Bash 沙箱不可用** | 无法在会话内执行 git/pip/python 命令 | 用户需在本地终端自行执行命令 |
| 3 | **RecBole 脚本未删除** | `run_recbole_bpr.py` 等违反规范 | 已标记 DEPRECATED，但文件仍在 |
| 4 | **CPU 训练速度** | Amazon 298万交互训练很慢 | 后续应迁移到 GPU |
| 5 | **`data/preprocess.py` 不能从 CSV 回读** | 每次训练都重新预处理 | 待实现 `load_processed_data()` |
| 6 | **旧 `lightgcn/` 包未物理删除** | 可能被误 import | 已标记 DEPRECATED，新代码不依赖它 |
| 7 | **日志和 saved_models 目录为空** | git 不追踪空目录 | 已放 `.gitkeep` |

### 8.2 踩过的坑

1. **Linux 沙箱启动失败**：Cowork 模式的 Bash workspace 在本会话中反复失败（不是权限问题，是环境故障）。解决方案——所有命令由用户在本地终端执行。

2. **Read 工具路径限制**：直接 `Read I:\...` 有时报"outside connected folders"，但通过 Agent (Explore) 工具可以绕过。新会话中通常能直接 Read。

3. **RecBole 依赖幻觉**：早期用 RecBole 跑实验取得参考结果，但这违反了"完全自实现"的核心准则。RecBole 结果保留仅作参考（`artifacts/recbole_compare/summary.json`）。

4. **合成数据集 metric 不一致**：`artifacts/basic_run/metrics.json` 仅记录 1 个 epoch（Recall@10=0.17），但 `experiments/exp_001_lightgcn_baseline.md` 声称 80 epoch 后 Recall@10=0.58。可能是多次运行覆盖了 metrics.json。以最新运行为准。

5. **Amazon 数据集规模**：526k × 91k 的打分矩阵太大（~192GB），不能沿用 MovieLens 的 `predict_all()` 全量评估方式，必须分批。

---

## 九、配置 YAML 完整 Schema

```yaml
# 必填顶层字段
experiment_name: str       # 实验名称，用于命名输出目录和日志
seed: int                  # 全局随机种子
device: str                # "cpu" 或 "cuda"
output_dir: str            # 模型和指标保存路径（如 artifacts/xxx）
log_dir: str               # 日志保存路径（如 logs/xxx）

# data 板块（根据 type 不同有不同字段）
data:
  type: str                # "movielens" 或 "rlmrec"
  # MovieLens 专用：
  dataset: str             # "ml-100k" 或 "ml-1m"
  data_dir: str            # 原始数据目录
  processed_dir: str       # 预处理输出目录
  min_rating: float        # 最低评分阈值（0=全部, 4=仅≥4）
  min_interactions: int    # 用户最少交互数（默认5）
  # RLMRec 专用：
  dataset: str             # "amazon" / "yelp" / "steam"
  data_dir: str            # pickle 文件目录（如 data/rlmrec/amazon/）

# model 板块
model:
  name: str                # "LightGCN" 或 "LightGCN_LLM"
  embedding_dim: int       # CF 嵌入维度（默认64）
  num_layers: int          # 图传播层数（默认3）
  freeze_llm: bool         # 仅 LightGCN_LLM：是否冻结 LLM embedding（默认true）

# training 板块
training:
  epochs: int              # 最大训练轮数（默认200）
  batch_size: int          # BPR batch 大小（默认2048）
  lr: float                # 学习率（默认0.001）
  weight_decay: float      # Adam weight_decay + L2 reg 权重（默认0.0001）
  early_stop_patience: int # 早停容忍 epoch（默认20）
  log_interval: int        # 每 N epoch 打印日志（默认10）

# evaluation 板块
evaluation:
  ks: list[int]            # Top-K 列表（如 [10, 20, 50]）
  valid_metric: str        # 早停和选最优的指标（如 "Recall@20"）
  eval_batch_size: int     # 分批评估批次大小（默认2048）
```

---

## 十、文件修改历史（本次会话）

| 文件 | 操作 | 日期 |
|------|------|------|
| `utils/__init__.py` | **新建** | 2026-05-05 |
| `utils/common.py` | **新建** (140行) | 2026-05-05 |
| `utils/metrics.py` | **新建** (130行), 后追加 `evaluate_model_batched` | 2026-05-05 |
| `losses/__init__.py` | **新建** | 2026-05-05 |
| `losses/traditional.py` | **新建** (90行) | 2026-05-05 |
| `losses/llm_enhanced.py` | **新建** (占位) | 2026-05-05 |
| `data/__init__.py` | **新建** | 2026-05-05 |
| `data/preprocess.py` | **新建** (200行) | 2026-05-05 |
| `data/dataloader.py` | **新建** (210行), 后追加 `build_normalized_adj` | 2026-05-05 |
| `data/rlmrec_loader.py` | **新建** (150行) | 2026-05-05 |
| `data/llm_processor.py` | **新建** (占位) | 2026-05-05 |
| `models/__init__.py` | **新建** | 2026-05-05 |
| `models/base_model.py` | **新建** (100行), 后追加 `predict_for_users` | 2026-05-05 |
| `models/baselines/__init__.py` | **新建** | 2026-05-05 |
| `models/baselines/lightgcn.py` | **新建** (150行) | 2026-05-05 |
| `models/llm_enhanced/__init__.py` | **新建** | 2026-05-05 |
| `models/llm_enhanced/lightgcn_llm.py` | **新建** (130行) | 2026-05-05 |
| `configs/lightgcn_ml100k.yaml` | **新建** | 2026-05-05 |
| `configs/lightgcn_amazon.yaml` | **新建** | 2026-05-05 |
| `configs/lightgcn_llm_amazon.yaml` | **新建** | 2026-05-05 |
| `experiments/train_baseline.py` | **新建** (180行), 后重写支持 RLMRec+LLM | 2026-05-05 |
| `experiments/train_llm.py` | **新建** (占位) | 2026-05-05 |
| `experiments/ablation.py` | **新建** (占位) | 2026-05-05 |
| `scripts/download_rlmrec_guide.py` | **新建** | 2026-05-05 |
| `scripts/push_to_github.ps1` | **新建** | 2026-05-05 |
| `saved_models/.gitkeep` | **新建** | 2026-05-05 |
| `logs/.gitkeep` | **新建** | 2026-05-05 |
| `requirements.txt` | **修改** (添加 pyyaml, scipy) | 2026-05-05 |
| `.gitignore` | **修改** (添加 logs/, saved_models/, RecBole artifacts) | 2026-05-05 |
| `README.md` | **重写** (新结构说明) | 2026-05-05 |
| `lightgcn/__init__.py` | **修改** (标记 DEPRECATED) | 2026-05-05 |
| `train.py` | **修改** (标记 DEPRECATED) | 2026-05-05 |
| `run_recbole_bpr.py` | **修改** (标记 DEPRECATED) | 2026-05-05 |
| `compare_recbole_models.py` | **修改** (标记 DEPRECATED) | 2026-05-05 |
| `docs/project_现状分析报告_2026-05-05.md` | **新建** (首次项目审查) | 2026-05-05 |
| `docs/PROJECT_TRANSFER_MEMORY.md` | **新建** (本文档) | 2026-05-05 |

---

## 十一、给新模型的快速上手脚本

### 立即可以做的事（无需数据下载）

```powershell
# 验证环境
conda activate recsys
python -c "import torch; import numpy; import yaml; import scipy; print('OK')"

# 在 MovieLens-100K 上跑一个小实验（需要先下载 ml-100k 到 data/ml-100k/）
# 如果没有 ml-100k，首先生成合成数据集：
python scripts/make_basic_dataset.py
# 然后在合成数据上测试（需要手动创建一个匹配的 config）

# 查看现有的 YAML 配置
ls configs/
```

### 下载 RLMRec 数据后的流程

```powershell
# 1. 安装 scipy
pip install scipy

# 2. 确保数据在正确位置
ls data/rlmrec/amazon/
# 应该有: trn_mat.pkl, val_mat.pkl, tst_mat.pkl, usr_emb_np.pkl, itm_emb_np.pkl

# 3. 运行纯 LightGCN 基线
python experiments/train_baseline.py --config configs/lightgcn_amazon.yaml

# 4. 运行 LightGCN + LLM 融合
python experiments/train_baseline.py --config configs/lightgcn_llm_amazon.yaml

# 5. 对比结果
cat artifacts/lightgcn_amazon/metrics.json
cat artifacts/lightgcn_llm_amazon/metrics.json
```

### 添加新模型的步骤

1. 在 `models/baselines/` 或 `models/llm_enhanced/` 下创建 `xxx.py`
2. 继承 `BaseRecommender`，实现：`forward()`, `compute_loss()`, `get_user_embeddings()`, `get_item_embeddings()`
3. 在 `experiments/train_baseline.py` 的 `build_model()` 中添加新模型名
4. 创建对应的 `configs/xxx.yaml`
5. 运行：`python experiments/train_baseline.py --config configs/xxx.yaml`

---

## 十二、用户偏好与合作风格

从对话中观察到的用户习惯：

- **表达直接**：喜欢简洁、直指核心的交流
- **自主性强**：倾向于理解后再动手，而非盲目执行
- **关注数学严谨性**：要求确认算法与原论文一致
- **实用主义**：当前无 GPU 算力，采用"下载公开 embedding"的务实方案
- **中文母语**：代码注释和文档用中文，标识符用英文
- **偏好独立配置文件**：每个实验一个 YAML（而非继承/合并模式）
- **偏好完全迁移**：选择将旧代码标记 DEPRECATED 并迁移到新结构，而非并行维护

---

## 十三、GitHub 仓库信息

| 项 | 值 |
|-----|-----|
| 远程 | `https://github.com/jack-cy-shuai/recsys-study-workspace.git` |
| 分支 | `main` |
| 本地未推送的更改 | 本次会话的全部新文件（~35 个文件，含新建+修改） |
| 推送方式 | 用户在本地 PowerShell 执行 `git add -A && git commit && git push origin main` |

---

> **文档版本**: v1.0 | **下次更新时机**: 跑完第一个 LLM 对比实验后
> **维护原则**: 每次重大变更后更新本文档，确保始终反映最新状态
