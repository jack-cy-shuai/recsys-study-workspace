# Git push script for recsys-study-workspace
# Run this in PowerShell from the project root: I:\claude_code文件\run_recmodels\

Write-Host "=== RecSys Study Workspace - Git Push ===" -ForegroundColor Cyan

# Step 1: Check current status
Write-Host "`n[1/4] Checking git status..." -ForegroundColor Yellow
git status

# Step 2: Stage all changes (new modules + modified files + deprecated markers)
Write-Host "`n[2/4] Staging all changes..." -ForegroundColor Yellow
git add -A

# Show what will be committed
Write-Host "`nFiles to be committed:"
git diff --cached --name-status

# Step 3: Commit
Write-Host "`n[3/4] Committing..." -ForegroundColor Yellow
git commit -m "feat: 搭建项目基础设施框架

新建模块化代码结构，完全迁移旧的 lightgcn/ 扁平包：

新增模块：
- utils/common.py: 随机种子、YAML配置、日志系统、早停、模型保存
- utils/metrics.py: Recall/NDCG/MRR/Precision/Hit@K 五种标准指标
- losses/traditional.py: BPR/BCE/MSE 损失函数 + L2正则化
- data/preprocess.py: MovieLens 数据加载 + 留一法(leave-one-out)划分
- data/dataloader.py: BPRBatchLoader + PointwiseDataset + 邻接矩阵构建
- models/base_model.py: 推荐模型统一抽象基类
- models/baselines/lightgcn.py: LightGCN 迁移至新结构(继承BaseRecommender)
- configs/lightgcn_ml100k.yaml: 第一个完整实验配置
- experiments/train_baseline.py: 统一基线训练入口

清理：
- 旧 lightgcn/ 包、train.py、RecBole脚本 标记为 DEPRECATED
- 更新 requirements.txt (新增pyyaml)、.gitignore、README.md"

# Step 4: Push
Write-Host "`n[4/4] Pushing to origin/main..." -ForegroundColor Yellow
git push origin main

Write-Host "`n=== Done ===" -ForegroundColor Green
