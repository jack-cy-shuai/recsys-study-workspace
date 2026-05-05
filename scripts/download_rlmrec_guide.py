"""
RLMRec 数据下载指南
===================
RLMRec 提供了 Amazon-book / Yelp / Steam 三个数据集的预处理版本，
包含稀疏交互矩阵和预计算的 LLM 语义 embedding。

下载步骤：

1. 访问 RLMRec GitHub 仓库:
   https://github.com/HKUDS/RLMRec

2. 在 README 中找到 Google Drive 下载链接，
   或直接搜索 "RLMRec dataset Google Drive"

3. 下载 Amazon-book 数据集（约 1-2 GB 压缩包）

4. 解压到项目 data/rlmrec/amazon/ 目录下

5. 确保目录包含以下文件:
   data/rlmrec/amazon/
   ├── trn_mat.pkl
   ├── val_mat.pkl
   ├── tst_mat.pkl
   ├── usr_emb_np.pkl  ← LLM 用户语义向量
   ├── itm_emb_np.pkl  ← LLM 商品语义向量
   ├── usr_prf.pkl     (可选) LLM 用户画像文本
   └── itm_prf.pkl     (可选) LLM 商品画像文本

6. 运行训练:
   # 纯 LightGCN 基线
   python experiments/train_baseline.py --config configs/lightgcn_amazon.yaml

   # LightGCN + LLM 融合
   python experiments/train_baseline.py --config configs/lightgcn_llm_amazon.yaml

数据集规模参考:
  Amazon-book: 526k 用户, 91k 商品, 298万训练交互
  Yelp:        42k 用户, 86k 商品, 113万训练交互
  Steam:       61k 用户, 35k 商品, 35万训练交互

硬件建议:
  - CPU: 32GB+ RAM (训练用稀疏矩阵，评估需分批)
  - GPU: 8GB+ VRAM (推荐使用 GPU 加速)
  - 磁盘: ~5GB (解压后)
"""
print("请按照这份指南下载 RLMRec 数据集。")
print("下载地址见 RLMRec GitHub README 中的 Google Drive 链接。")
