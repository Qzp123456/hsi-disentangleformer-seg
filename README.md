# HSI Semantic Segmentation via Spatial-Channel Disentangled Representation Learning

MSc 论文代码库（University of Leeds, School of Computing）。

把 DisentangleFormer（Liao et al., arXiv 2512.04314，原为逐像素分类）扩展为**端到端语义分割框架**：
- 整图切 tile（32×32）密集预测 + SegDecoder，mIoU 评估
- 额外实现了 gated CMA 双向跨流融合模块（负结果，见实验结果）
- 三种训练/测试划分协议：random（有重叠泄漏）/ spatial 左右切 / checker 棋盘式（无泄漏）

## 环境

```bash
module load miniforge/24.7.1     # Leeds Aire HPC；本地可用任意 conda
conda create -n hsi python=3.10
conda activate hsi
pip install torch==2.5.1 numpy scipy scikit-learn matplotlib
```

GPU: NVIDIA L40S 48GB（单卡即可，模型仅 0.3M 参数）。

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | DisentangleFormer 核心：窗口注意力（Swin 式 8×8）、ST/CT 双流、STE、MS-FFN、CMAFusion（γ 零初始化门控） |
| `seg_model.py` | 分割版 backbone + SegDecoder |
| `seg_dataset.py` | 整图切 32×32 tile；三种划分协议：`random` / `spatial`（左60%训右40%测）/ `checker`（棋盘格，配置见 `CHECKER_CONFIG`）；标签 0 → ignore 255 |
| `dataset.py` | 数据集注册表（Indian Pines / Pavia / Houston 2013）+ 分类用 patch 工具 |
| `seg_train_all.py` | 统一训练入口：`--model {disentangle,vit,spectralformer,cnn3d}` + `--use_cma` + `--split {random,spatial,checker}` + `--train_ratio` |
| `baselines.py` | ViTSeg / SpectralFormerSeg / CNN3DSeg 三个基线 |
| `seg_train.py` / `train.py` | 早期版本的训练脚本（保留备用） |
| `summarise.py` | 汇总日志 → 表格 + LaTeX；`--spatial` / `--checker` 单协议表，`--gap` 三协议对比表 |
| `visualise.py` | 5 面板定性对比图 + 差异图（整图滑窗推理） |
| `eval_perclass.py` | 空间/棋盘协议逐类 IoU（`--tag {sp,ch}`） |
| `analyze_gamma.py` | 打印 CMAFusion 各层 gamma_s/gamma_c 门控值（CMA 是否被网络关掉的证据） |
| `analyze_cca.py` | ST/CT 两流典型相关分析（含 shuffled 零假设地板），判断 CMA 是否破坏解耦 |
| `analyze_split_classes.py` | 诊断划分协议：逐类训练/测试区像素统计 + 棋盘偏移扫描（`--scan`） |
| `bench.sh` | 主 benchmark：5 方法 × 3 seeds × 2 数据集（随机协议） |
| `bench_sp.sh` | 空间左右切协议 benchmark |
| `bench_ch.sh` | 棋盘协议 benchmark |
| `bench_houston.sh` | Houston 2013 随机协议 benchmark |
| `low_ratio.sh` / `more_seeds.sh` | 低监督消融 / 10-seed 统计检验 |
| `*_job.sh` | Slurm 作业脚本（cca / perclass） |

## 数据准备

放到 `DATA_ROOT`（默认 `/mnt/scratch/znzs0468/data`）：

| 数据集 | 文件 | 来源 |
|---|---|---|
| Indian Pines | `Indian_pines_corrected.mat`, `Indian_pines_gt.mat` | [HybridSN 仓库镜像](https://github.com/gokriznastic/HybridSN/tree/master/data)（EHU 官网 404） |
| Pavia University | `PaviaU.mat`, `PaviaU_gt.mat` | 同上 |
| Houston 2013 | `Houston.mat`, `Houston_gt.mat` | [YangHuihan219/Houston2013](https://github.com/YangHuihan219/Houston2013)（Git LFS：`https://media.githubusercontent.com/media/YangHuihan219/Houston2013/main/Houston.mat`） |

## 复现主结果

```bash
# 完整 benchmark（5 方法 × 3 seeds × 2 数据集，随机协议）
sbatch bench.sh                      # 或直接 bash bench.sh 串行跑

# 单条训练
python seg_train_all.py --dataset indian_pines --model disentangle --epochs 100 --seed 42
python seg_train_all.py --dataset pavia --model disentangle --epochs 100 --seed 42 --use_cma

# 空间/棋盘协议
python seg_train_all.py --dataset indian_pines --model cnn3d --split spatial
python seg_train_all.py --dataset indian_pines --model cnn3d --split checker

# 汇总
python summarise.py --logs logs       # 主表（随机协议）
python summarise.py --logs logs --gap # 三协议对比
```

## 主要结果（3 seeds，mIoU mean±std）

### 随机协议（主 benchmark）
| 方法 | Indian Pines | Pavia | Houston 2013 |
|---|---|---|---|
| 3D-CNN | 65.09±5.82 | 96.54±0.55 | 97.27±0.41 |
| ViT | 65.68±4.19 | 97.26±0.63 | 96.11±0.41 |
| SpectralFormer | 61.32±4.15 | 97.10±1.43 | 97.01±0.65 |
| DisentangleFormer | 83.13±5.54 | 99.38±0.30 | **98.29±0.72** |
| Ours (Gated CMA) | **83.83±6.04** | **99.43±0.20** | 98.00±1.16 |

### 划分协议对比（泄漏 vs 诚实）
| | random | spatial 左右切 | checker 棋盘 |
|---|---|---|---|
| Indian Pines (Ours+CMA) | 83.83 | 1.33 | 21.12 |
| Pavia (Ours+CMA) | 99.43 | 41.92 | 88.89 |

关键结论：随机协议因重叠 tile 泄漏系统性抬高 mIoU 46–82 点；在无泄漏的 checker 协议下排名反转（Pavia: 3D-CNN 90.85 第一）。左右切在 Indian Pines 上结构性退化（测试区 3 类训练时从未出现）；棋盘协议通过按数据集扫描格大小/偏移保证 0 test-only 类。

## 重要实现细节

- **窗口注意力**：ST/CT 在 Swin 式 8×8 局部窗口内做注意力（原论文 N=M×M 指窗口内 token 数）。整图 flatten 的旧实现参数量 17M 严重过拟合（Indian mIoU 53），窗口化后 0.32M、mIoU 83。
- **CMA 是负结果**：10-seed 检验 mIoU +0.04（p=0.96）；γ 门控未被网络关闭但浅层方向随 seed 翻转；CCA 显示两流本来就不解耦（corr 0.75–0.88，地板 0.15），CMA 不改变两流关系。
- checkpoint 命名：`seg_{dataset}_{model}_r{ratio}_s{seed}[_sp|_ch]_best.pth`

## 引用

DisentangleFormer: Liao, Liò, de Kamps, Sarikaya — *Hyperspectral Image Classification via Spatial-Channel Disentangled Representation Learning*, arXiv:2512.04314.
