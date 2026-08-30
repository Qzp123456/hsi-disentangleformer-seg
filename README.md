# Hyperspectral Image Semantic Segmentation via Spatial-Channel Disentangled Representation Learning

MSc project code repository (University of Leeds, School of Computing).

This repository extends **DisentangleFormer** (Liao et al., arXiv 2512.04314), originally a per-pixel classifier, into an **end-to-end semantic segmentation framework**:

- Whole-image tiling into 32×32 patches with dense prediction and a SegDecoder head, evaluated with mIoU
- An additional gated cross-modality attention (CMA) bidirectional cross-stream fusion module (a negative result; see Results)
- Three train/test split protocols: `random` (overlapping-tile leakage) / `spatial` (left/right split) / `checker` (leakage-free checkerboard)

## Environment

```bash
module load miniforge/24.7.1     # Leeds Aire HPC; use any conda environment locally
conda create -n hsi python=3.10
conda activate hsi
pip install torch==2.5.1 numpy scipy scikit-learn matplotlib
```

GPU: NVIDIA L40S 48 GB (a single card suffices; the model has only ~0.3M parameters).

## File structure

| File | Purpose |
|---|---|
| `model.py` | DisentangleFormer core: window attention (Swin-style 8×8), ST/CT dual streams, STE, MS-FFN, CMAFusion (γ zero-initialised gating) |
| `seg_model.py` | Segmentation backbone + SegDecoder |
| `seg_dataset.py` | Whole-image tiling into 32×32 patches; three split protocols `random` / `spatial` (left 60% train / right 40% test) / `checker` (checkerboard, config in `CHECKER_CONFIG`); label 0 → ignore 255 |
| `dataset.py` | Dataset registry (Indian Pines / Pavia / Houston 2013) + patch utilities for classification |
| `seg_train_all.py` | Unified training entry point: `--model {disentangle,vit,spectralformer,cnn3d}` + `--use_cma` + `--split {random,spatial,checker}` + `--train_ratio` |
| `baselines.py` | ViTSeg / SpectralFormerSeg / CNN3DSeg baselines |
| `seg_train.py` / `train.py` | Earlier training scripts (kept for reference) |
| `summarise.py` | Aggregates logs → tables + LaTeX; `--spatial` / `--checker` for single-protocol tables, `--gap` for the three-protocol comparison |
| `visualise.py` | 5-panel qualitative comparison figures + difference maps (whole-image sliding-window inference) |
| `eval_perclass.py` | Per-class IoU for the spatial/checker protocols (`--tag {sp,ch}`) |
| `analyze_gamma.py` | Prints the gamma_s/gamma_c gating values of each CMAFusion layer (evidence for whether CMA is switched off by the network) |
| `analyze_cca.py` | Canonical correlation analysis of the ST/CT streams (with a shuffled null-hypothesis floor), to check whether CMA breaks disentanglement |
| `analyze_split_classes.py` | Diagnoses split protocols: per-class train/test pixel statistics + checkerboard offset scanning (`--scan`) |
| `bench.sh` | Main benchmark: 5 methods × 3 seeds × 2 datasets (random protocol) |
| `bench_sp.sh` | Spatial left/right split benchmark |
| `bench_ch.sh` | Checkerboard protocol benchmark |
| `bench_houston.sh` | Houston 2013 random-protocol benchmark |
| `low_ratio.sh` / `more_seeds.sh` | Low-supervision ablation / 10-seed statistical test |
| `*_job.sh` | Slurm job scripts (cca / perclass) |

## Data preparation

Place the files under `DATA_ROOT` (default `/mnt/scratch/znzs0468/data`):

| Dataset | Files | Source |
|---|---|---|
| Indian Pines | `Indian_pines_corrected.mat`, `Indian_pines_gt.mat` | [HybridSN repository mirror](https://github.com/gokriznastic/HybridSN/tree/master/data) (the EHU website returns 404) |
| Pavia University | `PaviaU.mat`, `PaviaU_gt.mat` | Same as above |
| Houston 2013 | `Houston.mat`, `Houston_gt.mat` | [YangHuihan219/Houston2013](https://github.com/YangHuihan219/Houston2013) (Git LFS: `https://media.githubusercontent.com/media/YangHuihan219/Houston2013/main/Houston.mat`) |

## Reproducing the main results

```bash
# Full benchmark (5 methods × 3 seeds × 2 datasets, random protocol)
sbatch bench.sh                      # or run `bash bench.sh` serially

# Single training run
python seg_train_all.py --dataset indian_pines --model disentangle --epochs 100 --seed 42
python seg_train_all.py --dataset pavia --model disentangle --epochs 100 --seed 42 --use_cma

# Spatial / checkerboard protocols
python seg_train_all.py --dataset indian_pines --model cnn3d --split spatial
python seg_train_all.py --dataset indian_pines --model cnn3d --split checker

# Summarise
python summarise.py --logs logs       # main table (random protocol)
python summarise.py --logs logs --gap # three-protocol comparison
```

## Main results (3 seeds, mIoU mean±std)

### Random protocol (main benchmark)

| Method | Indian Pines | Pavia | Houston 2013 |
|---|---|---|---|
| 3D-CNN | 65.09±5.82 | 96.54±0.55 | 97.27±0.41 |
| ViT | 65.68±4.19 | 97.26±0.63 | 96.11±0.41 |
| SpectralFormer | 61.32±4.15 | 97.10±1.43 | 97.01±0.65 |
| DisentangleFormer | 83.13±5.54 | 99.38±0.30 | **98.29±0.72** |
| Ours (Gated CMA) | **83.83±6.04** | **99.43±0.20** | 98.00±1.16 |

### Split-protocol comparison (leakage vs. honest)

| | random | spatial (left/right) | checker (checkerboard) |
|---|---|---|---|
| Indian Pines (Ours+CMA) | 83.83 | 1.33 | 21.12 |
| Pavia (Ours+CMA) | 99.43 | 41.92 | 88.89 |

Key findings: the random protocol systematically inflates mIoU by 46–82 points through overlapping-tile leakage; under the leakage-free checker protocol the ranking reverses (Pavia: 3D-CNN ranks first at 90.85). The left/right split degenerates structurally on Indian Pines (3 test-region classes never appear in the training region); the checkerboard protocol guarantees zero test-only classes by scanning grid size/offset per dataset.

## Important implementation details

- **Window attention**: ST/CT compute attention within Swin-style 8×8 local windows (in the original paper N=M×M denotes the number of tokens inside a window). An earlier whole-image flattening implementation had 17M parameters and severely overfit (Indian mIoU 53); after windowing it is 0.32M parameters with mIoU 83.
- **CMA is a negative result**: a 10-seed test gives mIoU +0.04 (p=0.96); the γ gate is not switched off by the network but its shallow-layer direction flips across seeds; CCA shows the two streams are not disentangled to begin with (corr 0.75–0.88 vs. a 0.15 floor), and CMA does not change the relationship between the two streams.
- Checkpoint naming: `seg_{dataset}_{model}_r{ratio}_s{seed}[_sp|_ch]_best.pth`

## Citation

DisentangleFormer: Liao, Liò, de Kamps, Sarikaya — *Hyperspectral Image Classification via Spatial-Channel Disentangled Representation Learning*, arXiv:2512.04314.
