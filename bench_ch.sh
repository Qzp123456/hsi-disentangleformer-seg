#!/bin/bash
#SBATCH --job-name=hsi_checker
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out

module load miniforge/24.7.1
conda activate hsi
cd ~/hsi_project

# Checkerboard protocol: 64px cells alternate train/test (per-dataset grid
# offsets in seg_dataset.CHECKER_OFFSETS, chosen so no class is test-only),
# boundary tiles dropped -> no cross-split pixel overlap.
# 5 methods x 3 seeds x 2 datasets = 30 runs
for seed in 42 123 2024; do
  for ds in indian_pines pavia; do
    python seg_train_all.py --dataset $ds --model cnn3d          --epochs 100 --seed $seed --split checker
    python seg_train_all.py --dataset $ds --model vit            --epochs 100 --seed $seed --split checker
    python seg_train_all.py --dataset $ds --model spectralformer --epochs 100 --seed $seed --split checker
    python seg_train_all.py --dataset $ds --model disentangle    --epochs 100 --seed $seed --split checker
    python seg_train_all.py --dataset $ds --model disentangle    --epochs 100 --seed $seed --split checker --use_cma
  done
done
