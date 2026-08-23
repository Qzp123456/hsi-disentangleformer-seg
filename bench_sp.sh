#!/bin/bash
#SBATCH --job-name=hsi_spatial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out

module load miniforge/24.7.1
conda activate hsi
cd ~/hsi_project

# Spatial-split protocol: left 60% train / right 40% test,
# straddling tiles dropped -> no overlap leakage across splits.
# 5 methods x 3 seeds x 2 datasets = 30 runs
for seed in 42 123 2024; do
  for ds in indian_pines pavia; do
    python seg_train_all.py --dataset $ds --model cnn3d          --epochs 100 --seed $seed --split spatial
    python seg_train_all.py --dataset $ds --model vit            --epochs 100 --seed $seed --split spatial
    python seg_train_all.py --dataset $ds --model spectralformer --epochs 100 --seed $seed --split spatial
    python seg_train_all.py --dataset $ds --model disentangle    --epochs 100 --seed $seed --split spatial
    python seg_train_all.py --dataset $ds --model disentangle    --epochs 100 --seed $seed --split spatial --use_cma
  done
done
