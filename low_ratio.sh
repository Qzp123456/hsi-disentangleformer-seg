#!/bin/bash
#SBATCH --job-name=hsi_lowratio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out

module load miniforge/24.7.1
conda activate hsi
cd ~/hsi_project

# Test CMA at low training ratios: 0.1 / 0.2 / 0.3, 8 seeds each
for ratio in 0.1 0.2 0.3; do
  for seed in 1 7 13 21 42 55 99 123; do
    for ds in indian_pines pavia; do
      python seg_train_all.py --dataset $ds --model disentangle --epochs 100 \
             --seed $seed --train_ratio $ratio
      python seg_train_all.py --dataset $ds --model disentangle --epochs 100 \
             --seed $seed --train_ratio $ratio --use_cma
    done
  done
done
