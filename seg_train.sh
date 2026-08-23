#!/bin/bash
#SBATCH --job-name=hsi_seg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out

module load miniforge/24.7.1
conda activate hsi
cd ~/hsi_project

# 语义分割: baseline vs CMA, 多seed
for seed in 42 123 2024; do
    python seg_train.py --dataset indian_pines --epochs 100 --seed $seed
    python seg_train.py --dataset indian_pines --epochs 100 --seed $seed --use_cma
    python seg_train.py --dataset pavia --epochs 100 --seed $seed
    python seg_train.py --dataset pavia --epochs 100 --seed $seed --use_cma
done
