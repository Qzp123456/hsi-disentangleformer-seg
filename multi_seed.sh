#!/bin/bash
#SBATCH --job-name=hsi_seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out

module load miniforge/24.7.1
conda activate hsi
cd ~/hsi_project

for seed in 42 123 2024; do
    python train.py --dataset indian_pines --epochs 100 --seed $seed
    python train.py --dataset indian_pines --epochs 100 --seed $seed --use_cma
done
