#!/bin/bash
#SBATCH --job-name=hsi_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out

module load miniforge/24.7.1
conda activate hsi

cd ~/hsi_project

# ── Baseline (原版DisentangleFormer) ──
python train.py --dataset indian_pines --epochs 100

# ── Ours (加CMA跨流注意力) ──
python train.py --dataset indian_pines --epochs 100 --use_cma
