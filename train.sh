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

# ── Baseline (original DisentangleFormer) ──
python train.py --dataset indian_pines --epochs 100

# ── Ours (with CMA cross-stream attention) ──
python train.py --dataset indian_pines --epochs 100 --use_cma
