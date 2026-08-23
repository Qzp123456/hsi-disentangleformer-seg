#!/bin/bash
#SBATCH --job-name=hsi_seeds10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out

module load miniforge/24.7.1
conda activate hsi
cd ~/hsi_project

for seed in 1 7 13 21 55 99 256 512 999 2048; do
  for ds in indian_pines pavia; do
    python seg_train_all.py --dataset $ds --model disentangle --epochs 100 --seed $seed
    python seg_train_all.py --dataset $ds --model disentangle --epochs 100 --seed $seed --use_cma
  done
done
