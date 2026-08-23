#!/bin/bash
#SBATCH --job-name=hsi_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out

module load miniforge/24.7.1
conda activate hsi
cd ~/hsi_project

# Full benchmark: 4 comparison methods + baseline + ours, 3 seeds, 2 datasets
for seed in 42 123 2024; do
  for ds in indian_pines pavia; do
    python seg_train_all.py --dataset $ds --model cnn3d          --epochs 100 --seed $seed
    python seg_train_all.py --dataset $ds --model vit            --epochs 100 --seed $seed
    python seg_train_all.py --dataset $ds --model spectralformer --epochs 100 --seed $seed
    python seg_train_all.py --dataset $ds --model disentangle    --epochs 100 --seed $seed
    python seg_train_all.py --dataset $ds --model disentangle    --epochs 100 --seed $seed --use_cma
  done
done
