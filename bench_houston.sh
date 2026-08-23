#!/bin/bash
#SBATCH --job-name=hsi_houston
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out

module load miniforge/24.7.1
conda activate hsi
cd ~/hsi_project

# Houston 2013: random-protocol benchmark, 5 methods x 3 seeds
for seed in 42 123 2024; do
  python seg_train_all.py --dataset houston --model cnn3d          --epochs 100 --seed $seed
  python seg_train_all.py --dataset houston --model vit            --epochs 100 --seed $seed
  python seg_train_all.py --dataset houston --model spectralformer --epochs 100 --seed $seed
  python seg_train_all.py --dataset houston --model disentangle    --epochs 100 --seed $seed
  python seg_train_all.py --dataset houston --model disentangle    --epochs 100 --seed $seed --use_cma
done
