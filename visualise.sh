#!/bin/bash
#SBATCH --job-name=hsi_vis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%j.out

module load miniforge/24.7.1
conda activate hsi
cd ~/hsi_project

python visualise.py --dataset indian_pines --seed 42
python visualise.py --dataset pavia --seed 42
