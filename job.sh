#!/bin/bash
#SBATCH --job-name=ppo_train_20251128_005703
#SBATCH --output=rl_log_%j.txt
#SBATCH --ntasks=16
#SBATCH --time=23:59:00
#SBATCH --mem=64G
#SBATCH --partition=standard
#SBATCH --account=b12901015


echo "Job started at $(date)"
/storage/undergrad/b12901015/miniconda3/envs/meep-env/bin/python train_ppo.py
echo "Job finished at $(date)"