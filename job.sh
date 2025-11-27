#!/bin/bash
#SBATCH --job-name=ppo_train
#SBATCH --output=rl_log_%j.txt
#SBATCH --ntasks=8
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=standard
#SBATCH --account=b12901074


echo "Job started at $(date)"
/storage/undergrad/b12901015/miniconda3/envs/meep-env/bin/python train_ppo.py
echo "Job finished at $(date)"