#!/bin/bash
#SBATCH --job-name=rl_hello_world
#SBATCH --output=hello_world_%j.txt
#SBATCH --ntasks=4
#SBATCH --time=00:01:00
#SBATCH --mem=8G
#SBATCH --partition=standard
#SBATCH --account=b12901074


echo "Job started at $(date)"
/storage/undergrad/b12901074/miniconda3/envs/rl_final/bin/python hello.py
echo "Job finished at $(date)"