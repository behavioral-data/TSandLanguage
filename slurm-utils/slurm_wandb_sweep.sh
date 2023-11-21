#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --output=/mmfs1/gscratch/bdata/mikeam/SensingResearch/slurm-utils/jobs/%x-%j.log
#SBATCH --error=/mmfs1/gscratch/bdata/mikeam/SensingResearch/slurm-utils/jobs/%x-%j.out
#SBATCH --account=bdata
#SBATCH --partition=gpu-rtx6k
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --mem=45G
#SBATCH --cpus-per-gpu=4
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/gscratch/bdata/mikeam/SensingResearch
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate SensingResearch
wandb agent $1 --count 1 
