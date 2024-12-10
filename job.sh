#!/bin/bash


#SBATCH --job-name=llama_eval
#SBATCH --output=x.out
#SBATCH --error=x.err
#SBATCH --account=aprakash0
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10GB
#SBATCH --cpus-per-task=4
#SBATCH --gpu_cmode=shared
#SBATCH --mail-type=BEGIN,END

source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh

conda activate llmeval

python eval.py
