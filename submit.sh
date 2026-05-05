#!/bin/bash
#SBATCH --mail-user=khanhnd@cs.uchicago.edu
#SBATCH --job-name=hybrid-ray-single
#SBATCH --account=mpcs52018
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --time=08:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=/scratch/midway3/khanhnd/Ezspeech/slurm/%j.hybrid-ray-a100-1.stdout
#SBATCH --error=/scratch/midway3/khanhnd/Ezspeech/slurm/%j.hybrid-ray-a100-1.stderr

source /scratch/midway3/khanhnd/miniconda3/bin/activate
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda activate ezspeech

python scripts/train.py
# 