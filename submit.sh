#!/bin/bash
#SBATCH --mail-user=khanhnd@cs.uchicago.edu
#SBATCH --job-name=hybrid-ray-single
#SBATCH --account=mpcs51087
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=/scratch/midway3/khanhnd/Ezspeech/slurm/%j.hybrid-ray-v100-2.stdout
#SBATCH --error=/scratch/midway3/khanhnd/Ezspeech/slurm/%j.hybrid-ray-v100-2.stderr

source /scratch/midway3/khanhnd/miniconda3/bin/activate
conda activate ezspeech

 python scripts/train.py --config-name ctc_llm
