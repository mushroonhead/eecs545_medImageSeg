#!/bin/bash
#SBATCH --job-name train_UNET_variant
#SBATCH --nodes=1
#SBATCH --account=eecs545w23_class
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5g
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --mail-type=NONE

python3 train.py