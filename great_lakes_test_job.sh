#!/bin/bash
#SBATCH --job-name train_simple_test
#SBATCH --nodes=1
#SBATCH --account=eecs545w23_class
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=200m
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --mail-type=NONE

python3 great_lakes_test.py