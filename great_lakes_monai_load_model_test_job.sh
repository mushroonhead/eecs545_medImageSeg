#!/bin/bash
#SBATCH --job-name train_simple_test
#SBATCH --nodes=1
#SBATCH --account=eecs545w23_class
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=200m
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=NONE

module purge
module load python

source /home/joshmah/eecs545_med_image_seg/med_seg_env/bin/activate
python3 great_lakes_monai_load_model_test.py
deactivate