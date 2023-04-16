#!/bin/bash
#SBATCH --job-name train_normal_unet
#SBATCH --nodes=1
#SBATCH --account=eecs545w23_class
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50g
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=NONE

module purge
module load python

source /home/joshmah/eecs545_med_image_seg/med_seg_env/bin/activate
python3 train_one_hot.py -d "/home/joshmah/eecs545_med_image_seg/data/task_data" -m "/home/joshmah/eecs545_med_image_seg/data/task_results" --cache_rate 1.0 --train_batch_size 64 --val_batch_size 1 --network "UNet"
deactivate