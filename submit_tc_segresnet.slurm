#!/bin/bash
#SBATCH --job-name=tc_segresnet       	    # create a short name for your job
#SBATCH --output=%x-%j.log                	# Standard output in run directory, formatted: jobName-jobID.log
#SBATCH --error=%x-%j-E.log               	# Standard error log
#SBATCH --account=eecs545w23_class        	# account (check with $ my_accounts)
#SBATCH --partition=spgpu                     # (Submit to partition: standard, gpu. viz, largemem, oncampus, debug)    
#SBATCH --gres=gpu:1                      	# GPUs per node
#SBATCH --time=2-23:00:00                     # total run time limit (dd-hh:mm:ss)
#SBATCH --nodes=1                         	# node count
#SBATCH --tasks-per-node=1                	# total number of tasks across all nodes
#SBATCH --cpus-per-task=1                 	# cpu-cores per task (>1 if multi-threaded tasks) 
#SBATCH --mem=60gb                         	# total memory per node
#SBATCH --mail-type=END,FAIL              	# send mail if job fails
#SBATCH --export=ALL                      	# Copy environment

# Load modules
module purge
module load python/3.10.4 cuda/11.7.1 cudnn/11.7-v8.7.0

# Run the test
python3 train_two_class.py -m "./task_results" --epochs 1000 --cache_rate 1.0 --val_interval 5 --network SegResNet

