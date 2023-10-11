#!/bin/bash                                                                                                                                                                                     
#SBATCH --partition=bme.gpuresearch.q      		# Partition name
#SBATCH --nodelist=bme-gpuB001
#SBATCH --nodes=1                        		# Use one node                                                                                                                                             
#SBATCH --time=100:00:00                  		# Time limit hrs:min:sec
#SBATCH --output=./log/output_%A_Training.out         	# Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --job-name=mood_06_26


module load cuda11.8/toolkit

python ./reconstruct_mood_submission.py  --save_nifti --easy_detection --model_name 'mood_simplex_ddp_07_12'  --run_val 0 --run_in 0 --run_out 1
