#!/bin/bash                                                                                                                                                                                     
#SBATCH --partition=bme.gpuresearch.q      		# Partition name
#SBATCH --nodelist=bme-gpuB001
#SBATCH --nodes=1                        		# Use one node     how many machines we are using                                                                          
#SBATCH --time=200:00:00                  		        # Time limit hrs:min:sec
#SBATCH --output=./log/output_%A_Training.out         	# Standard output and error log
#SBATCH --gres=gpu:4
#SBATCH --job-name=mood_07_11_p


module load cuda11.8/toolkit

# python ./train_ddpm_mood.py  --parallel False --batch_size 4 --eval_freq 5 --checkpoint_every 5 --model_name 'mood_06_27' --is_grayscale 1 --n_epochs 100 --beta_schedule 'scaled_linear_beta'  --beta_start 0.0015 --beta_end 0.0195

# 
### experiments with simplex noise and guissian noise with a smaller beta range
# 
# torchrun --nproc_per_node 2 --nnodes 1  train_ddpm_mood.py --parallel --simplex_noise 1 --batch_size 4 --model_name 'mood_simplex_ddp_07_12' --eval_freq 10 --checkpoint_every 10 --is_grayscale 1 --n_epochs 100 --beta_schedule 'scaled_linear_beta'  --beta_start 0.001 --beta_end 0.015
# torchrun --nproc_per_node 2 --nnodes 1  train_ddpm_mood.py --parallel --batch_size 4 --model_name 'mood_beta001_015_ddp_07_12' --eval_freq 10 --checkpoint_every 10 --is_grayscale 1 --n_epochs 100 --beta_schedule 'scaled_linear_beta'  --beta_start 0.001 --beta_end 0.015

### training for pelvic data
# torchrun --nproc_per_node 2 --nnodes 1  train_ddpm_mood.py --parallel --simplex_noise 1 --data_dir '/home/bme001/shared/mood/data/abdom_train/abdom_train' --data_dir_val '/home/bme001/shared/mood/data/abdom_val/abdom_val' --batch_size 4 --model_name 'mood_simplex_ddp_abdom_07_26' --eval_freq 10 --checkpoint_every 10 --is_grayscale 1 --n_epochs 100 --beta_schedule 'scaled_linear_beta'  --beta_start 0.001 --beta_end 0.015
# python train_ddpm_mood.py --simplex_noise 1 --data_dir '/home/bme001/shared/mood/data/abdom_train/abdom_train' --data_dir_val '/home/bme001/shared/mood/data/abdom_val/abdom_val' --batch_size 4 --model_name 'mood_simplex_ddp_abdom_07_26' --eval_freq 10 --checkpoint_every 10 --is_grayscale 1 --n_epochs 100 --beta_schedule 'scaled_linear_beta'  --beta_start 0.001 --beta_end 0.015
# torchrun --nproc_per_node 2 --nnodes 1 train_ddpm_mood.py --simplex_noise 1 --parallel --data_dir '/home/bme001/shared/mood/data/abdom_train/abdom_train' --data_dir_val '/home/bme001/shared/mood/data/abdom_val/abdom_val' --batch_size 4 --model_name 'mood_simplex_ddp_abdom_07_27' --eval_freq 10 --checkpoint_every 10 --is_grayscale 1 --n_epochs 100 --beta_schedule 'scaled_linear_beta'  --beta_start 0.001 --beta_end 0.015


torchrun --nproc_per_node 4 --nnodes 1 train_ddpm_mood.py --simplex_noise 1 --parallel --data_dir '/home/bme001/shared/mood/abdom_train/abdom_train_resampled' --data_dir_val '/home/bme001/shared/mood/abdom_val/abdom_val_resampled' --batch_size 4 --model_name 'mood_simplex_ddp_abdom_08_23' --eval_freq 10 --checkpoint_every 10 --is_grayscale 1 --n_epochs 100 --beta_schedule 'scaled_linear_beta'  --beta_start 0.0015 --beta_end 0.0195
