#!/bin/bash                                                                                                                                                                                     
#SBATCH --partition=bme.gpuresearch.q      		# Partition name
#SBATCH --nodelist=bme-gpuB001
#SBATCH --nodes=1                        		# Use one node                                                                                                                                             
#SBATCH --time=100:00:00                  		# Time limit hrs:min:sec
#SBATCH --output=./log/output_%A_Training.out         	# Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --job-name=mood_06_26


module load cuda11.8/toolkit

# python ./train_ddpm_mood.py  --batch_size 4 --eval_freq 5 --checkpoint_every 10 --model_name 'mood_ddpm_07_07' --is_grayscale 1 --n_epochs 100 --beta_schedule 'scaled_linear_beta'  --beta_start 0.0015 --beta_end 0.0195

# python ./reconstruct_mood.py --modified  --inference_start 200 --output_dir  './checkpoints' --save_nifti --model_name 'mood_ddpm_07_07' --ddpm_checkpoint_epoch 90 --batch_size 1 --is_grayscale 1 --beta_schedule scaled_linear_beta --beta_start 0.0015 --beta_end 0.0195 --num_inference_steps 3 --inference_skip_factor 33 --run_val 0 --run_in 0 --run_out 1
# python ./reconstruct_mood.py --modified --simplex_noise 1 --inference_name 'T100300600' --inference_start 200 --output_dir  './checkpoints' --save_nifti --model_name 'mood_simplex_ddp_07_12' --ddpm_checkpoint_epoch 60 --batch_size 1 --is_grayscale 1 --beta_schedule scaled_linear_beta --beta_start 0.0015 --beta_end 0.0195 --num_inference_steps 3 --inference_skip_factor 33 --run_val 0 --run_in 0 --run_out 1
# python ./reconstruct_mood.py --modified --simplex_noise 1 --inference_name 'T100' --inference_start 100 --output_dir  './checkpoints' --save_nifti --model_name 'mood_simplex_ddp_07_12' --ddpm_checkpoint_epoch 60 --batch_size 1 --is_grayscale 1 --beta_schedule scaled_linear_beta --beta_start 0.0015 --beta_end 0.0195 --num_inference_steps 3 --inference_skip_factor 33 --run_val 1 --run_in 0 --run_out 0

# python ./reconstruct_mood.py --modified --simplex_noise 1 --inference_name 'T300' --inference_start 300 --output_dir  './checkpoints' --save_nifti --model_name 'mood_simplex_ddp_07_12' --ddpm_checkpoint_epoch 60 --batch_size 1 --is_grayscale 1 --beta_schedule scaled_linear_beta --beta_start 0.0015 --beta_end 0.0195 --num_inference_steps 3 --inference_skip_factor 33 --run_val 1 --run_in 0 --run_out 0

# python ./reconstruct_mood.py --modified --simplex_noise 1 --inference_name 'T500' --inference_start 500 --output_dir  './checkpoints' --save_nifti --model_name 'mood_simplex_ddp_07_12' --ddpm_checkpoint_epoch 60 --batch_size 1 --is_grayscale 1 --beta_schedule scaled_linear_beta --beta_start 0.0015 --beta_end 0.0195 --num_inference_steps 3 --inference_skip_factor 33 --run_val 0 --run_in 1 --run_out 0

# python ./reconstruct_mood.py --modified  --inference_start 200 --output_dir  './checkpoints' --save_nifti --model_name 'mood_beta001_015_ddp_07_12' --ddpm_checkpoint_epoch 60 --batch_size 1 --is_grayscale 1 --beta_schedule scaled_linear_beta --beta_start 0.001 --beta_end 0.015 --num_inference_steps 3 --inference_skip_factor 33 --run_val 0 --run_in 0 --run_out 1

# python ./reconstruct_mood.py --modified  --inference_start 200 --output_dir  './checkpoints' --save_nifti --model_name 'mood_beta001_015_ddp_07_12' --ddpm_checkpoint_epoch 60 --batch_size 1 --is_grayscale 1 --beta_schedule scaled_linear_beta --beta_start 0.001 --beta_end 0.015 --num_inference_steps 3 --inference_skip_factor 33 --run_val 1 --run_in 0 --run_out 0
# --debugging
# --save_intermediate
# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=1 python ./train_ddpm_mood.py --batch_size 8 --model_name 'mood_06_26' --is_grayscale 1 --n_epochs 100 --beta_schedule 'scaled_linear_beta'  --beta_start 0.0015 --beta_end 0.0195

# python ./train_ddpm_mood.py 


python ./reconstruct_mood.py --modified --simplex_noise 1 --inference_name 'T200' --inference_start 200 --out_data_dir '/home/bme001/shared/mood/abdom_toy/toy_resampled' --output_dir  './checkpoints' --save_nifti --model_name 'mood_simplex_ddp_abdom_08_23' --batch_size 1 --is_grayscale 1 --beta_schedule scaled_linear_beta --beta_start 0.0015 --beta_end 0.0195 --num_inference_steps 3 --inference_skip_factor 33 --run_val 0 --run_in 0 --run_out 1
