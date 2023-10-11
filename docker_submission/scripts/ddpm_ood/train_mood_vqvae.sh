#!/bin/bash                                                                                                                                                                                     
#SBATCH --partition=bme.gpuresearch.q      		# Partition name
#SBATCH --nodelist=bme-gpuB001
#SBATCH --nodes=1                        		# Use one node                                                                                                                                             
#SBATCH --time=150:00:00                  		# Time limit hrs:min:sec
#SBATCH --output=./log/output_%A_Training.out         	# Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --job-name=mood_06_26


module load cuda11.8/toolkit
###### training LDM
# 1- VQVAE
# python ./train_vqvae_mood.py --batch_size 4 --cache_data 0 --eval_freq 10 --checkpoint_every 10 --model_name 'mood_vqvae_07_21' --is_grayscale 1 --n_epochs 100 
# .
# .
# 2- training LDM
# python train_ddpm_mood.py --batch_size 40  --output_dir  './checkpoints' --model_name 'mood_ldm_07_25' --vqvae_checkpoint 'checkpoints/mood_vqvae_07_21/checkpoint.pth' --is_grayscale 1 --n_epochs 200 --eval_freq 10 --checkpoint_every 10 --cache_data 0  --prediction_type 'epsilon' --model_type small --simplex_noise 1 --b_scale 1.0  --spatial_dimension 2  --beta_schedule 'scaled_linear_beta'  --beta_start 0.0015 --beta_end 0.0195

##### latent diffusion models inference

# python ./reconstruct_mood.py --modified --simplex_noise 1 --inference_name 'T200' --inference_start 200 --output_dir  './checkpoints' --vqvae_checkpoint 'checkpoints/mood_vqvae_07_21/checkpoint_100.pth' --save_nifti --model_name 'mood_ldm_07_25' --ddpm_checkpoint_epoch 80 --batch_size 1 --is_grayscale 1 --beta_schedule scaled_linear_beta --beta_start 0.0015 --beta_end 0.0195 --num_inference_steps 3 --inference_skip_factor 33 --run_val 1 --run_in 0 --run_out 0

# python ./reconstruct_mood.py --modified --simplex_noise 1 --inference_name 'T100200300500' --inference_start 100 --output_dir  './checkpoints' --vqvae_checkpoint 'checkpoints/mood_vqvae_07_21/checkpoint.pth' --save_nifti --model_name 'mood_ldm_07_25'  --batch_size 1 --is_grayscale 1 --beta_schedule scaled_linear_beta --beta_start 0.0015 --beta_end 0.0195 --num_inference_steps 3 --inference_skip_factor 33 --run_val 0 --run_in 0 --run_out 1
python ./reconstruct_mood.py --modified --simplex_noise 1 --inference_name 'T100200300500' --inference_start 100 --output_dir  './checkpoints' --vqvae_checkpoint 'checkpoints/mood_vqvae_07_21/checkpoint.pth' --save_nifti --model_name 'mood_ldm_07_25'  --batch_size 1 --is_grayscale 1 --beta_schedule scaled_linear_beta --beta_start 0.0015 --beta_end 0.0195 --num_inference_steps 3 --inference_skip_factor 33 --run_val 1 --run_in 0 --run_out 0


# python ./reconstruct_mood.py --output_dir  './checkpoints' --save_nifti --model_name 'mood_06_27' --ddpm_checkpoint_epoch 100 --batch_size 1 --is_grayscale 1 --beta_schedule scaled_linear_beta --beta_start 0.0015 --beta_end 0.0195 --num_inference_steps 3 --inference_skip_factor 33 --run_val 0 --run_in 0 --run_out 1
# --debugging
# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=1 python ./train_ddpm_mood.py --batch_size 8 --model_name 'mood_06_26' --is_grayscale 1 --n_epochs 100 --beta_schedule 'scaled_linear_beta'  --beta_start 0.0015 --beta_end 0.0195

# python ./train_ddpm_mood.py 