#!/bin/bash                                                                                                                                                                                     
#SBATCH --partition=bme.gpuresearch.q      		# Partition name
#SBATCH --nodelist=bme-gpuB001
#SBATCH --nodes=1                        		# Use one node                                                                                                                                             
#SBATCH --time=200:00:00                  		# Time limit hrs:min:sec
#SBATCH --output=./log/output_%A_Training.out         	# Standard output and error log
#SBATCH --gres=gpu:4                    		# Number of GPU(s) per node
#SBATCH --job-name=mood_06_26


module load cuda11.8/toolkit
###### training VQVAE
# 1- VQVAE
# python ./train_vqvae_mood.py --batch_size 4 --cache_data 0 --eval_freq 10 --checkpoint_every 10 --model_name 'mood_vqvae_07_21' --is_grayscale 1 --n_epochs 100 
# .
# .
# python ./train_vqvae_mood.py --batch_size 4 --cache_data 0 --eval_freq 10 --checkpoint_every 10 --model_name 'mood_vqvae_08_10' --is_grayscale 1 --n_epochs 200 

# torchrun --nproc_per_node 2 --nnodes 1 ./train_vqvae_mood.py --adversarial_warmup 1 --batch_size 32 --vqvae_num_embeddings 512 --parallel --cache_data 0 --eval_freq 10 --checkpoint_every 10 --model_name 'mood_vqvae_ddp_512_08_11' --is_grayscale 1 --n_epochs 200 

torchrun --nproc_per_node 4 --nnodes 1 ./train_vqvae_mood.py --batch_size 32 --vqvae_num_embeddings 1024 --parallel --cache_data 0 --eval_freq 10 --checkpoint_every 10 --model_name 'mood_vqvae_ddp_512_08_16' --is_grayscale 1 --n_epochs 200 
# python ./train_vqvae_mood.py --adversarial_warmup 1 --batch_size 32 --vqvae_num_embeddings 512 --parallel --cache_data 0 --eval_freq 10 --checkpoint_every 10 --model_name 'mood_vqvae_ddp_512_2_08_16' --is_grayscale 1 --n_epochs 200 

# python ./reconstruct_vqvae_mood.py --inference_name 'vqvae_10082023' --output_dir  './checkpoints' --vqvae_checkpoint 'checkpoints/mood_vqvae_07_21/checkpoint.pth' --save_nifti --model_name 'mood_ldm_07_25'  --batch_size 1 --is_grayscale 1 --run_val 0 --run_in 0 --run_out 1


# train vqvae on abdomen data
# torchrun --nproc_per_node 2 --nnodes 1 ./train_vqvae_mood.py --model_name 'mood_vqvae_ddp_abdom_08_11' --data_dir '/home/bme001/shared/mood/abdom_train/abdom_train_resampled' --data_dir_val '/home/bme001/shared/mood/abdom_val/abdom_val_resampled' --batch_size 12  --parallel --cache_data 0 --eval_freq 10 --checkpoint_every 10  --is_grayscale 1 --n_epochs 200 


# 2- test VQVAEd
# python ./reconstruct_vqvae_mood.py --inference_name 'vqvae_10082023' --output_dir  './checkpoints' --vqvae_checkpoint 'checkpoints/mood_vqvae_07_21/checkpoint.pth' --save_nifti --model_name 'mood_ldm_07_25'  --batch_size 1 --is_grayscale 1 --run_val 0 --run_in 0 --run_out 1


# python ./reconstruct_vqvae_mood.py --inference_name 'vqvae_130_1408' --output_dir  './checkpoints' --vqvae_checkpoint 'checkpoints/mood_vqvae_ddp_512_08_11/checkpoint.pth' --save_nifti --model_name 'vqvae'  --batch_size 1 --is_grayscale 1 --run_val 0 --run_in 0 --run_out 1


# python ./reconstruct_vqvae_mood.py --inference_name 'mood_vqvae_08_10' --output_dir  './checkpoints' --vqvae_checkpoint 'checkpoints/mood_vqvae_08_10/checkpoint.pth' --save_nifti --model_name 'vqvae'  --batch_size 1 --is_grayscale 1 --run_val 0 --run_in 0 --run_out 1

