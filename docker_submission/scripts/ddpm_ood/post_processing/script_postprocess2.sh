#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=e.m.c.huijben@tue.nl
#SBATCH --partition=bme.gpuresearch.q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:0
#SBATCH --job-name=mood
#SBATCH --output=/home/bme001/s144823/output/other/output_%j.out

source /home/bme001/s144823/conda/etc/profile.d/conda.sh
conda activate mood

cd /home/bme001/shared/mood/code/ddpm-ood/post_processing
region="brain"


#### LDM model
methods="select,mask,ssim,blur,mask"


# ## TOY SET
phase="toy"
run="mood_ldm_07_25/nifti_vqvae/out/vqvae_10082023"
for select_recon in 0; do
for blur_sigma in 5 10 15; do
for mask_radius in 5 10; do
echo python analysis.py --run $run --phase $phase --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon --save_predictions False
srun python analysis.py --run $run --phase $phase --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon --save_predictions False
done
done
done


# TRANSFORMED VAL SET
suffix="_transformed"
phase="val"
run="mood_ldm_07_25/nifti_vqvae/val/vqvae_10082023"
for select_recon in 0; do
for blur_sigma in 5 10 15; do
for mask_radius in 5 10; do
echo python analysis.py --run $run --phase $phase --suffix $suffix --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon --save_predictions False
srun python analysis.py --run $run --phase $phase --suffix $suffix --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon --save_predictions False
done
done
done

### ORIGINAL VAL SET
phase="val"
run="mood_ldm_07_25/nifti_vqvae/in/vqvae_10082023"
for select_recon in 0; do
for blur_sigma in 5 10 15; do
for mask_radius in 5 10; do
echo python analysis.py --run $run --phase $phase --suffix "" --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon --save_predictions False
srun python analysis.py --run $run --phase $phase --suffix "" --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon --save_predictions False
done
done
done
