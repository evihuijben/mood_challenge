#!/bin/bash
#SBATCH --mail-type=END                  		              				# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=e.m.c.huijben@tue.nl  		              				# Where to send mail
#SBATCH --partition=bme.gpuresearch.q                               				# Partition name
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:0                                              			# Time limit hrs:min:sec
#SBATCH --job-name=mood
#SBATCH --output=/home/bme001/s144823/output/other/output_%j.out  		# Standard output and error log


source /home/bme001/s144823/conda/etc/profile.d/conda.sh
conda activate mood

cd /home/bme001/shared/mood/code/ddpm-ood/post_processing
region="brain"


# ## TOY SET
phase="toy"
#run="mood_simplex_ddp_07_12/nifti_modified/out/T100300600"

# methods="mean,mask,ssim,blur,mask"
# for blur_sigma in 5 10 15; do
# for mask_radius in 5 10; do
# echo python analysis.py --run $run --phase $phase --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma
# srun python analysis.py --run $run --phase $phase --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma
# done
# done

# methods="select,mask,ssim,blur,mask"
# for select_recon in 0 1; do
# for blur_sigma in 5 10 15; do
# for mask_radius in 5 10; do
# echo python analysis.py --run $run --phase $phase --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon
# srun python analysis.py --run $run --phase $phase --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon
# done
# done
# done

#select_recon=0
#methods="select,mask,ssim,blur,mask"
#for run in "mood_simplex_ddp_07_12/nifti_modified/out/T200" "mood_simplex_ddp_07_12/nifti_modified/out/T500"; do
#for blur_sigma in 5 10 15; do
#for mask_radius in 5 10; do
#echo python analysis.py --run $run --phase $phase --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon
#srun python analysis.py --run $run --phase $phase --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon
#done
#done
#done




### TRANSFORMED VAL SET
suffix="_transformed"
phase="val"
select_recon=0
methods="select,mask,ssim,blur,mask"
#
#for blur_sigma in 5 10 15; do
#for mask_radius in 5 10; do
#for run in "mood_simplex_ddp_07_12/nifti_modified/val/T100" "mood_simplex_ddp_07_12/nifti_modified/val/T200" "mood_simplex_ddp_07_12/nifti_modified/val/T500"; do
#srun python analysis.py --run $run --phase $phase --suffix $suffix --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon --save_predictions False
#done
#done
#done

#
#for blur_sigma in 5 10 15; do
#for mask_radius in 5 10; do
#for run in "mood_simplex_ddp_07_12/nifti_modified/val/T300"; do
#srun python analysis.py --run $run --phase $phase --suffix $suffix --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon --save_predictions False
#done
#done
#done



## ORIGINAL VAL SET
phase="val"
select_recon=0
methods="select,mask,ssim,blur,mask"

#for blur_sigma in 5 10 15; do
#for mask_radius in 5 10; do
#for run in "mood_simplex_ddp_07_12/nifti_modified/in/T100" "mood_simplex_ddp_07_12/nifti_modified/in/T200"; do
#srun python analysis.py --run $run --phase $phase --suffix "" --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon 
#done
#done
#done

run="mood_simplex_ddp_07_12/nifti_modified/in/T300"
#run="mood_simplex_ddp_07_12/nifti_modified/in/T500"
for blur_sigma in 5 10 15; do
for mask_radius in 5; do
echo python analysis.py --run $run --phase $phase --suffix "" --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon
srun python analysis.py --run $run --phase $phase --suffix "" --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon 
done
done

#
###### LDM model
#methods="select,mask,ssim,blur,mask"
#
#
### ## TOY SET
##phase="toy"
##run="mood_ldm_07_25/nifti_modified/out/T100200300500"
##for select_recon in 0 1 2 3; do
##for blur_sigma in 5 10 15; do
##for mask_radius in 5 10; do
##echo python analysis.py --run $run --phase $phase --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon
##srun python analysis.py --run $run --phase $phase --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon
##done
##done
##done
##
##
### TRANSFORMED VAL SET
#suffix="_transformed"
#phase="val"
#run="mood_ldm_07_25/nifti_modified/val/T100200300500"
##for select_recon in 0 1 2 3; do
##for blur_sigma in 5 10 15; do
##for mask_radius in 5 10; do
##echo python analysis.py --run $run --phase $phase --suffix $suffix --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon
##srun python analysis.py --run $run --phase $phase --suffix $suffix --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon
##done
##done
##done
##
##### ORIGINAL VAL SET
#phase="val"
#run="mood_ldm_07_25/nifti_modified/in/T100200300500"
##for select_recon in 0 1 2 3; do
##for blur_sigma in 5 10 15; do
##for mask_radius in 5 10; do
##echo python analysis.py --run $run --phase $phase --suffix "" --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon
##srun python analysis.py --run $run --phase $phase --suffix "" --combine_method $methods --mask_radius $mask_radius --blur_sigma $blur_sigma --select_recon $select_recon
##done
##done
##done
