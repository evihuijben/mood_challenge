#!/bin/bash
#SBATCH --mail-type=END                  		              				# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=e.m.c.huijben@tue.nl  		              				# Where to send mail
#SBATCH --partition=bme.gpuresearch.q                               				# Partition name
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:0                                              			# Time limit hrs:min:sec
#SBATCH --job-name=mood
#SBATCH --output=/home/bme001/s144823/output/other/output_%j.out  		# Standard output and error log
#SBATCH --nodelist=bme-gpuB001


source /home/bme001/s144823/conda/etc/profile.d/conda.sh
conda activate mood

cd /home/bme001/shared/mood/code/ddpm-ood/post_processing

srun python create_gt_hull_objects.py

