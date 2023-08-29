import os
import tempfile
custom_tempdir = '/mnt/pred/tmp'
tempfile.tempdir = custom_tempdir
    

import nibabel as nib
import numpy as np
from pathlib import Path
import time
import torch

import sys
sys.path.append('docker_example/scripts')

from data import MoodDataModule
from options import load_config
from mood_model import ReconstructMoodSubmission


def predict(config,  mood_model, loader):
    print()
    print('Processing cases ...')
    mood_model.model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            t_start = time.time()
            if config.batch_size == 1:
                pred_local = mood_model.score_one_case(batch)
                
                f = Path(batch['path'][0]).name
                if config.mode == "pixel":
                    target_file = os.path.join(config.result_dir, f)
                    affine = batch['image']['affine'][0]  # idx 0 in batch
                    final_nib_img = nib.Nifti1Image(pred_local, affine=affine)
                    nib.save(final_nib_img, target_file)
                elif config.mode == "sample":
                    if pred_local.any()>0:
                        abnomal_score = 1.0
                    else:
                        abnomal_score = 0.0
                    with open(os.path.join(config.result_dir, f + ".txt"), "w") as write_file:
                        write_file.write(str(abnomal_score))                
                else:
                    print("Mode not correctly defined. Either choose 'pixel' oder 'sample'")

            else:
                raise NotImplementedError()
            print(f'Case {i} ({f}) took {time.time()-t_start:.2f} seconds. Sum of prediction = {pred_local.sum()}')
    print('Finished')

        
        
if __name__ == "__main__":

    
    config = load_config()

    mood_model = ReconstructMoodSubmission(config)
    mooddata = MoodDataModule(args=config)
    loader = mooddata.test_dataloader()

    predict(config, mood_model, loader)
    
    
    