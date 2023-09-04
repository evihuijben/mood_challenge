import os
import tempfile
custom_tempdir = '/mnt/pred/tmp'
tempfile.tempdir = custom_tempdir
print('Using tmp dir', custom_tempdir)
if os.path.isdir('/mnt/pred'):

    print('listdir /mnt/pred', os.listdir('/mnt/pred'))
    for sub in os.listdir('/mnt/pred'):
        if os.path.isdir(sub):
            print(sub , 'ls:', os.listdir(os.path.join('/mnt/pred', sub)))
else:
    print('/mnt/pred not existing')
    


import nibabel as nib
import numpy as np
from pathlib import Path
import time
import torch

import sys
sys.path.append('/workspace')
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
            print('batch', i ,'loaded', batch['path'])
            t_start = time.time()
            if config.batch_size == 1:
                pred_local = mood_model.score_one_case(batch)
                
                print('score_one_case done')
                f = Path(batch['path'][0]).name
                
                if config.mode == "pixel":
                    target_file = os.path.join(config.result_dir, f)
                    affine = batch['image']['affine'][0]  # idx 0 in batch
                    final_nib_img = nib.Nifti1Image(pred_local, affine=affine)
                    print('Saving pixel result ...', target_file)
                    nib.save(final_nib_img, target_file)
                    print('Pixel saved to', target_file, os.listdir(config.result_dir))
                    
                elif config.mode == "sample":
                    if pred_local.any():
                        abnomal_score = 1.0
                        print('Saving sample result 1', os.path.join(config.result_dir, f + ".txt"))
                    else:
                        abnomal_score = 0.0
                        print('Saving sample result 0', os.path.join(config.result_dir, f + ".txt"))
                        
                    with open(os.path.join(config.result_dir, f + ".txt"), "w") as write_file:
                        write_file.write(str(abnomal_score))   
                    print('Sample saved to', os.path.join(config.result_dir, f + ".txt"), os.listdir(config.result_dir))  
                                     
                else:
                    print("Mode not correctly defined. Either choose 'pixel' oder 'sample'")

            else:
                raise NotImplementedError()
            print(f'Case {i} ({f}) took {time.time()-t_start:.2f} seconds. Sum of prediction = {pred_local.sum()}')
    print('Finished')

        
        
if __name__ == "__main__":

    print('loading config ...')
    config = load_config()
    print('config loaded')

    print('init model ...')
    mood_model = ReconstructMoodSubmission(config)
    print('model initialized')
    
    print('init data module ...' )
    mooddata = MoodDataModule(args=config)
    print('data module iniotialized')
    loader = mooddata.test_dataloader()
    print('dataloader created')

    print('start predict')
    predict(config, mood_model, loader)
    print('finished predict')
    
    if os.path.isdir('/mnt/pred'):
    
        print('listdir /mnt/pred', os.listdir('/mnt/pred'))
        for sub in os.listdir('/mnt/pred'):
            if os.path.isdir(sub):
                print(sub , 'ls:', os.listdir(os.path.join('/mnt/pred', sub)))
    else:
        print('/mnt/pred not existing')
        
        
    
    
    
    