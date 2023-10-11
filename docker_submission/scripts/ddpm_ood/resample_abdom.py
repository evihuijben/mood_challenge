import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
import torchio as tio
import torch

# data_dir = '/home/bme001/shared/mood/data/abdom_train/abdom_train'
# out_dir = '/home/bme001/shared/mood/abdom_train/abdom_train_resampled'
# data_dir = '/home/bme001/shared/mood/data/abdom_val/abdom_val'
# out_dir = '/home/bme001/shared/mood/abdom_val/abdom_val_resampled'
data_dir = '/home/bme001/shared/mood/data/abdom_toy/toy'
out_dir = '/home/bme001/shared/mood/abdom_toy/toy_resampled'
os.makedirs(out_dir, exist_ok=True)
#  write a function to read all the images in a data_dir using SimpleITK, resample them to (256, 256, 256) using tio and save them in out_dir
#  use the function to resample the images in data_dir and save them in out_dir

def resample_abdomen(data_dir, out_dir):
    list_of_files = sorted(os.listdir(data_dir))
    for file in tqdm(list_of_files):
        img_nib = nib.load(os.path.join(data_dir, file))
        img=torch.from_numpy(img_nib.get_fdata())[None]
        new_img = tio.Resize((256, 256, 256))(img)
        new_img = new_img.squeeze().numpy()
        header = img_nib.header
        header['dim'][1:4] = np.array([256, 256, 256])
        header['pixdim'][1:4] /= 2
        result_img = nib.Nifti1Image(new_img, img_nib.affine, header)
        nib.save(result_img, os.path.join(out_dir, file))

resample_abdomen(data_dir, out_dir)
