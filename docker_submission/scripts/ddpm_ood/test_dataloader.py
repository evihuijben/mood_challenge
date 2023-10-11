####### DESCRIPTION #######
# This is for testing the functionality of the dataloader for MOOD brain data and visualize a batch of training images
# it's an interactive python file with executable cells similar to jupyter notebook


# %%
import argparse
from src.data.get_mood_dataset import MoodDataModule
from src.data.get_mood_medtorch_dataset import MoodMedtorchDataModule
import torch
import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
# to make the argparse work for jupyter notebook
import sys
sys.argv = ['']

parser = argparse.ArgumentParser()

# parser.add_argument("--data_dir", type=str, default='/home/bme001/shared/mood/data/abdom_train/abdom_train', help="data directory.")
# parser.add_argument("--data_dir_val", type=str, default='/home/bme001/shared/mood/data/abdom_train/abdom_train', help="data directory.")
parser.add_argument("--data_dir", type=str, default='/home/bme001/shared/mood/data/brain_train/brain_train', help="data directory.")
parser.add_argument("--data_dir_val", type=str, default='/home/bme001/shared/mood/data/brain_val/brain_val', help="data directory.")
parser.add_argument("--data_dir_val_in", type=str, default='/home/bme001/shared/mood/data/brain_val/brain_val', help="data directory.")
parser.add_argument("--out_data_dir", type=str, default='/home/bme001/shared/mood/data/brain_toy/toy', help="data directory.")

parser.add_argument("--batch_size",default=32, help="batch size.")
parser.add_argument("--image_size",default=256, help="image size")

parser.add_argument("--patches_per_volume",default=128, help="nubmer of patches per volume for torchio dataloader")
parser.add_argument("--max_queue_length",default=4096, help="maximum queue length for torchio dataloader")
parser.add_argument("--patch_size",default=(256, 256, 1), help="patch size for torchio dataloader, for 2D patches use (128, 128 ,1)")
parser.add_argument("--weightes_sampling",default=True, type=bool , help="wheighted sampling for torchio dataloader (could make the dataloader slow)")
parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
parser.add_argument("--is_abdomen",default=False, type=bool , help="wheighted sampling for torchio dataloader (could make the dataloader slow)")

parser.add_argument('--parallel', action='store_true', help='training with multiple GPUs to adapt torchio dataloader')

args = parser.parse_args()
from src.data.get_mood_medtorch_dataset import create_dataloader_train
mooddata = MoodDataModule(args=args)
mooddata.setup()
# test_loader = create_dataloader_train(args)
# train_loader = mooddata.train_dataloader()
# print('size of dataset is {}'.format(len(train_loader)))
test_loader = mooddata.train_dataloader()
print('size of dataset is {}'.format(len(test_loader)))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)




# %%
def show_batch(batch):
    batch_size = batch.size(0)
    grid_size = int(np.ceil(np.sqrt(batch_size)))  # Calculate the grid size for subplots
    
    fig, axes = plt.subplots(2, grid_size, figsize=(16, 8))
    
    for i, ax in enumerate(axes.flat):
        if i < batch_size:
            img = batch[i].squeeze().numpy()  # Convert the tensor to a numpy array
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.show()


def check_nans(patches_batch):
    inputs = patches_batch['image'][tio.DATA]
    # inputs[0,0,0,0] = torch.nan
    nan_mask = torch.isnan(inputs)
    nan_indices = torch.nonzero(nan_mask)
    if nan_indices.shape[0] > 0:
        raise ValueError("NaN values found in the following indices:\n" + patches_batch["path"][0].split('/')[-1])


num_epochs = 1
model = torch.nn.Identity()
tic = time.time()
for epoch_index in range(num_epochs):
    for idx, patches_batch in enumerate(tqdm(test_loader)):
        if idx>10:
            break
        inputs = patches_batch['image'][tio.DATA].to(device)  # key 't1' is in subject
        ## check for nan values
        # check_nans(patches_batch)
        print(inputs.shape)
        logits = model(inputs)  # model being an instance of torch.nn.Module
        show_batch(inputs)
elapsed_time = time.time() - tic
print(f"Elapsed time: {elapsed_time} seconds")
# %%
import torch
import numpy as np
epoch_loss = torch.nan
nan_mask = torch.isnan(torch.tensor(epoch_loss))
nan_indices = torch.nonzero(nan_mask)
for epoch in range(100):
    if nan_indices.shape[0] > 0:
        RuntimeError('training stopped due to nan loss')
        print('training stopped due to nan loss')
        break
        
    print(epoch)


# %%

# the dataloader becomes really slow when loading the abdominal CT data of size 512 x 512 x 512 > downsample the data to 256 x 256 x 256 and then load them for training
import torch
x = torch.zeros(4, 4, requires_grad=True)
out = torch.linalg.matrix_norm(x)
out.backward()
print(x.grad)
# %%

import argparse
from src.data.get_mood_dataset import MoodDataModule 
import torch
import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
# to make the argparse work for jupyter notebook
import sys
sys.argv = ['']

parser = argparse.ArgumentParser()

# parser.add_argument("--data_dir", type=str, default='/home/bme001/shared/mood/data/abdom_train/abdom_train', help="data directory.")
# parser.add_argument("--data_dir_val", type=str, default='/home/bme001/shared/mood/data/abdom_train/abdom_train', help="data directory.")
parser.add_argument("--data_dir", type=str, default='/home/bme001/shared/mood/data/brain_train/brain_train', help="data directory.")
parser.add_argument("--data_dir_val", type=str, default='/home/bme001/shared/mood/data/brain_val/brain_val', help="data directory.")
parser.add_argument("--data_dir_val_in", type=str, default='/home/bme001/shared/mood/data/brain_val/brain_val', help="data directory.")
parser.add_argument("--out_data_dir", type=str, default='/home/bme001/shared/mood/data/brain_toy/toy', help="data directory.")

parser.add_argument("--batch_size",default=32, help="batch size.")
parser.add_argument("--image_size",default=256, help="image size")

parser.add_argument("--patches_per_volume",default=64, help="nubmer of patches per volume for torchio dataloader")
parser.add_argument("--max_queue_length",default=4096, help="maximum queue length for torchio dataloader")
parser.add_argument("--patch_size",default=(256, 256, 1), help="patch size for torchio dataloader, for 2D patches use (128, 128 ,1)")
parser.add_argument("--weightes_sampling",default=False, type=bool , help="wheighted sampling for torchio dataloader (could make the dataloader slow)")
parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
parser.add_argument("--is_abdomen",default=False, type=bool , help="wheighted sampling for torchio dataloader (could make the dataloader slow)")

parser.add_argument('--parallel', action='store_true', help='training with multiple GPUs to adapt torchio dataloader')

args = parser.parse_args()

from time import time
import multiprocessing as mp
for num_workers in range(0, mp.cpu_count(), 2):
    args.num_workers = num_workers  
    mooddata = MoodDataModule(args=args)
    mooddata.setup()
    train_loader = mooddata.train_dataloader()
    print('size of dataset is {}'.format(len(train_loader)))
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(tqdm(train_loader), 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
# %%
