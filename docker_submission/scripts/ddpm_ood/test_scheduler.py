# in this script I want to test different schedulers and their effect on the noisy input image


# some packages to use
# %%
import argparse
from src.data.get_mood_dataset import MoodDataModule 
import torch
import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
import time
# to make the argparse work for jupyter notebook
import sys
sys.argv = ['']

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default='/home/bme001/shared/mood/data/brain_train/brain_train', help="data directory.")
parser.add_argument("--out_data_dir", type=str, default='/home/bme001/shared/mood/data/brain_toy/toy', help="data directory.")
    ## options for mood
parser.add_argument("--data_dir_val", type=str, default='/home/bme001/shared/mood/data/brain_val/brain_val', help="data directory.")

parser.add_argument("--batch_size",default=4, help="batch size.")
parser.add_argument("--image_size",default=256, help="image size")

parser.add_argument("--patches_per_volume",default=1, help="nubmer of patches per volume for torchio dataloader")
parser.add_argument("--max_queue_length",default=16, help="maximum queue length for torchio dataloader")
parser.add_argument("--patch_size",default=(256, 256, 1), help="patch size for torchio dataloader, for 2D patches use (128, 128 ,1)")
parser.add_argument("--weightes_sampling",default=True, type=bool , help="wheighted sampling for torchio dataloader (could make the dataloader slow)")
parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

parser.add_argument('--parallel', action='store_true', help='training with multiple GPUs to adapt torchio dataloader')

args = parser.parse_args()

mooddata = MoodDataModule(args=args)
mooddata.setup()
train_loader = mooddata.train_dataloader()
print('size of dataset is {}'.format(len(train_loader)))


# 
# %%
device = 'cpu'
batch = next(iter(train_loader))
images = batch["image"][tio.DATA].float().squeeze(dim=-1).to(device)
print('the size of the input image is {}'.format(images.shape))
# %%
# build the scheduler
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler , PNDMScheduler
from src.utils.simplex_noise import generate_simplex_noise
from src.utils.simplex_noise import Simplex_CLASS
from generative.inferers import DiffusionInferer
prediction_type = 'epsilon'
beta_schedule = 'scaled_linear_beta'
# beta_start = 0.001
# beta_end = 0.015

beta_start = 0.0015
beta_end = 0.0195
# scheduler = DDPMScheduler(
#             num_train_timesteps=1000,
#             prediction_type=prediction_type,
#             schedule=beta_schedule,
#             beta_start=beta_start,
#             beta_end=beta_end,
#         )
scheduler = PNDMScheduler(
                    num_train_timesteps=1000,
                    skip_prk_steps=True,
                    prediction_type=prediction_type,
                    schedule=beta_schedule,
                    beta_start=beta_start,
                    beta_end=beta_end,
                )

simplex_noise = False
inferer = DiffusionInferer(scheduler)
timesteps = torch.randint(
                    0,
                    inferer.scheduler.num_train_timesteps,
                    (images.shape[0],),
                    device=device,
                ).long()
if simplex_noise:
    simplex = Simplex_CLASS()
    noise = generate_simplex_noise(
        simplex, x=images, t=timesteps, in_channels=images.shape[1]
    )
else:
    noise = torch.randn_like(images).to(device)
# %%
b_scale = 1
noisy_image = scheduler.add_noise(
                    original_samples=images * b_scale, noise=noise, timesteps=timesteps
                )
# %%
print(noisy_image.shape)

# %%
# based on this cell I should figure out how the reconstrcution works for backward process
scheduler.set_timesteps(50)
pndm_timesteps = scheduler.timesteps
# pndm_start_points = reversed(pndm_timesteps)[1::136]
inference_start = 200
pndm_start_points = torch.tensor([inference_start])

all_reconstrcutions = np.zeros((256,256, 256, len(pndm_start_points), int(inference_start/10) ))
print('size of the all_reconstcaacution {}'.format(all_reconstrcutions.shape))

print('pndm_start_points : {}'.format(pndm_start_points))
print('pndm_timesteps: {}'.format(pndm_timesteps))
for t_idx, t_start in enumerate(pndm_start_points):

    print('t_start: {}'.format(t_start))
    start_timesteps = torch.Tensor([t_start] * images.shape[0]).long()
    print('start_timesteps: {}'.format(start_timesteps))
    for step in pndm_timesteps[pndm_timesteps <= t_start]:
        timesteps = torch.Tensor([step] * images.shape[0]).long()
        print('timesteps: {}'.format(timesteps))
        print('step: {}'.format(step))
        if step%10 ==0:
            print('save slice : {}'.format(int(20-step/10)))
# %%
timesteps
# %%
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.utils import make_grid

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs, t=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(figsize= (16, 20), ncols=len(imgs), squeeze=False)

    for i, img in enumerate(imgs):
        img = img.squeeze().detach().numpy()
        axs[0, i].imshow(np.asarray(img), cmap = 'gray')
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if (t is not None):
            axs[0, i].set_title(' {}'.format(str(t[i])))




image_list = []
noisy_list = []
for i in range(images.shape[0]):
    image_list.append(images[i,...])
    noisy_list.append(noisy_image[i,...]) 

show(image_list)
show(noisy_list, t=timesteps.detach().numpy())
# %%
b_scale = 1
time_Steps = torch.tensor([5, 10, 20, 30, 50, 100, 200, 300, 400, 500, 600, 700, 900])
one_image = images[3,...]
noisy_list = []
for i in range(time_Steps.shape[0]):
    noisy_image = scheduler.add_noise(
                        original_samples=one_image * b_scale, noise=noise, timesteps=torch.tensor(time_Steps[i])
                    )
    noisy_list.append(noisy_image.squeeze())
# %%
noisy_list_a = [noisy_list[i][1,...].squeeze() for i in range(len(noisy_list))]
show(noisy_list_a, t=time_Steps.detach().numpy())
# %%
len(noisy_list_a)
# %%
