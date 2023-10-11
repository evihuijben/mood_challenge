import argparse

import json
import torchio as tio
import torch
import os
import time
from scipy.ndimage import binary_erosion, binary_dilation
import numpy as np

from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from torch.cuda.amp import autocast
from monai.utils import set_determinism
import pytorch_lightning as pl



import sys
# TODO : change paths for docker
sys.path.append('/home/ehuijben/ownCloud2/Code/models/202306_OOD/ddpm-mood')
sys.path.append('/home/ehuijben/ownCloud2/Code/models/202306_OOD/GenerativeModels')
from src.trainers.base import BaseTrainer
from src.utils.simplex_noise import generate_simplex_noise
from generative.networks.schedulers import PNDMScheduler

def show_image_mask(image_array, thresholded_image):
    import matplotlib.pyplot as plt
    slice_id=[np.argmax(thresholded_image.sum((0,1))), 1]
    
    # Display the original image and the segmented masks
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.concatenate((image_array[:,:, slice_id[0]],image_array[:,:, slice_id[1]])), cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(np.concatenate((thresholded_image[:,:, slice_id[0]],thresholded_image[:,:, slice_id[1]])), cmap='gray')
    axs[1].set_title('mask')
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()


class MoodDataModule(pl.LightningDataModule):
    # taken from main code 49d9c074a3ebe2670c49ab2b02593ea80f19206e

  def __init__(self, args):
    super().__init__()
    self.task = args.task
    self.input_dir = args.input_dir
    self.num_workers = args.num_workers
    self.batch_size = args.batch_size
    

  def test_dataloader(self):
    #TODO: resize if task == abdomen or abdom
    names = sorted(os.listdir(self.input_dir))
    file_names = [os.path.join(self.input_dir, name) for name in names]
    test_subjects = [tio.Subject(image=tio.ScalarImage(image_path), path=str(image_path)) for image_path in file_names]

    test_set = tio.SubjectsDataset(test_subjects, transform=None)
  
    data_loader = DataLoader(test_set, self.batch_size,shuffle=False, num_workers=self.num_workers)
    return data_loader
  


  

class ReconstructMoodSubmission(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        set_determinism(seed=args.seed)
        if not self.found_checkpoint:
            raise FileNotFoundError("Failed to find a saved model checkpoint.")

        self.inference_start = args.inference_start

        self.easy_detection = args.easy_detection
        if self.easy_detection:
            self.n_bins = 4096
            if args.task == 'brain':
                self.all_hist = np.load(os.path.join(args.hist_dir, f'all_hist_{self.n_bins}.npy'))
                assert self.all_hist.shape==self.all_hist.shape
            else:
                self.all_hist = np.load(os.path.join(args.hist_dir, f'all_hist_abdomen{self.n_bins}.npy'))
                assert self.all_hist.shape==self.all_hist.shape

        self.pndm_scheduler = PNDMScheduler(num_train_timesteps=1000,
                                            skip_prk_steps=True,
                                            prediction_type=self.prediction_type,
                                            schedule=self.beta_schedule,
                                            beta_start=self.beta_start,
                                            beta_end=self.beta_end,
                                            )


    
    def easy_localization(self, image, std_th = 16, n_std_outlier=4):
        # Calculate the mean and std of all training histograms and adjust the std to a predefined threshold
        hist_mean = np.mean(self.all_hist, axis=0, keepdims=True).squeeze()
        hist_std = np.std(self.all_hist, axis=0, keepdims=True).squeeze()
        hist_std[hist_std<std_th] = std_th
        
        # Prepare image data and calculate histogram for this case
        image_data = image.cpu().numpy().squeeze()
        image_flatened = image_data.flatten()
        hist, bins = np.histogram(image_flatened, bins=self.n_bins, range=(0, 1))

        # Check for outliers intensity values
        sub_hist = np.abs(hist - hist_mean) 
        sub_hist[sub_hist <= n_std_outlier * hist_std] = 0 
        spike_inds = sub_hist.nonzero()[0]
        spike_inds.sort()

        # Remove spike at zero (background)
        if len(spike_inds) > 0  and spike_inds[0] == 0:
            spike_inds = spike_inds[1:]
        
        # Threshold image based on outlier spikes
        thresholded_image = np.zeros(image_data.shape, dtype=np.uint8)
        if len(spike_inds)> 0:
            
            # Find connected outlier bins to speed up the process
            connected_groups = []
            current_group = [spike_inds[0]]
            for i in range(1,len(spike_inds)):
                if spike_inds[i] == spike_inds[i-1] +1:
                    current_group.append(spike_inds[i])
                else:
                    connected_groups.append(current_group)
                    current_group = [spike_inds[i]]
            connected_groups.append(current_group)
            
            # Threshold image based on this group's lower and upper threshold
            for group in connected_groups:
                first_ind = group[0]
                last_ind = group[-1]
                
                threshold_lower = bins[first_ind]
                threshold_upper = bins[last_ind+1]
                    
                thresholded_image += ((image_data > threshold_lower) & (image_data < threshold_upper)).astype(np.uint8)
            
            # Apply erosion on thesholded image
            thresholded_image = binary_erosion(thresholded_image, np.ones((3,3,3)))
        
            # Apply dilation on thesholded image
            thresholded_image = binary_dilation(thresholded_image, np.ones((3,3,3)))
            return thresholded_image.astype(np.uint8)
        else:
            return thresholded_image
        
    def ddpm_localization(self, batch):
        if self.snr_shift != 1:
            snr = self.pndm_scheduler.alphas_cumprod / (1 - self.pndm_scheduler.alphas_cumprod)
            target_snr = snr * self.snr_shift
            new_alphas_cumprod = 1 / (torch.pow(target_snr, -1) + 1)
            new_alphas = torch.zeros_like(new_alphas_cumprod)
            new_alphas[0] = new_alphas_cumprod[0]
            for i in range(1, len(new_alphas)):
                new_alphas[i] = new_alphas_cumprod[i] / new_alphas_cumprod[i - 1]
            new_betas = 1 - new_alphas
            self.pndm_scheduler.betas = new_betas
            self.pndm_scheduler.alphas = new_alphas
            self.pndm_scheduler.alphas_cumprod = new_alphas_cumprod
        self.pndm_scheduler.set_timesteps(100)

        pndm_timesteps = self.pndm_scheduler.timesteps
        pndm_start_points = torch.tensor([self.inference_start])
        
        #TODO : remove range
        images_volume_original = batch["image"][tio.DATA].float()

        ranges = images_volume_original.shape[-1]        
        all_reconstrcutions = np.zeros((images_volume_original.shape[-3], images_volume_original.shape[-2], images_volume_original.shape[-1], len(pndm_start_points) ))

        if not self.simplex_noise:
            noise = torch.randn_like(images_volume_original[..., 0].to(self.device)).to(self.device)

        for slices in range(ranges):
            images_original = images_volume_original[..., slices].to(self.device)
            images = self.vqvae_model.encode_stage_2_inputs(images_original)
            if self.do_latent_pad:
                images = F.pad(input=images, pad=self.latent_pad, mode="constant", value=0)
            # loop over different values to reconstruct from
            for t_idx, t_start in enumerate(pndm_start_points):
                with autocast(enabled=True):
                    start_timesteps = torch.Tensor([t_start] * images.shape[0]).long()

                    # noise images
                    if self.simplex_noise:
                        noise = generate_simplex_noise(
                            self.simplex,
                            x=images,
                            t=start_timesteps,
                            in_channels=images.shape[1],
                        )
                    reconstructions = self.pndm_scheduler.add_noise(
                        original_samples=images * self.b_scale,
                        noise=noise,
                        timesteps=start_timesteps,
                    )
                    # perform reconstruction
                    for step in pndm_timesteps[pndm_timesteps <= t_start]:
                        timesteps = torch.Tensor([step] * images.shape[0]).long()
                        model_output = self.model(
                            reconstructions, timesteps=timesteps.to(self.device)
                        )
                        # 2. compute previous image: x_t -> x_t-1
                        reconstructions, _ = self.pndm_scheduler.step(
                            model_output, step, reconstructions
                        )
                # try clamping the reconstructions
                if self.do_latent_pad:
                    reconstructions = F.pad(
                        input=reconstructions,
                        pad=self.inverse_latent_pad,
                        mode="constant",
                        value=0,
                    )
                # if not self.save_intermediate:
                reconstructions = self.vqvae_model.decode_stage_2_outputs(reconstructions)
                reconstructions = reconstructions / self.b_scale
                reconstructions.clamp_(0, 1)
                all_reconstrcutions[:,:, slices, t_idx] = reconstructions.squeeze().cpu().numpy()

        return all_reconstrcutions


    def score_one_case(self, batch):
        if self.easy_detection:
            ############################
            # easy detection
            ############################
            easy_result = self.easy_localization(batch["image"][tio.DATA].float())
            
            #TODO : remove plot
            original = batch['image'][tio.DATA].float().squeeze()
            show_image_mask(original, easy_result)
            
            if easy_result.any():
                return easy_result

        ############################
        # ddpm detection
        ############################
        ddpm_recon = self.ddpm_localization(batch)
        # TODO
        ddpm_result = np.where(ddpm_recon>0.5, 1, 0.)
        # ddpm_result = process_reconstruction(ddpm_recon)
        #TODO : remove plot
        show_image_mask(original, ddpm_result)
        

        
        return ddpm_result
        

    def get_scores(self, loader):
        mooddata = MoodDataModule(args=args)
        loader = mooddata.test_dataloader()
        
        print()
        print('Processing cases ...')
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                t_start = time.time()
                pred_local = self.score_one_case(batch)
                print(f'Case {i} took {time.time()-t_start:.2f} seconds.')
                
        print('Finished')
                
if __name__ == "__main__":
    # Load configuration from JSON file
    with open('../submission_config.json', 'r') as config_file:
        config = json.load(config_file)
    
    # Create an argparse.Namespace object from the configuration dictionary
    args = argparse.Namespace(**config)

    recon = ReconstructMoodSubmission(args)
    recon.get_scores(args)
