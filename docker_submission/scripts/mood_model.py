from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import torchio as tio
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import os
import sys

sys.path.append('/workspace/ddpm_ood')
sys.path.append('/workspace/GenerativeModels')

sys.path.append('docker_example/scripts')
sys.path.append('docker_example/scripts/ddpm_ood')
sys.path.append('docker_example/scripts/GenerativeModels')

sys.path.append('ddpm_ood')
sys.path.append('GenerativeModels')

from src.trainers.base import BaseTrainer
from src.utils.simplex_noise import generate_simplex_noise
from generative.networks.schedulers import PNDMScheduler
from post_processing.utils import segment, ssim_pad
from data import upsample

# # TODO : remove plot functions for docker
# from plots import create_qualitative_results, create_histogram_plot_paper

class ReconstructMoodSubmission(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.config = args
        # set_determinism(seed=args.seed)
        if not self.found_checkpoint:
            raise FileNotFoundError("Failed to find a saved model checkpoint.")

        self.inference_start = args.inference_start

        self.easy_detection = args.easy_detection
        if self.easy_detection:
            self.n_bins = 4096
            self.all_hist = np.load(os.path.join(args.hist_dir, f'all_hist_{args.task}{self.n_bins}.npy'))
            assert self.all_hist.shape==self.all_hist.shape

        self.pndm_scheduler = PNDMScheduler(num_train_timesteps=1000,
                                            skip_prk_steps=True,
                                            prediction_type=self.prediction_type,
                                            schedule=self.beta_schedule,
                                            beta_start=self.beta_start,
                                            beta_end=self.beta_end,
                                            )

        
    def easy_localization(self, batch, n_std_outlier=4):
        image = batch["image"][tio.DATA].float()
        if self.config.task == 'brain':
            std_th = 64
        else:
            std_th = 128
        
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
            
            # Apply erosion and dilation on thesholded image
            structure_size = 6
            thresholded_image = binary_erosion(thresholded_image, np.ones((structure_size,structure_size,structure_size)))
            thresholded_image = binary_dilation(thresholded_image, np.ones((structure_size,structure_size,structure_size)))
            
            # # TODO : remove plot
            # create_histogram_plot_paper(bins,hist, hist_mean, hist_std, batch, image, thresholded_image, self.config)
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
        
        images_volume_original = batch["image"][tio.DATA].float()
        
        # normalize image
        transform = tio.RescaleIntensity(out_min_max=(0, 1))
        images_volume_original[0] = transform(images_volume_original[0])

        ranges = images_volume_original.shape[-1]
        all_reconstrcutions = np.zeros((images_volume_original.shape[-3], images_volume_original.shape[-2], images_volume_original.shape[-1], len(pndm_start_points) ))

        # noise images
        if self.simplex_noise:
            noise = generate_simplex_noise(
                self.simplex,
                x=images_volume_original[..., int(images_volume_original.shape[-1]/2)].to(self.device),
                t=torch.Tensor([pndm_start_points[0]] * images_volume_original.shape[0]).long(),
                in_channels=images_volume_original.shape[1],
            )
        else:
            noise = torch.randn_like(images_volume_original[..., 0].to(self.device)).to(self.device)


        for slices in tqdm(range(ranges)):
            images_original = images_volume_original[..., slices].to(self.device)
            
            if images_original.sum()> 0:
                
                images = self.vqvae_model.encode_stage_2_inputs(images_original)
                if self.do_latent_pad:
                    images = F.pad(input=images, pad=self.latent_pad, mode="constant", value=0)
                # loop over different values to reconstruct from
                for t_idx, t_start in enumerate(pndm_start_points):
                    with autocast(enabled=True):
                        start_timesteps = torch.Tensor([t_start] * images.shape[0]).long()

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
                    
                    reconstructions = self.vqvae_model.decode_stage_2_outputs(reconstructions)
                    reconstructions = reconstructions / self.b_scale
                    reconstructions.clamp_(0, 1)
                    
                    all_reconstrcutions[:,:, slices, t_idx] = reconstructions.squeeze().cpu().numpy()
                    
        return all_reconstrcutions
    

    
    def process_reconstruction(self, batch, recon):
        gt = batch["image"][tio.DATA].float().squeeze().cpu().numpy()
        result = recon[..., 0]
    
        # Calculate body mask
        mask = segment(gt, r=self.config.mask_radius)
                
        # Apply mask
        result = mask * result
       
        # Calculate SSIM map per voxel
        result = ssim_pad(ssim(gt, result, data_range=1, full=True)[1])
 
        # Blur SSIM map
        result = gaussian_filter(result, self.config.blur_sigma)
 
        # Apply mask to the SSIM map
        result = np.where(mask==0, 1, result)
 
        # Binarize prediction
        thresholded_image = np.where(result > self.config.ssim_threshold, 0., 1.)
        return thresholded_image.astype(np.uint8), result
    

    def score_one_case(self, batch):
        if self.easy_detection:
            ############################
            # easy detection
            ############################
            print('    Easy local started ...', batch['path'])
            easy_result = self.easy_localization(batch)
            print('    Easy local finished. sum=', easy_result.sum() )
             
            
            # # TODO : remove plot
            # create_qualitative_results(batch, None, None, easy_result, self.config)
            if easy_result.any():
                print('    Easy local found!')
                if self.config.task == 'brain':
                    print('returning easy brain result')
                    return easy_result
                else:
                    print('returning easy abdom result')
                    return upsample(easy_result)

        ############################
        # ddpm detection
        ############################
        ddpm_recon = self.ddpm_localization(batch)
        ddpm_result, ssim_map = self.process_reconstruction(batch, ddpm_recon)
        
        # #TODO : remove plot
        # create_qualitative_results(batch, ddpm_recon, ssim_map, ddpm_result, self.config)
        
        
        if self.config.task == 'brain':
            print('returning ddpm brain result')
            return ddpm_result
        else:
            print('returning ddpm abdom result')
            return upsample(ddpm_result)
