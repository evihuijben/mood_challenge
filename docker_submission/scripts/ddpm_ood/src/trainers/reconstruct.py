# import matplotlib.pyplot as plt

import os
import sys
import time
from pathlib import Path
from scipy.ndimage import binary_erosion, binary_dilation
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from generative.networks.schedulers import PNDMScheduler
from torch.cuda.amp import autocast
from torch.nn.functional import pad

from src.data.get_train_and_val_dataloader import get_training_data_loader
from src.losses import PerceptualLoss
from src.utils.simplex_noise import generate_simplex_noise

from .base import BaseTrainer
from src.data.get_mood_dataset import MoodDataModule
import torchio as tio
import nibabel as nib
from tqdm import tqdm
def shuffle(x):
    return np.transpose(x.cpu().numpy(), (1, 2, 0))


class Reconstruct(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        if not self.found_checkpoint:
            raise FileNotFoundError("Failed to find a saved model checkpoint.")
        # set up dirs
        self.out_dir = self.run_dir / "ood"
        self.out_dir.mkdir(exist_ok=True)
        self.project = args.project

        # set up loaders
        if self.project == 'mood':
            mooddata = MoodDataModule(args=args)
            mooddata.setup()
            # self.train_loader = mooddata.train_dataloader()
            self.val_loader = mooddata.val_dataloader()
            self.in_loader = self.val_loader
        else:
            self.val_loader = get_training_data_loader(
                batch_size=args.batch_size,
                training_ids=args.validation_ids,
                validation_ids=args.validation_ids,
                augmentation=bool(args.augmentation),
                only_val=True,
                num_workers=args.num_workers,
                num_val_workers=args.num_workers,
                cache_data=bool(args.cache_data),
                drop_last=bool(args.drop_last),
                first_n=int(args.first_n_val) if args.first_n_val else args.first_n_val,
                is_grayscale=bool(args.is_grayscale),
                image_size=self.image_size,
                image_roi=args.image_roi,
            )

            self.in_loader = get_training_data_loader(
                batch_size=args.batch_size,
                training_ids=args.in_ids,
                validation_ids=args.in_ids,
                augmentation=bool(args.augmentation),
                only_val=True,
                num_workers=args.num_workers,
                num_val_workers=args.num_workers,
                cache_data=bool(args.cache_data),
                drop_last=bool(args.drop_last),
                first_n=int(args.first_n) if args.first_n else args.first_n,
                is_grayscale=bool(args.is_grayscale),
                image_size=self.image_size,
                image_roi=args.image_roi,
            )

    def get_scores(self, loader, dataset_name, inference_skip_factor):
        if dist.is_initialized():
            # temporarily enable logging on every node
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"{dist.get_rank()}: {dataset_name}")
        else:
            print(f"{dataset_name}")

        results = []
        pl = PerceptualLoss(
            dimensions=self.spatial_dimension,
            include_pixel_loss=False,
            is_fake_3d=True if self.spatial_dimension == 3 else False,
            lpips_normalize=True,
            spatial=False,
        ).to(self.device)
        # ms_ssim = MSSSIM(
        #     data_range=torch.tensor(1.0).to(self.device),
        #     spatial_dims=2,
        #     weights=torch.Tensor([0.0448, 0.2856]).to(self.device),
        # )

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                pndm_scheduler = PNDMScheduler(
                    num_train_timesteps=1000,
                    skip_prk_steps=True,
                    prediction_type=self.prediction_type,
                    schedule=self.beta_schedule,
                    beta_start=self.beta_start,
                    beta_end=self.beta_end,
                )
                if self.snr_shift != 1:
                    snr = pndm_scheduler.alphas_cumprod / (1 - pndm_scheduler.alphas_cumprod)
                    target_snr = snr * self.snr_shift
                    new_alphas_cumprod = 1 / (torch.pow(target_snr, -1) + 1)
                    new_alphas = torch.zeros_like(new_alphas_cumprod)
                    new_alphas[0] = new_alphas_cumprod[0]
                    for i in range(1, len(new_alphas)):
                        new_alphas[i] = new_alphas_cumprod[i] / new_alphas_cumprod[i - 1]
                    new_betas = 1 - new_alphas
                    pndm_scheduler.betas = new_betas
                    pndm_scheduler.alphas = new_alphas
                    pndm_scheduler.alphas_cumprod = new_alphas_cumprod
                pndm_scheduler.set_timesteps(100)
                pndm_timesteps = pndm_scheduler.timesteps
                pndm_start_points = reversed(pndm_timesteps)[1::inference_skip_factor]

                t1 = time.time()
                images_original = batch["image"].to(self.device)
                images = self.vqvae_model.encode_stage_2_inputs(images_original)
                if self.do_latent_pad:
                    images = F.pad(input=images, pad=self.latent_pad, mode="constant", value=0)
                # loop over different values to reconstruct from
                for t_start in pndm_start_points:
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
                        else:
                            noise = torch.randn_like(images).to(self.device)

                        reconstructions = pndm_scheduler.add_noise(
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
                            reconstructions, _ = pndm_scheduler.step(
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
                    # compute similarity
                    if self.spatial_dimension == 2:
                        if images_original.shape[3] == 28:
                            perceptual_difference = pl(
                                pad(images_original, (2, 2, 2, 2)),
                                pad(
                                    reconstructions,
                                    (2, 2, 2, 2),
                                ),
                            )
                        else:
                            perceptual_difference = pl(images_original, reconstructions)
                    else:
                        # in 3D need to calculate perceptual difference for each batch item seperately for now
                        perceptual_difference = torch.empty(images.shape[0])
                        for b in range(images.shape[0]):
                            perceptual_difference[b] = pl(
                                images_original[b, None, ...], reconstructions[b, None, ...]
                            )
                    non_batch_dims = tuple(range(images_original.dim()))[1:]
                    mse_metric = torch.square(images_original - reconstructions).mean(
                        axis=non_batch_dims
                    )
                    for b in range(images.shape[0]):
                        filename = batch["image_meta_dict"]["filename_or_obj"][b]
                        stem = Path(filename).stem.replace(".nii", "").replace(".gz", "")

                        results.append(
                            {
                                "filename": stem,
                                "type": dataset_name,
                                "t": t_start.item(),
                                "perceptual_difference": perceptual_difference[b].item(),
                                "mse": mse_metric[b].item(),
                            }
                        )
                    # plot
                    if not dist.is_initialized():
                        import matplotlib.pyplot as plt

                        n_rows = min(images.shape[0], 8)
                        fig, ax = plt.subplots(n_rows, 2, figsize=(2, n_rows))
                        for i in range(n_rows):
                            image_slice = (
                                np.s_[i, :, :, images_original.shape[4] // 2]
                                if self.spatial_dimension == 3
                                else np.s_[i, :, :]
                            )
                            plt.subplot(n_rows, 2, i * 2 + 1)
                            plt.imshow(
                                shuffle(images_original[image_slice]), vmin=0, vmax=1, cmap="gray"
                            )
                            plt.axis("off")
                            plt.subplot(n_rows, 2, i * 2 + 2)
                            plt.imshow(
                                shuffle(reconstructions[image_slice]), vmin=0, vmax=1, cmap="gray"
                            )
                            # plt.title(f"{mse_metric[i].item():.3f}")
                            plt.title(f"{perceptual_difference[i].item():.3f}")
                            plt.axis("off")
                        plt.suptitle(f"Recon from: {t_start}")
                        plt.tight_layout()
                        plt.show()
                t2 = time.time()
                if dist.is_initialized():
                    print(f"{dist.get_rank()}: Took {t2-t1}s for a batch size of {images.shape[0]}")
                else:
                    print(f"Took {t2-t1}s for a batch size of {images.shape[0]}")
        # gather results from all processes
        if dist.is_initialized():
            all_results = [None] * dist.get_world_size()
            dist.all_gather_object(all_results, results)
            # un-nest
            all_results = [item for sublist in all_results for item in sublist]
            # return to only logging on the first device
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0:
                f = open(os.devnull, "w")
                sys.stdout = sys.stderr = f
            return all_results
        else:
            return results

    def reconstruct(self, args):
        if bool(args.run_val):
            results_list = self.get_scores(self.val_loader, "val", args.inference_skip_factor)

            results_df = pd.DataFrame(results_list)
            results_df.to_csv(self.out_dir / "results_val.csv")

        if bool(args.run_in):
            results_list = self.get_scores(self.in_loader, "in", args.inference_skip_factor)

            results_df = pd.DataFrame(results_list)
            results_df.to_csv(self.out_dir / "results_in.csv")

        if bool(args.run_out):
            if self.project == 'mood':
                mooddata = MoodDataModule(args=args)
                out_loader = mooddata.test_dataloader()
            else:
                
                for out in args.out_ids.split(","):
                    print(out)
                    if "vflip" in out:
                        out = out.replace("_vflip", "")
                        out_loader = get_training_data_loader(
                            batch_size=args.batch_size,
                            training_ids=out,
                            validation_ids=out,
                            augmentation=bool(args.augmentation),
                            only_val=True,
                            num_workers=args.num_workers,
                            num_val_workers=args.num_workers,
                            cache_data=bool(args.cache_data),
                            drop_last=bool(args.drop_last),
                            first_n=int(args.first_n) if args.first_n else args.first_n,
                            is_grayscale=bool(args.is_grayscale),
                            image_size=self.image_size,
                            add_vflip=True,
                            image_roi=args.image_roi,
                        )
                        dataset_name = Path(out).stem.split("_")[0] + "_vflip"

                    elif "hflip" in out:
                        out = out.replace("_hflip", "")
                        out_loader = get_training_data_loader(
                            batch_size=args.batch_size,
                            training_ids=out,
                            validation_ids=out,
                            augmentation=bool(args.augmentation),
                            only_val=True,
                            num_workers=args.num_workers,
                            num_val_workers=args.num_workers,
                            cache_data=bool(args.cache_data),
                            drop_last=bool(args.drop_last),
                            first_n=int(args.first_n) if args.first_n else args.first_n,
                            is_grayscale=bool(args.is_grayscale),
                            image_size=self.image_size,
                            add_hflip=True,
                            image_roi=args.image_roi,
                        )
                        dataset_name = Path(out).stem.split("_")[0] + "_hflip"

                    else:
                        out_loader = get_training_data_loader(
                            batch_size=args.batch_size,
                            training_ids=out,
                            validation_ids=out,
                            augmentation=bool(args.augmentation),
                            only_val=True,
                            num_workers=args.num_workers,
                            num_val_workers=args.num_workers,
                            cache_data=bool(args.cache_data),
                            drop_last=bool(args.drop_last),
                            first_n=int(args.first_n) if args.first_n else args.first_n,
                            is_grayscale=bool(args.is_grayscale),
                            image_size=self.image_size,
                            image_roi=args.image_roi,
                        )
                        dataset_name = Path(out).stem.split("_")[0]
                results_list = self.get_scores(out_loader, "out", args.inference_skip_factor)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(self.out_dir / f"results_{dataset_name}.csv")




from monai.utils import set_determinism
class ReconstructMood(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        set_determinism(seed=args.seed)
        if not self.found_checkpoint:
            raise FileNotFoundError("Failed to find a saved model checkpoint.")
        # set up dirs
        self.out_dir = self.run_dir / "ood"
        self.out_dir.mkdir(exist_ok=True)

        self.nifti_out_dir = self.run_dir / "nifti"
        self.nifti_out_dir.mkdir(exist_ok=True)

        self.project = args.project
        self.debugging = args.debugging
        self.save_nifti = args.save_nifti

        # set up loaders
        if self.project == 'mood':
            mooddata = MoodDataModule(args=args)
            mooddata.setup()
            # self.train_loader = mooddata.train_dataloader()
            self.val_loader = mooddata.val_dataloader()
            self.in_loader = self.val_loader

    def get_scores(self, loader, dataset_name, inference_skip_factor):
        if dist.is_initialized():
            # temporarily enable logging on every node
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"{dist.get_rank()}: {dataset_name}")
        else:
            print(f"{dataset_name}")

        results = []
        pl = PerceptualLoss(
            dimensions=self.spatial_dimension,
            include_pixel_loss=False,
            is_fake_3d=True if self.spatial_dimension == 3 else False,
            lpips_normalize=True,
            spatial=False,
        ).to(self.device)
        # ms_ssim = MSSSIM(
        #     data_range=torch.tensor(1.0).to(self.device),
        #     spatial_dims=2,
        #     weights=torch.Tensor([0.0448, 0.2856]).to(self.device),
        # )

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                pndm_scheduler = PNDMScheduler(
                    num_train_timesteps=1000,
                    skip_prk_steps=True,
                    prediction_type=self.prediction_type,
                    schedule=self.beta_schedule,
                    beta_start=self.beta_start,
                    beta_end=self.beta_end,
                )
                if self.snr_shift != 1:
                    snr = pndm_scheduler.alphas_cumprod / (1 - pndm_scheduler.alphas_cumprod)
                    target_snr = snr * self.snr_shift
                    new_alphas_cumprod = 1 / (torch.pow(target_snr, -1) + 1)
                    new_alphas = torch.zeros_like(new_alphas_cumprod)
                    new_alphas[0] = new_alphas_cumprod[0]
                    for i in range(1, len(new_alphas)):
                        new_alphas[i] = new_alphas_cumprod[i] / new_alphas_cumprod[i - 1]
                    new_betas = 1 - new_alphas
                    pndm_scheduler.betas = new_betas
                    pndm_scheduler.alphas = new_alphas
                    pndm_scheduler.alphas_cumprod = new_alphas_cumprod
                pndm_scheduler.set_timesteps(100)
                pndm_timesteps = pndm_scheduler.timesteps
                pndm_start_points = reversed(pndm_timesteps)[1::inference_skip_factor]

                t1 = time.time()
                if self.project == 'mood':
                    images_volume_original = batch["image"][tio.DATA].float()
                else:
                    images_original = batch["image"].to(self.device)

                # for the test brain toy dataset we have 3D images of size 256x256x256 and need to iterate over the last dimension which is for slices
                if self.debugging:
                    ranges = 60
                else:
                    ranges = images_volume_original.shape[-1]
                # TODO: constrcus a numpy array to collect reconstructions and save as nifti files
                all_reconstrcutions = np.zeros((images_volume_original.shape[-3], images_volume_original.shape[-2], images_volume_original.shape[-1], len(pndm_start_points) ))
                for slices in tqdm(range(ranges)):
                    
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
                            else:
                                noise = torch.randn_like(images).to(self.device)

                            reconstructions = pndm_scheduler.add_noise(
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
                                reconstructions, _ = pndm_scheduler.step(
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

                        # TODO collect all reconstrutions belonging to a subject and save as nifti file
                        # print('reconstruction shape is {}'.format(reconstructions.shape))
                        # compute similarity
                        if self.spatial_dimension == 2:
                            if images_original.shape[3] == 28:
                                perceptual_difference = pl(
                                    pad(images_original, (2, 2, 2, 2)),
                                    pad(
                                        reconstructions,
                                        (2, 2, 2, 2),
                                    ),
                                )
                            else:
                                perceptual_difference = pl(images_original, reconstructions)
                        else:
                            # in 3D need to calculate perceptual difference for each batch item seperately for now
                            perceptual_difference = torch.empty(images.shape[0])
                            for b in range(images.shape[0]):
                                perceptual_difference[b] = pl(
                                    images_original[b, None, ...], reconstructions[b, None, ...]
                                )
                        non_batch_dims = tuple(range(images_original.dim()))[1:]
                        mse_metric = torch.square(images_original - reconstructions).mean(
                            axis=non_batch_dims
                        )
                        for b in range(images.shape[0]):
                            # filename = batch["image_meta_dict"]["filename_or_obj"][b]
                            filename = batch["path"][b]
                            stem = filename.split('/')[-1].replace(".nii", "").replace(".gz", "")
                            stem += str('_slice_' + str(slices))

                            results.append(
                                {
                                    "filename": stem,
                                    "type": dataset_name,
                                    "t": t_start.item(),
                                    "perceptual_difference": perceptual_difference[b].item(),
                                    "mse": mse_metric[b].item(),
                                }
                            )
                        # plot
                        if not dist.is_initialized():
                            import matplotlib.pyplot as plt

                            n_rows = min(images.shape[0], 8)
                            fig, ax = plt.subplots(n_rows, 2, figsize=(2, n_rows))
                            for i in range(n_rows):
                                image_slice = (
                                    np.s_[i, :, :, images_original.shape[4] // 2]
                                    if self.spatial_dimension == 3
                                    else np.s_[i, :, :]
                                )
                                plt.subplot(n_rows, 2, i * 2 + 1)
                                plt.imshow(
                                    shuffle(images_original[image_slice]), vmin=0, vmax=1, cmap="gray"
                                )
                                plt.axis("off")
                                plt.subplot(n_rows, 2, i * 2 + 2)
                                plt.imshow(
                                    shuffle(reconstructions[image_slice]), vmin=0, vmax=1, cmap="gray"
                                )
                                # plt.title(f"{mse_metric[i].item():.3f}")
                                plt.title(f"{perceptual_difference[i].item():.3f}")
                                plt.axis("off")
                            plt.suptitle(f"Recon from: {t_start}")
                            plt.tight_layout()
                            plt.show()
                    t2 = time.time()
                    if dist.is_initialized():
                        print(f"{dist.get_rank()}: Took {t2-t1}s for a batch size of {images.shape[0]}")
                    else:
                        print(f"Took {t2-t1}s for a batch size of {images.shape[0]}")
        
                if self.save_nifti:
                    filename = batch["path"][0]
                    stem = filename.split('/')[-1].replace(".nii", "").replace(".gz", "")
                    os.makedirs(os.path.join(self.nifti_out_dir, dataset_name), exist_ok=True)
                    out_nifti_name = os.path.join(self.nifti_out_dir, dataset_name, str(str(stem) + '_recon.nii.gz'))
                    affine = np.eye(4)
                    nifti_img = nib.Nifti1Image(all_reconstrcutions, affine)
                    nib.save(nifti_img, out_nifti_name)

        # gather results from all processes
        if dist.is_initialized():
            all_results = [None] * dist.get_world_size()
            dist.all_gather_object(all_results, results)
            # un-nest
            all_results = [item for sublist in all_results for item in sublist]
            # return to only logging on the first device
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0:
                f = open(os.devnull, "w")
                sys.stdout = sys.stderr = f
            return all_results
        else:
            return results

    def reconstruct(self, args):
        if bool(args.run_val):
            if self.project == 'mood':
                mooddata = MoodDataModule(args=args)
                valid_loader = mooddata.test_valid_dataloader()
                results_list = self.get_scores(valid_loader, "val", args.inference_skip_factor)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(self.out_dir / "results_val.csv")

        if bool(args.run_in):
            results_list = self.get_scores(self.in_loader, "in", args.inference_skip_factor)

            results_df = pd.DataFrame(results_list)
            results_df.to_csv(self.out_dir / "results_in.csv")

        if bool(args.run_out):
            if self.project == 'mood':
                mooddata = MoodDataModule(args=args)
                out_loader = mooddata.test_dataloader()
                dataset_name = 'brain_toy'

                results_list = self.get_scores(out_loader, "out", args.inference_skip_factor)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(self.out_dir / f"results_{dataset_name}.csv")




class ReconstructMoodModified(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        set_determinism(seed=args.seed)
        if not self.found_checkpoint:
            raise FileNotFoundError("Failed to find a saved model checkpoint.")
        # set up dirs
        self.out_dir = self.run_dir / "ood_modified"
        self.out_dir.mkdir(exist_ok=True)

        self.nifti_out_dir = self.run_dir / "nifti_modified"
        self.nifti_out_dir.mkdir(exist_ok=True)

        self.project = args.project
        self.debugging = args.debugging
        self.save_nifti = args.save_nifti
        self.save_intermediate = args.save_intermediate
        self.inference_start = args.inference_start
        self.inference_name = args.inference_name

        # set up loaders
        if self.project == 'mood':
            mooddata = MoodDataModule(args=args)
            mooddata.setup()
            # self.train_loader = mooddata.train_dataloader()
            self.val_loader = mooddata.val_dataloader()
            self.in_loader = self.val_loader

    def get_scores(self, loader, dataset_name, inference_skip_factor):
        if dist.is_initialized():
            # temporarily enable logging on every node
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"{dist.get_rank()}: {dataset_name}")
        else:
            print(f"{dataset_name}")

        results = []
        pl = PerceptualLoss(
            dimensions=self.spatial_dimension,
            include_pixel_loss=False,
            is_fake_3d=True if self.spatial_dimension == 3 else False,
            lpips_normalize=True,
            spatial=False,
        ).to(self.device)
        # ms_ssim = MSSSIM(
        #     data_range=torch.tensor(1.0).to(self.device),
        #     spatial_dims=2,
        #     weights=torch.Tensor([0.0448, 0.2856]).to(self.device),
        # )

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                pndm_scheduler = PNDMScheduler(
                    num_train_timesteps=1000,
                    skip_prk_steps=True,
                    prediction_type=self.prediction_type,
                    schedule=self.beta_schedule,
                    beta_start=self.beta_start,
                    beta_end=self.beta_end,
                )
                if self.snr_shift != 1:
                    snr = pndm_scheduler.alphas_cumprod / (1 - pndm_scheduler.alphas_cumprod)
                    target_snr = snr * self.snr_shift
                    new_alphas_cumprod = 1 / (torch.pow(target_snr, -1) + 1)
                    new_alphas = torch.zeros_like(new_alphas_cumprod)
                    new_alphas[0] = new_alphas_cumprod[0]
                    for i in range(1, len(new_alphas)):
                        new_alphas[i] = new_alphas_cumprod[i] / new_alphas_cumprod[i - 1]
                    new_betas = 1 - new_alphas
                    pndm_scheduler.betas = new_betas
                    pndm_scheduler.alphas = new_alphas
                    pndm_scheduler.alphas_cumprod = new_alphas_cumprod
                pndm_scheduler.set_timesteps(100)
                pndm_timesteps = pndm_scheduler.timesteps
                # pndm_start_points = reversed(pndm_timesteps)[1::inference_skip_factor]
                # inference_start = 200
                # pndm_start_points = torch.tensor([100, 200, 300, 500])
                pndm_start_points = torch.tensor([self.inference_start])
                t1 = time.time()
                if self.project == 'mood':
                    images_volume_original = batch["image"][tio.DATA].float()
                else:
                    images_original = batch["image"].to(self.device)

                # for the test brain toy dataset we have 3D images of size 256x256x256 and need to iterate over the last dimension which is for slices
                if self.debugging:
                    ranges = 60
                else:
                    ranges = images_volume_original.shape[-1]
                # TODO: constrcus a numpy array to collect reconstructions and save as nifti files
                
                if self.save_intermediate:
                    all_reconstrcutions = np.zeros((images_volume_original.shape[-3], images_volume_original.shape[-2], images_volume_original.shape[-1], len(pndm_start_points), int(1 + int(self.inference_start/10)) ))
                else:
                    all_reconstrcutions = np.zeros((images_volume_original.shape[-3], images_volume_original.shape[-2], images_volume_original.shape[-1], len(pndm_start_points) ))
                # to use the same noise for all images in one batch which belongs to one subject
                # noise images
                # if self.simplex_noise:
                #     noise = generate_simplex_noise(
                #         self.simplex,
                #         x=images_volume_original[..., int(images_volume_original.shape[-1]/2)].to(self.device),
                #         t=start_timesteps,
                #         in_channels=images.shape[1],
                #     )
                if not self.simplex_noise:
                    noise = torch.randn_like(images_volume_original[..., 0].to(self.device)).to(self.device)

                for slices in tqdm(range(ranges)):
                    
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
                            reconstructions = pndm_scheduler.add_noise(
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
                                reconstructions, _ = pndm_scheduler.step(
                                    model_output, step, reconstructions
                                )

                                # save intermediate reconstrcutions
                                if self.save_intermediate:
                                    if step%10 ==0:
                                        # reconstructions = self.vqvae_model.decode_stage_2_outputs(reconstructions)
                                        # reconstructions = reconstructions / self.b_scale
                                        # reconstructions.clamp_(0, 1)
                                        all_reconstrcutions[:,:, slices, t_idx, int(int(self.inference_start/10)-step/10) ] = (reconstructions/ self.b_scale).clamp_(0, 1).squeeze().cpu().numpy()


                        # try clamping the reconstructions
                        if self.do_latent_pad:
                            reconstructions = F.pad(
                                input=reconstructions,
                                pad=self.inverse_latent_pad,
                                mode="constant",
                                value=0,
                            )
                        if not self.save_intermediate:
                            reconstructions = self.vqvae_model.decode_stage_2_outputs(reconstructions)
                            reconstructions = reconstructions / self.b_scale
                            reconstructions.clamp_(0, 1)
                            all_reconstrcutions[:,:, slices, t_idx] = reconstructions.squeeze().cpu().numpy()

                        # TODO collect all reconstrutions belonging to a subject and save as nifti file
                        # print('reconstruction shape is {}'.format(reconstructions.shape))
                        # compute similarity
                        if self.spatial_dimension == 2:
                            if images_original.shape[3] == 28:
                                perceptual_difference = pl(
                                    pad(images_original, (2, 2, 2, 2)),
                                    pad(
                                        reconstructions,
                                        (2, 2, 2, 2),
                                    ),
                                )
                            else:
                                perceptual_difference = pl(images_original, reconstructions)
                        else:
                            # in 3D need to calculate perceptual difference for each batch item seperately for now
                            perceptual_difference = torch.empty(images.shape[0])
                            for b in range(images.shape[0]):
                                perceptual_difference[b] = pl(
                                    images_original[b, None, ...], reconstructions[b, None, ...]
                                )
                        non_batch_dims = tuple(range(images_original.dim()))[1:]
                        mse_metric = torch.square(images_original - reconstructions).mean(
                            axis=non_batch_dims
                        )
                        for b in range(images.shape[0]):
                            # filename = batch["image_meta_dict"]["filename_or_obj"][b]
                            filename = batch["path"][b]
                            stem = filename.split('/')[-1].replace(".nii", "").replace(".gz", "")
                            stem += str('_slice_' + str(slices))

                            results.append(
                                {
                                    "filename": stem,
                                    "type": dataset_name,
                                    "t": t_start.item(),
                                    "perceptual_difference": perceptual_difference[b].item(),
                                    "mse": mse_metric[b].item(),
                                }
                            )
                        # plot
                        # if not dist.is_initialized():
                        #     import matplotlib.pyplot as plt

                        #     n_rows = min(images.shape[0], 8)
                        #     fig, ax = plt.subplots(n_rows, 2, figsize=(2, n_rows))
                        #     for i in range(n_rows):
                        #         image_slice = (
                        #             np.s_[i, :, :, images_original.shape[4] // 2]
                        #             if self.spatial_dimension == 3
                        #             else np.s_[i, :, :]
                        #         )
                        #         plt.subplot(n_rows, 2, i * 2 + 1)
                        #         plt.imshow(
                        #             shuffle(images_original[image_slice]), vmin=0, vmax=1, cmap="gray"
                        #         )
                        #         plt.axis("off")
                        #         plt.subplot(n_rows, 2, i * 2 + 2)
                        #         plt.imshow(
                        #             shuffle(reconstructions[image_slice]), vmin=0, vmax=1, cmap="gray"
                        #         )
                        #         # plt.title(f"{mse_metric[i].item():.3f}")
                        #         plt.title(f"{perceptual_difference[i].item():.3f}")
                        #         plt.axis("off")
                        #     plt.suptitle(f"Recon from: {t_start}")
                        #     plt.tight_layout()
                        #     plt.show()
                    t2 = time.time()
                    if dist.is_initialized():
                        print(f"{dist.get_rank()}: Took {t2-t1}s for a batch size of {images.shape[0]}")
                    else:
                        print(f"Took {t2-t1}s for a batch size of {images.shape[0]}")
        
                if self.save_nifti:
                    filename = batch["path"][0]
                    stem = filename.split('/')[-1].replace(".nii", "").replace(".gz", "")
                    os.makedirs(os.path.join(self.nifti_out_dir, dataset_name, self.inference_name), exist_ok=True)
                    out_nifti_name = os.path.join(self.nifti_out_dir, dataset_name, self.inference_name, str(str(stem) + '_recon.nii.gz'))
                    affine = np.eye(4)
                    nifti_img = nib.Nifti1Image(all_reconstrcutions, affine)
                    nib.save(nifti_img, out_nifti_name)

        # gather results from all processes
        if dist.is_initialized():
            all_results = [None] * dist.get_world_size()
            dist.all_gather_object(all_results, results)
            # un-nest
            all_results = [item for sublist in all_results for item in sublist]
            # return to only logging on the first device
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0:
                f = open(os.devnull, "w")
                sys.stdout = sys.stderr = f
            return all_results
        else:
            return results

    def reconstruct(self, args):
        if bool(args.run_val):
            if self.project == 'mood':
                mooddata = MoodDataModule(args=args)
                valid_loader = mooddata.test_valid_dataloader()
                results_list = self.get_scores(valid_loader, "val", args.inference_skip_factor)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(self.out_dir / "results_val.csv")

        if bool(args.run_in):
            if self.project == 'mood':
                mooddata = MoodDataModule(args=args)
                valid_loader = mooddata.test_valid_in_dataloader()
                results_list = self.get_scores(valid_loader, "in", args.inference_skip_factor)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(self.out_dir / "results_in.csv")

        if bool(args.run_out):
            if self.project == 'mood':
                mooddata = MoodDataModule(args=args)
                out_loader = mooddata.test_dataloader()
                dataset_name = 'brain_toy'

                results_list = self.get_scores(out_loader, "out", args.inference_skip_factor)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(self.out_dir / f"results_{dataset_name}.csv")



class ReconstructMoodSubmission(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        set_determinism(seed=args.seed)
        if not self.found_checkpoint:
            raise FileNotFoundError("Failed to find a saved model checkpoint.")
        # set up dirs
        self.out_dir = self.run_dir / "ood_modified"
        self.out_dir.mkdir(exist_ok=True)

        self.nifti_out_dir = self.run_dir / "nifti_modified_submission"
        self.nifti_out_dir.mkdir(exist_ok=True)

        self.project = args.project
        self.debugging = args.debugging
        self.save_nifti = args.save_nifti
        self.save_intermediate = args.save_intermediate
        self.inference_start = args.inference_start
        self.inference_name = args.inference_name
        self.easy_detection = args.easy_detection

        ### loading histogram stats for training data out_dir = r'C:\Users\20180883\TUE\Desktop\MOOD\data'

        self.n_bins = 4096
        if args.task == 'brain':
            self.all_hist = np.load(os.path.join(args.hist_dir, f'all_hist_{self.n_bins}.npy'))
            self.all_bins = np.load(os.path.join(args.hist_dir, f'all_bins_{self.n_bins}.npy'))
            assert self.all_hist.shape==self.all_hist.shape
        else:
            self.all_hist = np.load(os.path.join(args.hist_dir, f'all_hist_abdomen{self.n_bins}.npy'))
            self.all_bins = np.load(os.path.join(args.hist_dir, f'all_bins_abdomen{self.n_bins}.npy'))
            assert self.all_hist.shape==self.all_hist.shape
    
    def easy_localization(self, image):
        hist_std = np.std(self.all_hist, axis=0, keepdims=True).squeeze()
        hist_std[hist_std<16] = 16
        hist_mean = np.mean(self.all_hist, axis=0, keepdims=True).squeeze()
        image_data = image.cpu().numpy().squeeze()
        image_flatened = image_data.flatten()
        hist, bins = np.histogram(image_flatened, bins=self.n_bins, range=(0, 1))
        sub_hist = np.abs(hist - hist_mean) 
        sub_hist[sub_hist <= 4* hist_std] = 0 
        
        spike_inds = sub_hist.nonzero()[0]
        spike_inds.sort()
        if spike_inds[0] == 0:
            spike_inds = spike_inds[1:]
        
        
        # Calculate the lower and upper thresholds
        thresholded_image = np.zeros(image_data.shape, dtype=np.uint8)
        if len(spike_inds)> 0:
            
            connected_groups = []
            current_group = [spike_inds[0]]
            
            for i in range(1,len(spike_inds)):
                if spike_inds[i] == spike_inds[i-1] +1:
                    current_group.append(spike_inds[i])
                        
                else:
                    connected_groups.append(current_group)
                    current_group = [spike_inds[i]]
                
            connected_groups.append(current_group)
                    
                
            for group in connected_groups:
                first_ind = group[0]
                last_ind = group[-1]
                
                threshold_lower = bins[first_ind]
                threshold_upper = bins[last_ind+1]
                    
                thresholded_image += ((image_data > threshold_lower) & (image_data < threshold_upper)).astype(np.uint8)
                    
            
        
            ## erosion and dilation on thresholded image
        
            # Apply erosion
            thresholded_image = binary_erosion(thresholded_image, np.ones((3,3,3)))
        
            # Apply dilation
            thresholded_image = binary_dilation(thresholded_image, np.ones((3,3,3)))
            thresholded_image = thresholded_image.astype(np.uint8)
            return thresholded_image
        else:
            return thresholded_image





    def get_scores(self, loader, dataset_name, inference_skip_factor):
        if dist.is_initialized():
            # temporarily enable logging on every node
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"{dist.get_rank()}: {dataset_name}")
        else:
            print(f"{dataset_name}")

        results = []
        pl = PerceptualLoss(
            dimensions=self.spatial_dimension,
            include_pixel_loss=False,
            is_fake_3d=True if self.spatial_dimension == 3 else False,
            lpips_normalize=True,
            spatial=False,
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):

                ############################
                # easy detection
                ############################
                if self.easy_detection:
                    easy_mask = self.easy_localization(batch["image"][tio.DATA].float())
                    print('size of the easy mask {}'.format(easy_mask.shape))
                    filename = batch["path"][0]
                    image_nib = nib.load(filename)
                    result_img = nib.Nifti1Image(easy_mask.astype(np.uint16), image_nib.affine, image_nib.header)
                    stem = filename.split('/')[-1].replace(".nii", "").replace(".gz", "")
                    os.makedirs(os.path.join(self.nifti_out_dir, dataset_name, self.inference_name), exist_ok=True)
                    out_nifti_name = os.path.join(self.nifti_out_dir, dataset_name, self.inference_name, str(str(stem) + '_easyMask.nii.gz'))
                    nib.save(result_img, out_nifti_name)



                ############################
                # ddpm detection
                ############################

                pndm_scheduler = PNDMScheduler(
                    num_train_timesteps=1000,
                    skip_prk_steps=True,
                    prediction_type=self.prediction_type,
                    schedule=self.beta_schedule,
                    beta_start=self.beta_start,
                    beta_end=self.beta_end,
                )
                if self.snr_shift != 1:
                    snr = pndm_scheduler.alphas_cumprod / (1 - pndm_scheduler.alphas_cumprod)
                    target_snr = snr * self.snr_shift
                    new_alphas_cumprod = 1 / (torch.pow(target_snr, -1) + 1)
                    new_alphas = torch.zeros_like(new_alphas_cumprod)
                    new_alphas[0] = new_alphas_cumprod[0]
                    for i in range(1, len(new_alphas)):
                        new_alphas[i] = new_alphas_cumprod[i] / new_alphas_cumprod[i - 1]
                    new_betas = 1 - new_alphas
                    pndm_scheduler.betas = new_betas
                    pndm_scheduler.alphas = new_alphas
                    pndm_scheduler.alphas_cumprod = new_alphas_cumprod
                pndm_scheduler.set_timesteps(100)
                pndm_timesteps = pndm_scheduler.timesteps
                # pndm_start_points = reversed(pndm_timesteps)[1::inference_skip_factor]
                # inference_start = 200
                # pndm_start_points = torch.tensor([100, 200, 300, 500])
                pndm_start_points = torch.tensor([self.inference_start])
                t1 = time.time()
                if self.project == 'mood':
                    images_volume_original = batch["image"][tio.DATA].float()
                else:
                    images_original = batch["image"].to(self.device)

                # for the test brain toy dataset we have 3D images of size 256x256x256 and need to iterate over the last dimension which is for slices
                if self.debugging:
                    ranges = 60
                else:
                    ranges = images_volume_original.shape[-1]
                
                if self.save_intermediate:
                    all_reconstrcutions = np.zeros((images_volume_original.shape[-3], images_volume_original.shape[-2], images_volume_original.shape[-1], len(pndm_start_points), int(1 + int(self.inference_start/10)) ))
                else:
                    all_reconstrcutions = np.zeros((images_volume_original.shape[-3], images_volume_original.shape[-2], images_volume_original.shape[-1], len(pndm_start_points) ))

                if not self.simplex_noise:
                    noise = torch.randn_like(images_volume_original[..., 0].to(self.device)).to(self.device)

                for slices in tqdm(range(ranges)):
            
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
                            reconstructions = pndm_scheduler.add_noise(
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
                                reconstructions, _ = pndm_scheduler.step(
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
                        if not self.save_intermediate:
                            reconstructions = self.vqvae_model.decode_stage_2_outputs(reconstructions)
                            reconstructions = reconstructions / self.b_scale
                            reconstructions.clamp_(0, 1)
                            all_reconstrcutions[:,:, slices, t_idx] = reconstructions.squeeze().cpu().numpy()


                        # print('reconstruction shape is {}'.format(reconstructions.shape))
                        # compute similarity
                        if self.spatial_dimension == 2:
                            if images_original.shape[3] == 28:
                                perceptual_difference = pl(
                                    pad(images_original, (2, 2, 2, 2)),
                                    pad(
                                        reconstructions,
                                        (2, 2, 2, 2),
                                    ),
                                )
                            else:
                                perceptual_difference = pl(images_original, reconstructions)
                        else:
                            # in 3D need to calculate perceptual difference for each batch item seperately for now
                            perceptual_difference = torch.empty(images.shape[0])
                            for b in range(images.shape[0]):
                                perceptual_difference[b] = pl(
                                    images_original[b, None, ...], reconstructions[b, None, ...]
                                )
                        non_batch_dims = tuple(range(images_original.dim()))[1:]
                        mse_metric = torch.square(images_original - reconstructions).mean(
                            axis=non_batch_dims
                        )
                        for b in range(images.shape[0]):
                            # filename = batch["image_meta_dict"]["filename_or_obj"][b]
                            filename = batch["path"][b]
                            stem = filename.split('/')[-1].replace(".nii", "").replace(".gz", "")
                            stem += str('_slice_' + str(slices))

                            results.append(
                                {
                                    "filename": stem,
                                    "type": dataset_name,
                                    "t": t_start.item(),
                                    "perceptual_difference": perceptual_difference[b].item(),
                                    "mse": mse_metric[b].item(),
                                }
                            )
                    t2 = time.time()
                    if dist.is_initialized():
                        print(f"{dist.get_rank()}: Took {t2-t1}s for a batch size of {images.shape[0]}")
                    else:
                        print(f"Took {t2-t1}s for a batch size of {images.shape[0]}")
                
                ############################
                # use the all_reconstrcutions to calculate the metrics and save as nifti
                ############################
                if self.save_nifti:
                    filename = batch["path"][0]
                    stem = filename.split('/')[-1].replace(".nii", "").replace(".gz", "")
                    os.makedirs(os.path.join(self.nifti_out_dir, dataset_name, self.inference_name), exist_ok=True)
                    out_nifti_name = os.path.join(self.nifti_out_dir, dataset_name, self.inference_name, str(str(stem) + '_recon.nii.gz'))
                    affine = np.eye(4)
                    nifti_img = nib.Nifti1Image(all_reconstrcutions, affine)
                    nib.save(nifti_img, out_nifti_name)

        # gather results from all processes
        if dist.is_initialized():
            all_results = [None] * dist.get_world_size()
            dist.all_gather_object(all_results, results)
            # un-nest
            all_results = [item for sublist in all_results for item in sublist]
            # return to only logging on the first device
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0:
                f = open(os.devnull, "w")
                sys.stdout = sys.stderr = f
            return all_results
        else:
            return results

    def reconstruct(self, args):
        if bool(args.run_out):
            if self.project == 'mood':
                mooddata = MoodDataModule(args=args)
                out_loader = mooddata.test_dataloader()
                dataset_name = 'brain_toy'

                results_list = self.get_scores(out_loader, "out", args.inference_skip_factor)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(self.out_dir / f"results_{dataset_name}.csv")


class ReconstructMoodVQVAE(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        set_determinism(seed=args.seed)
        # if not self.found_checkpoint:
        #     raise FileNotFoundError("Failed to find a saved model checkpoint.")
        if args.vqvae_checkpoint is None:
            raise FileNotFoundError("Failed to find a saved model checkpoint.")
        # set up dirs
        self.run_dir = os.path.dirname(args.vqvae_checkpoint)
        
        self.out_dir = os.path.join(self.run_dir , "ood_vqvae")
        os.makedirs(self.out_dir, exist_ok=True)

        self.nifti_out_dir = os.path.join(self.run_dir ,"nifti_vqvae")
        os.makedirs(self.nifti_out_dir, exist_ok=True)

        self.project = args.project
        self.debugging = args.debugging
        self.save_nifti = args.save_nifti
        self.save_intermediate = args.save_intermediate
        self.inference_start = args.inference_start
        self.inference_name = args.inference_name

        # set up loaders
        if self.project == 'mood':
            mooddata = MoodDataModule(args=args)
            mooddata.setup()
            # self.train_loader = mooddata.train_dataloader()
            self.val_loader = mooddata.val_dataloader()
            self.in_loader = self.val_loader

    def get_scores(self, loader, dataset_name, inference_skip_factor):
        if dist.is_initialized():
            # temporarily enable logging on every node
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"{dist.get_rank()}: {dataset_name}")
        else:
            print(f"{dataset_name}")

        results = []
        pl = PerceptualLoss(
            dimensions=self.spatial_dimension,
            include_pixel_loss=False,
            is_fake_3d=True if self.spatial_dimension == 3 else False,
            lpips_normalize=True,
            spatial=False,
        ).to(self.device)
        # ms_ssim = MSSSIM(
        #     data_range=torch.tensor(1.0).to(self.device),
        #     spatial_dims=2,
        #     weights=torch.Tensor([0.0448, 0.2856]).to(self.device),
        # )

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                t1 = time.time()
                if self.project == 'mood':
                    images_volume_original = batch["image"][tio.DATA].float()
                else:
                    images_original = batch["image"].to(self.device)

                # for the test brain toy dataset we have 3D images of size 256x256x256 and need to iterate over the last dimension which is for slices
                if self.debugging:
                    ranges = 60
                else:
                    ranges = images_volume_original.shape[-1]
                # TODO: constrcus a numpy array to collect reconstructions and save as nifti files
                all_reconstrcutions = np.zeros((images_volume_original.shape[-3], images_volume_original.shape[-2], images_volume_original.shape[-1]))
                
                for slices in tqdm(range(ranges)):
                    images_original = images_volume_original[..., slices].to(self.device)
                    ### encode and decode for VQVAE
                    z = self.vqvae_model.encoder(images_original)
                    e, _ = self.vqvae_model.quantize(z)
                    reconstructions = self.vqvae_model.decode(e)
                    reconstructions = reconstructions / self.b_scale
                    reconstructions.clamp_(0, 1)
                    all_reconstrcutions[:,:, slices] = reconstructions.squeeze().cpu().numpy()
                    t2 = time.time()
                    if dist.is_initialized():
                        print(f"{dist.get_rank()}: Took {t2-t1}s for a batch size of {images_original.shape[0]}")
                    else:
                        print(f"Took {t2-t1}s for a batch size of {images_original.shape[0]}")
        
                if self.save_nifti:
                    filename = batch["path"][0]
                    stem = filename.split('/')[-1].replace(".nii", "").replace(".gz", "")
                    os.makedirs(os.path.join(self.nifti_out_dir, dataset_name, self.inference_name), exist_ok=True)
                    out_nifti_name = os.path.join(self.nifti_out_dir, dataset_name, self.inference_name, str(str(stem) + '_recon.nii.gz'))
                    affine = np.eye(4)
                    nifti_img = nib.Nifti1Image(all_reconstrcutions, affine)
                    nib.save(nifti_img, out_nifti_name)

        # gather results from all processes
        if dist.is_initialized():
            all_results = [None] * dist.get_world_size()
            dist.all_gather_object(all_results, results)
            # un-nest
            all_results = [item for sublist in all_results for item in sublist]
            # return to only logging on the first device
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0:
                f = open(os.devnull, "w")
                sys.stdout = sys.stderr = f
            return all_results
        else:
            return results

    def reconstruct(self, args):
        if bool(args.run_val):
            if self.project == 'mood':
                mooddata = MoodDataModule(args=args)
                valid_loader = mooddata.test_valid_dataloader()
                results_list = self.get_scores(valid_loader, "val", args.inference_skip_factor)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(self.out_dir / "results_val.csv")

        if bool(args.run_in):
            if self.project == 'mood':
                mooddata = MoodDataModule(args=args)
                valid_loader = mooddata.test_valid_in_dataloader()
                results_list = self.get_scores(valid_loader, "in", args.inference_skip_factor)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(self.out_dir / "results_in.csv")

        if bool(args.run_out):
            if self.project == 'mood':
                mooddata = MoodDataModule(args=args)
                out_loader = mooddata.test_dataloader()
                dataset_name = 'brain_toy'

                results_list = self.get_scores(out_loader, "out", args.inference_skip_factor)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(os.path.join(self.out_dir / f"results_{dataset_name}.csv"))