import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from generative.inferers import DiffusionInferer
from generative.networks.nets import VQVAE, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel

from src.networks import PassthroughVQVAE
from src.utils.simplex_noise import Simplex_CLASS


class BaseTrainer:
    def __init__(self, args):

        # initialise DDP if run was launched with torchrun
        if "LOCAL_RANK" in os.environ:
            print("Setting up DDP.")
            self.ddp = True
            # disable logging for processes except 0 on every node
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0:
                f = open(os.devnull, "w")
                sys.stdout = sys.stderr = f

            # initialize the distributed training process, every GPU runs in a process
            dist.init_process_group(backend="nccl")
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.ddp = False
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)

        print(f"Arguments: {str(args)}")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

        # set up model
        if args.vqvae_checkpoint:
            vqvae_checkpoint_path = Path(args.vqvae_checkpoint)
            vqvae_config_path = vqvae_checkpoint_path.parent / "vqvae_config.json"
            if not vqvae_checkpoint_path.exists():
                raise FileNotFoundError(f"Cannot find VQ-VAE checkpoint {vqvae_checkpoint_path}")
            if not vqvae_config_path.exists():
                raise FileNotFoundError(f"Cannot find VQ-VAE config {vqvae_config_path}")
            with open(vqvae_config_path, "r") as f:
                self.vqvae_config = json.load(f)
            self.vqvae_model = VQVAE(**self.vqvae_config)
            vqvae_checkpoint = torch.load(vqvae_checkpoint_path, map_location= self.device)
            self.vqvae_model.load_state_dict(vqvae_checkpoint["model_state_dict"])
            self.vqvae_model.to(self.device)
            self.vqvae_model.eval()
            vqvae_epoch = vqvae_checkpoint["epoch"]

            print(f"checkpoint {args.vqvae_checkpoint} at epoch {vqvae_epoch}")
            print("Loaded vqvae model with config:")
            for k, v in self.vqvae_config.items():
                print(f"  {k}: {v}")
            ddpm_channels = self.vqvae_config["embedding_dim"]
        else:
            self.vqvae_model = PassthroughVQVAE()
            ddpm_channels = 1 if args.is_grayscale else 3
        if args.model_type == "small":
            self.model = DiffusionModelUNet(
                spatial_dims=args.spatial_dimension,
                in_channels=ddpm_channels,
                out_channels=ddpm_channels,
                num_channels=(128, 256, 256),
                attention_levels=(False, False, True),
                num_res_blocks=1,
                num_head_channels=256,
                with_conditioning=False,
            ).to(self.device)
        elif args.model_type == "big":
            self.model = DiffusionModelUNet(
                spatial_dims=args.spatial_dimension,
                in_channels=ddpm_channels,
                out_channels=ddpm_channels,
                num_channels=(256, 512, 768),
                attention_levels=(True, True, True),
                num_res_blocks=2,
                num_head_channels=256,
                with_conditioning=False,
            ).to(self.device)
        else:
            raise ValueError(f"Do not recognise model type {args.model_type}")
        print(f"{sum(p.numel() for p in self.model.parameters()):,} model parameters")
        # set up noise scheduler parameters
        self.prediction_type = args.prediction_type
        self.beta_schedule = args.beta_schedule
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        self.b_scale = args.b_scale
        self.snr_shift = args.snr_shift
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            prediction_type=self.prediction_type,
            schedule=self.beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )
        if self.snr_shift != 1:
            print("Changing scheduler parameters to shift SNR")
            snr = self.scheduler.alphas_cumprod / (1 - self.scheduler.alphas_cumprod)
            target_snr = snr * self.snr_shift
            new_alphas_cumprod = 1 / (torch.pow(target_snr, -1) + 1)
            new_alphas = torch.zeros_like(new_alphas_cumprod)
            new_alphas[0] = new_alphas_cumprod[0]
            for i in range(1, len(new_alphas)):
                new_alphas[i] = new_alphas_cumprod[i] / new_alphas_cumprod[i - 1]
            new_betas = 1 - new_alphas
            self.scheduler.betas = new_betas
            self.scheduler.alphas = new_alphas
            self.scheduler.alphas_cumprod = new_alphas_cumprod

        self.simplex_noise = bool(args.simplex_noise)
        if self.simplex_noise:
            self.simplex = Simplex_CLASS()
        self.inferer = DiffusionInferer(self.scheduler)
        self.scaler = GradScaler()
        self.spatial_dimension = args.spatial_dimension
        self.image_size = int(args.image_size) if args.image_size else args.image_size
        if args.latent_pad:
            self.do_latent_pad = True
            self.latent_pad = args.latent_pad
            self.inverse_latent_pad = [-x for x in self.latent_pad]
        else:
            self.do_latent_pad = False

        # set up optimizer, loss, checkpoints
        self.run_dir = Path(args.output_dir) / args.model_name
        # can choose to resume/reconstruct from a specific checkpoint
        if args.ddpm_checkpoint_epoch:
            checkpoint_path = self.run_dir / f"checkpoint_{int(args.ddpm_checkpoint_epoch)}.pth"
        # otherwise select the best checkpoint
        else:
            checkpoint_path = self.run_dir / "checkpoint.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.found_checkpoint = True
            self.start_epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint["global_step"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.best_loss = checkpoint["best_loss"]
            print(
                f"Resuming training using checkpoint {checkpoint_path} at epoch {self.start_epoch}"
            )
        else:
            self.start_epoch = 0
            self.best_loss = 1000
            self.global_step = 0
            self.found_checkpoint = False

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=2.5e-5)
        if checkpoint_path.exists():
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # wrap the model with DistributedDataParallel module
        if self.ddp:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device], find_unused_parameters=True
            )

    def save_checkpoint(self, path, epoch, save_message=None):
        if self.ddp and dist.get_rank() == 0:
            # if DDP save a state dict that can be loaded by non-parallel models
            checkpoint = {
                "epoch": epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step": self.global_step,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)
        if not self.ddp:
            checkpoint = {
                "epoch": epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)
