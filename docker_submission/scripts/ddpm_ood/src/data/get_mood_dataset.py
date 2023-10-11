

import torch
import torchvision
import torchvision.transforms as T
import torchio as tio
import pytorch_lightning as pl
import os
from torch.utils.data import DataLoader

# dataloader 

# Usage with distributed training
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

class MoodDataModule(pl.LightningDataModule):
    
  def __init__(self, args):
    super().__init__()
    # self.task = task
    self.out_data_dir = args.out_data_dir
    self.num_workers = args.num_workers
    self.batch_size = args.batch_size
    self.data_dir = args.data_dir 
    self.image_size = args.image_size
    self.patch_size = args.patch_size  # 2D slices
    self.max_queue_length = args.max_queue_length
    self.patches_per_volume = args.patches_per_volume
    self.weightes_sampling = args.weightes_sampling
    self.data_dir_val =  args.data_dir_val
    self.data_dir_val_in = args.data_dir_val_in
    # self.abdomen = args.is_abdomen
    # self.task = task
    # self.base_dir = root_dir
    # self.dataset_dir = os.path.join(root_dir, task)
    self.train_val_ratio = 0.9
    self.parallel = args.parallel

  def prepare_data(self):
    train_names = sorted(os.listdir(self.data_dir))
    train_subjects = [os.path.join(self.data_dir, name) for name in train_names]
    
    val_names = sorted(os.listdir(self.data_dir_val))
    val_subjects = [os.path.join(self.data_dir_val, name) for name in val_names]
    # num_subjects = len(file_names)
    # num_train_subjects = int(round(num_subjects * 0.9))
    # num_val_subjects = num_subjects - num_train_subjects
    # splits = num_train_subjects, num_val_subjects
    # train_subjects, val_subjects = torch.utils.data.random_split(file_names, splits)
    return train_subjects, val_subjects

  def get_preprocessing_transform(self):


    preprocess = tio.Compose([
      tio.RescaleIntensity(out_min_max=(0, 1)),
      # tio.CropOrPad(target_shape=(512, 256, 1))
      # tio.CropOrPad((self.image_size, self.image_size, 200), mask_name='msk'),
      # tio.Resample((1.25, 1.25, 9.8), image_interpolation ='bspline'),
      # tio.ToCanonical(),
      # tio.CropOrPad((self.image_size, self.image_size, 200)),
      
      # tio.ZNormalization(),
      # tio.RescaleIntensity((0, 1)),
      # tio.CropOrPad((self.image_size,self.image_size,self.get_max_shape(self.subjects))),
      # tio.Resample((self.get_mean_res(self.subjects), self.get_mean_res(self.subjects)), image_interpolation ='bspline'),
      # tio.EnsureShapeMultiple(8),  # for the U-Net
      # tio.OneHot(),
    ])
    return preprocess

  def get_augmentation_transform(self):
    augment = tio.Compose([
      # tio.RandomFlip(axes=(1), flip_probability=0.5)
        # tio.CropOrPad((self.image_size, self.image_size, 1), mask_name='msk'),
        
        # tio.RandomAffine(p=0.5),
        # tio.RandomGamma(p=0.5),
        # tio.RandomNoise(p=0.5),
        # tio.RandomMotion(p=0.1),
        # tio.RandomBiasField(p=0.25),
    ])
    return augment
  
  def setup(self):
    train_subjects_paths, val_subjects_paths = self.prepare_data()
    print(f'total number of train subjects {len(train_subjects_paths)}')
    train_subjects = [tio.Subject(image=tio.ScalarImage(image_path), path=str(image_path)) for image_path in train_subjects_paths]
    val_subjects = [tio.Subject(image=tio.ScalarImage(image_path), path=str(image_path)) for image_path in val_subjects_paths]
    # print(f'total number of slices trainig {self.get_total_images(train_subjects)}')
    # print(f'total number of slices validation {self.get_total_images(val_subjects)}')
    self.preprocess = self.get_preprocessing_transform()
    augment = self.get_augmentation_transform()
    self.transform = tio.Compose([self.preprocess, augment]) # add agument here for data augmentationd during trainig
    self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
    self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
    # self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

  def train_dataloader(self):

    if self.weightes_sampling:
      sampler = tio.WeightedSampler(self.patch_size, probability_map='image') # makes it really slow
    else:
      sampler = tio.UniformSampler(self.patch_size)
    
    if dist.is_initialized() or self.parallel:
      print('local rank is {}'.format(dist.get_rank()))
      print('world size is {}'.format(dist.get_world_size()))
      subject_sampler = DistributedSampler(self.train_set, shuffle=True, drop_last=True,)


      queue = tio.Queue(self.train_set, self.max_queue_length, self.patches_per_volume, sampler, num_workers=self.num_workers, subject_sampler=subject_sampler, shuffle_subjects=False, shuffle_patches=False)
      data_loader = DataLoader(queue, shuffle=False, batch_size= self.batch_size, num_workers=0, sampler=DistributedSampler(queue))  # this must be 0
    else:
      queue = tio.Queue(self.train_set, self.max_queue_length, self.patches_per_volume, sampler)
      data_loader = DataLoader(queue, self.batch_size,shuffle=True, num_workers=self.num_workers)
    return data_loader

  def val_dataloader(self):
    if self.weightes_sampling:
      sampler = tio.WeightedSampler(self.patch_size, probability_map='image') # makes it really slow
    else:
      sampler = tio.UniformSampler(self.patch_size)

    
    
        

    if dist.is_initialized() or self.parallel:
      print('local rank is {}'.format(dist.get_rank()))
      print('world size is {}'.format(dist.get_world_size()))

      # # Assume a process running on distributed node 3
      # local_rank = local_rank = int(os.environ["LOCAL_RANK"])
      # subject_sampler = DistributedSampler(
      # self.val_set,
      # rank=local_rank,
      # shuffle=True,
      # drop_last=True,
      # )
      # Each process is assigned (len(subjects_dataset) // num_processes) subjects

    
      queue = tio.Queue(self.val_set, self.max_queue_length, self.patches_per_volume, sampler,num_workers=self.num_workers)
      data_loader = DataLoader(queue, self.batch_size,shuffle=False,  num_workers=0, sampler=DistributedSampler(queue))  # this must be 0
    else:
      queue = tio.Queue(self.val_set, self.max_queue_length, self.patches_per_volume, sampler)
      data_loader = DataLoader(queue, self.batch_size,shuffle=False, num_workers=self.num_workers)
    return data_loader

  def test_dataloader(self):
    # return DataLoader(self.test_set, self.batch_size,shuffle=False, num_workers=self.num_workers)
    # self.preprocess = self.get_preprocessing_transform()
    self.preprocess = None
    names = sorted(os.listdir(self.out_data_dir))
    file_names = [os.path.join(self.out_data_dir, name) for name in names]
    test_subjects = [tio.Subject(image=tio.ScalarImage(image_path), path=str(image_path)) for image_path in file_names]

    test_set = tio.SubjectsDataset(test_subjects, transform=self.preprocess)
  
    data_loader = DataLoader(test_set, self.batch_size,shuffle=False, num_workers=self.num_workers)
    return data_loader
  
  def test_valid_dataloader(self):
    # this is only used for the final inference on all slices of the validation images
    self.preprocess = self.get_preprocessing_transform()
    val_names = sorted(os.listdir(self.data_dir_val))
    val_subjects = [os.path.join(self.data_dir_val, name) for name in val_names]
    test_subjects = [tio.Subject(image=tio.ScalarImage(image_path), path=str(image_path)) for image_path in val_subjects]

    test_set = tio.SubjectsDataset(test_subjects, transform=self.preprocess)
  
    data_loader = DataLoader(test_set, self.batch_size,shuffle=False, num_workers=self.num_workers)
    return data_loader
  
  def test_valid_in_dataloader(self):
    # this is only used for the final inference on all slices of the validation images
    self.preprocess = self.get_preprocessing_transform()
    val_names = sorted(os.listdir(self.data_dir_val_in))
    val_subjects = [os.path.join(self.data_dir_val_in, name) for name in val_names]
    test_subjects = [tio.Subject(image=tio.ScalarImage(image_path), path=str(image_path)) for image_path in val_subjects]

    test_set = tio.SubjectsDataset(test_subjects, transform=self.preprocess)
  
    data_loader = DataLoader(test_set, self.batch_size,shuffle=False, num_workers=self.num_workers)
    return data_loader
  

  