

import torch
import torchvision
import torchvision.transforms as T
import torchio as tio
import pytorch_lightning as pl
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import nibabel as nib
# dataloader 
from tqdm import tqdm
# Usage with distributed training
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import torchvision.transforms.functional as F

class MoodMedtorchDataModule(Dataset):
    
  def __init__(self, args):
    super(MoodMedtorchDataModule, self).__init__()
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
    self.abdomen = args.is_abdomen
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

    if self.abdomen:
      preprocess = tio.Compose([
        tio.Resize((256, 256, 512), image_interpolation='bspline'),
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
    else:
      preprocess = tio.Compose([
        ToTensor(),
        # tio.RescaleIntensity(out_min_max=(0, 1)),
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
    self.filename_pairs_train =[]
    train_subjects_paths, val_subjects_paths = self.prepare_data()
    for i in range(len(train_subjects_paths)):
      self.filename_pairs_train += [(train_subjects_paths[i], train_subjects_paths[i] )]
    # self.filename_pairs_train = [(train_subjects_paths, train_subjects_paths)]
    self.filename_pairs_val = [(val_subjects_paths, val_subjects_paths)]
    
    
    # print(f'total number of slices trainig {self.get_total_images(train_subjects)}')
    # print(f'total number of slices validation {self.get_total_images(val_subjects)}')
    self.preprocess = self.get_preprocessing_transform()
    augment = self.get_augmentation_transform()
    self.dataset = MRI2DSegmentationDataset(self.filename_pairs_train, transform = self.preprocess, slice_axis=2, canonical = False)
    print(f'total number of slices trainig {len(self.dataset)}')
  def __getitem__(self, index):
    # Label Image
    data_input = self.dataset[index]

    input_dict = {'label': data_input['gt'],

                  'image': data_input['input'],
                  'path': data_input['filename'],
                  'gtname': data_input['gtname'],
                  'index': data_input['index'],
                  'segpair_slice': data_input['segpair_slice'],

                  }

    return input_dict
  def __len__(self):
    return self.dataset.__len__()


def create_dataloader_train(opt):
  dataset = MoodMedtorchDataModule(opt)
  dataset.setup()
  
  if dist.is_initialized() or opt.parallel:
    print('local rank is {}'.format(dist.get_rank()))
    print('world size is {}'.format(dist.get_world_size()))
    data_loader = DataLoader(dataset, pin_memory=True, shuffle=False, batch_size= opt.batch_size, sampler=DistributedSampler(dataset))  # this must be 0
  else:
    data_loader = DataLoader(dataset, opt.batch_size,shuffle=True, num_workers=opt.num_workers)
  return data_loader

def create_dataloader_valid(self):
  val_dataset = MRI2DSegmentationDataset(self.filename_pairs_val, transform = self.preprocess, slice_axis=2, canonical = False)
  print(f'total number of slices validation {len(val_dataset)}')

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

  
    data_loader = DataLoader(val_dataset, pin_memory=True, batch_size= self.batch_size, shuffle=False, sampler=DistributedSampler(val_dataset))  # this must be 0
  else:
    data_loader = DataLoader(val_dataset, self.batch_size,shuffle=False, num_workers=self.num_workers)
  return data_loader




class SampleMetadata(object):
  def __init__(self, d=None):
    self.metadata = {} or d

  def __setitem__(self, key, value):
    self.metadata[key] = value

  def __getitem__(self, key):
    return self.metadata[key]

  def __contains__(self, key):
    return key in self.metadata

  def keys(self):
    return self.metadata.keys()


class MTTransform(object):

    def __call__(self, sample):
        raise NotImplementedError("You need to implement the transform() method.")

    def undo_transform(self, sample):
        raise NotImplementedError("You need to implement the undo_transform() method.")

class ToTensor(MTTransform):
  """Convert a PIL image or numpy array to a PyTorch tensor."""

  def __init__(self, labeled=True):
    self.labeled = labeled

  def __call__(self, sample):
    rdict = {}
    input_data = sample['input']

    if isinstance(input_data, list):
        ret_input = [F.to_tensor(item)
                      for item in input_data]
    else:
        ret_input = F.to_tensor(input_data)

    rdict['input'] = ret_input

    if self.labeled:
        gt_data = sample['gt']
        if gt_data is not None:
            if isinstance(gt_data, list):
                ret_gt = [F.to_tensor(item)
                          for item in gt_data]
            else:
                ret_gt = F.to_tensor(gt_data)

            rdict['gt'] = ret_gt
    sample.update(rdict)
    return sample


class ToPIL(MTTransform):
  def __init__(self, labeled=True):
      self.labeled = labeled

  def sample_transform(self, sample_data):
    # Numpy array
    if not isinstance(sample_data, np.ndarray):
        input_data_npy = sample_data.numpy()
    else:
        input_data_npy = sample_data

    input_data_npy = np.transpose(input_data_npy, (1, 2, 0))
    input_data_npy = np.squeeze(input_data_npy, axis=2)
    input_data = Image.fromarray(input_data_npy, mode='F')
    return input_data

  def __call__(self, sample):
    rdict = {}
    input_data = sample['input']

    if isinstance(input_data, list):
        ret_input = [self.sample_transform(item)
                      for item in input_data]
    else:
        ret_input = self.sample_transform(input_data)

    rdict['input'] = ret_input

    if self.labeled:
        gt_data = sample['gt']

        if isinstance(gt_data, list):
            ret_gt = [self.sample_transform(item)
                      for item in gt_data]
        else:
            ret_gt = self.sample_transform(gt_data)

        rdict['gt'] = ret_gt

    sample.update(rdict)
    return sample

class MRI2DSegmentationDataset(Dataset):
  """This is a generic class for 2D (slice-wise) segmentation datasets.

  :param filename_pairs: a list of tuples in the format (input filename,
                          ground truth filename).
  :param slice_axis: axis to make the slicing (default axial).
  :param cache: if the data should be cached in memory or not.
  :param transform: transformations to apply.
  """
  def __init__(self, filename_pairs, slice_axis=2, cache=True,
            transform=None, slice_filter_fn=None, canonical=False):
    self.filename_pairs = filename_pairs
    self.handlers = []
    self.indexes = []
    self.transform = transform
    self.cache = cache
    self.slice_axis = slice_axis
    self.slice_filter_fn = slice_filter_fn
    self.canonical = canonical

    self._load_filenames()
    self._prepare_indexes()

  def _load_filenames(self):
    for input_filename, gt_filename in self.filename_pairs:
        segpair = SegmentationPair2D(input_filename, gt_filename,
                                      self.cache, self.canonical)
        self.handlers.append(segpair)

  def _prepare_indexes(self):
    for segpair in self.handlers:
        input_data_shape, _ = segpair.get_pair_shapes()
        for segpair_slice in range(input_data_shape[2]):

            # Check if slice pair should be used or not
            if self.slice_filter_fn:
                slice_pair = segpair.get_pair_slice(segpair_slice,
                                                    self.slice_axis)

                filter_fn_ret = self.slice_filter_fn(slice_pair)
                if not filter_fn_ret:
                    continue

            item = (segpair, segpair_slice)
            self.indexes.append(item)

  def set_transform(self, transform):
    """This method will replace the current transformation for the
    dataset.

    :param transform: the new transformation
    """
    self.transform = transform

  def compute_mean_std(self, verbose=False):
    """Compute the mean and standard deviation of the entire dataset.

    :param verbose: if True, it will show a progress bar.
    :returns: tuple (mean, std dev)
    """
    sum_intensities = 0.0
    numel = 0

    with DatasetManager(self,
                        override_transform=ToTensor()) as dset:
        pbar = tqdm(dset, desc="Mean calculation", disable=not verbose)
        for sample in pbar:
            input_data = sample['input']
            sum_intensities += input_data.sum()
            numel += input_data.numel()
            pbar.set_postfix(mean="{:.2f}".format(sum_intensities / numel),
                              refresh=False)

        training_mean = sum_intensities / numel

        sum_var = 0.0
        numel = 0

        pbar = tqdm(dset, desc="Std Dev calculation", disable=not verbose)
        for sample in pbar:
            input_data = sample['input']
            sum_var += (input_data - training_mean).pow(2).sum()
            numel += input_data.numel()
            pbar.set_postfix(std="{:.2f}".format(np.sqrt(sum_var / numel)),
                              refresh=False)

    training_std = np.sqrt(sum_var / numel)
    return training_mean.item(), training_std.item()
  
  def compute_mean_resolution(self, verbose=False):
    """Compute the mean resolution of the entire dataset.

    :param verbose: if True, it will show a progress bar.
    :returns: tuple (mean, std dev)
    """
    #TODO: complete it
    sum_resolution = 0.0
    numel = 0

    with DatasetManager(self,
                        override_transform=ToTensor()) as dset:
        pbar = tqdm(dset, desc="Mean resolution calculation", disable=not verbose)
        for sample in pbar:
            input_zoom = sample['input_metadata']['zooms'][0]
            # print(sample['input_metadata']['zooms'])
            # print(sample['input'].shape)
            sum_resolution += input_zoom
            numel += 1
            pbar.set_postfix(mean="{:.2f}".format(sum_resolution / numel),
                              refresh=False)

        training_mean = sum_resolution / numel

    return training_mean.item()

  def __len__(self):
    """Return the dataset size."""
    return len(self.indexes)

  def __getitem__(self, index):
    """Return the specific index pair slices (input, ground truth).

    :param index: slice index.
    """
    segpair, segpair_slice = self.indexes[index]
    pair_slice = segpair.get_pair_slice(segpair_slice,
                                        self.slice_axis)

    # Consistency with torchvision, returning PIL Image
    # Using the "Float mode" of PIL, the only mode
    # supporting unbounded float32 values
    input_img = Image.fromarray(pair_slice["input"], mode='F')

    # Handle unlabeled data
    if pair_slice["gt"] is None:
        gt_img = None
    else:
        gt_img = Image.fromarray(pair_slice["gt"], mode='F')

    data_dict = {
        'input': input_img,
        'gt': gt_img,
        'input_metadata': pair_slice['input_metadata'],
        'gt_metadata': pair_slice['gt_metadata'],

        'filename':segpair.input_filename,
        'gtname':segpair.gt_filename,
        'index': index,
        'segpair_slice':segpair_slice,
    }

    if self.transform is not None:
        data_dict = self.transform(data_dict)

    return data_dict
    

class DatasetManager(object):
    def __init__(self, dataset, override_transform=None):
        self.dataset = dataset
        self.override_transform = override_transform
        self._transform_state = None

    def __enter__(self):
        if self.override_transform:
            self._transform_state = self.dataset.transform
            self.dataset.transform = self.override_transform
        return self.dataset

    def __exit__(self, *args):
        if self._transform_state:
            self.dataset.transform = self._transform_state


class SegmentationPair2D(object):
  """This class is used to build 2D segmentation datasets. It represents
  a pair of of two data volumes (the input data and the ground truth data).

  :param input_filename: the input filename (supported by nibabel).
  :param gt_filename: the ground-truth filename.
  :param cache: if the data should be cached in memory or not.
  :param canonical: canonical reordering of the volume axes.
  """
  def __init__(self, input_filename, gt_filename, cache=True,
              canonical=False):
    self.input_filename = input_filename
    self.gt_filename = gt_filename
    self.canonical = canonical
    self.cache = cache

    self.input_handle = nib.load(self.input_filename)

    # Unlabeled data (inference time)
    if self.gt_filename is None:
        self.gt_handle = None
    else:
        self.gt_handle = nib.load(self.gt_filename)

    if len(self.input_handle.shape) > 3:
        raise RuntimeError("4-dimensional volumes not supported.")

    # Sanity check for dimensions, should be the same
    input_shape, gt_shape = self.get_pair_shapes()

    if self.gt_handle is not None:
        if not np.allclose(input_shape, gt_shape):
            raise RuntimeError('Input and ground truth with different dimensions.')

    if self.canonical:
        self.input_handle = nib.as_closest_canonical(self.input_handle)

        # Unlabeled data
        if self.gt_handle is not None:
            self.gt_handle = nib.as_closest_canonical(self.gt_handle)

  def get_pair_shapes(self):
    """Return the tuple (input, ground truth) representing both the input
    and ground truth shapes."""
    input_shape = self.input_handle.header.get_data_shape()

    # Handle unlabeled data
    if self.gt_handle is None:
        gt_shape = None
    else:
        gt_shape = self.gt_handle.header.get_data_shape()

    return input_shape, gt_shape

  def get_pair_data(self):
    """Return the tuble (input, ground truth) with the data content in
    numpy array."""
    cache_mode = 'fill' if self.cache else 'unchanged'
    input_data = self.input_handle.get_fdata(cache_mode, dtype=np.float32)

    # Handle unlabeled data
    if self.gt_handle is None:
        gt_data = None
    else:
        gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)

    return input_data, gt_data

  def get_pair_slice(self, slice_index, slice_axis=2):
    """Return the specified slice from (input, ground truth).

    :param slice_index: the slice number.
    :param slice_axis: axis to make the slicing.
    """
    if self.cache:
        input_dataobj, gt_dataobj = self.get_pair_data()
    else:
        # use dataobj to avoid caching
        input_dataobj = self.input_handle.dataobj

        if self.gt_handle is None:
            gt_dataobj = None
        else:
            gt_dataobj = self.gt_handle.dataobj

    if slice_axis not in [0, 1, 2]:
        raise RuntimeError("Invalid axis, must be between 0 and 2.")

    if slice_axis == 2:
        input_slice = np.asarray(input_dataobj[..., slice_index],
                                  dtype=np.float32)
    elif slice_axis == 1:
        input_slice = np.asarray(input_dataobj[:, slice_index, ...],
                                  dtype=np.float32)
    elif slice_axis == 0:
        input_slice = np.asarray(input_dataobj[slice_index, ...],
                                  dtype=np.float32)

    # Handle the case for unlabeled data
    gt_meta_dict = None
    if self.gt_handle is None:
        gt_slice = None
    else:
        if slice_axis == 2:
            gt_slice = np.asarray(gt_dataobj[..., slice_index],
                                  dtype=np.float32)
        elif slice_axis == 1:
            gt_slice = np.asarray(gt_dataobj[:, slice_index, ...],
                                  dtype=np.float32)
        elif slice_axis == 0:
            gt_slice = np.asarray(gt_dataobj[slice_index, ...],
                                  dtype=np.float32)

        gt_meta_dict = SampleMetadata({
            "zooms": self.gt_handle.header.get_zooms()[:2],
            "data_shape": self.gt_handle.header.get_data_shape()[:2],
        })

    input_meta_dict = SampleMetadata({
        "zooms": self.input_handle.header.get_zooms()[:2],
        "data_shape": self.input_handle.header.get_data_shape()[:2],
    })

    dreturn = {
        "input": input_slice,
        "gt": gt_slice,
        "input_metadata": input_meta_dict,
        "gt_metadata": gt_meta_dict,
    }

    return dreturn