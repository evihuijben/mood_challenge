from torch.utils.data import DataLoader
import torchio as tio
import os
import torch


class MoodDataModule():
    # taken from main code 49d9c074a3ebe2670c49ab2b02593ea80f19206e

  def __init__(self, args):
    super().__init__()
    self.task = args.task
    self.input_dir = args.input_dir
    self.num_workers = args.num_workers
    self.batch_size = args.batch_size
    

  def test_dataloader(self):
    if self.task == 'brain':
        # transform = tio.Compose([tio.RescaleIntensity(out_min_max=(0, 1))])  
        transform = None
    else:
         transform =  tio.Compose([tio.Resize((256, 256, 256))])
         
        
    names = sorted(os.listdir(self.input_dir))
    file_names = [os.path.join(self.input_dir, name) for name in names]

    print('File names test dataloader' ,len(file_names), file_names)
    test_subjects = [tio.Subject(image=tio.ScalarImage(image_path), path=str(image_path)) for image_path in file_names]
    
    test_set = tio.SubjectsDataset(test_subjects, transform=transform)
    data_loader = DataLoader(test_set, 
                             self.batch_size,
                             shuffle=False, 
                             num_workers=self.num_workers)
    return data_loader


def upsample(img):
    img=torch.from_numpy(img)[None]
    transform = tio.Resize((512, 512, 512), image_interpolation='nearest')
    new_img = transform(img)
    new_img = new_img.squeeze().numpy()
    return new_img
    