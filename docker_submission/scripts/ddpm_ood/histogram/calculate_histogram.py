# %%
import os
import nibabel as nib
from tqdm import tqdm
import numpy as  np
import matplotlib.pyplot as plt
### average histogram of all training data
# train_dir = '/home/bme001/shared/mood/data/abdom_train/abdom_train'
train_dir = '/home/bme001/shared/mood/abdom_train/abdom_train_resampled'
how_many = len(sorted(os.listdir(train_dir)))
n_bins = 4096
hist_mean = np.zeros(n_bins)
x_vals_mean = np.zeros(n_bins)
all_hist = []
all_bins = []
for idx, image in enumerate(tqdm(sorted(os.listdir(train_dir)))):
    if idx > how_many:
        break
    image_data = nib.load(os.path.join(train_dir, image)).get_fdata()
    image_flatened = image_data.flatten()
    hist, bins = np.histogram(image_flatened, bins=n_bins, range=(0, 1))
    all_hist.append(hist)
    all_bins.append(bins)
    hist_mean +=hist
    x_vals = bins[:-1]
    x_vals_mean+=x_vals
    plt.plot(x_vals, hist)
hist_mean/=how_many
x_vals_mean/=how_many
all_hist = np.array(all_hist)
all_bins = np.array(all_bins)
print(all_hist.shape)
plt.plot(x_vals_mean,hist_mean, "b*" )
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.ylim(0.01, 100000)
plt.show()
# %%
out_dir = '/home/bme001/shared/mood/code/ddpm-ood/histogram'
np.save(os.path.join(out_dir, f'all_hist_abdomen{n_bins}.npy'), all_hist)
np.save(os.path.join(out_dir, f'all_bins_abdomen{n_bins}.npy'), all_bins)
print('saving histogram finished')
# %%

# %%
