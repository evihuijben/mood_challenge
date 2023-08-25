import os

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

def load_train_hist(hist_root, region, n_bins = 4096):
    all_hist = np.load(os.path.join(hist_root, f'all_hist_{region}{n_bins}.npy'))
    
    hist_std = np.std(all_hist, axis=0, keepdims=True).squeeze()
    hist_std[hist_std<16] = 16
    hist_mean = np.mean(all_hist, axis=0, keepdims=True).squeeze()
    print('histogram loaded')
    
    
    
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(hist_mean, label='mean')
    # plt.plot(hist_std, label='std')
    # plt.plot(4*hist_std, label='4std')
    # # for i in range(50):
    # #     plt.plot(all_hist[i])
    # plt.ylim([0,4000])
    # plt.legend()
    # plt.show()
    
    return {'mean': hist_mean,
            'std': hist_std,
            'n_bins': n_bins}
def show_image_mask(image_array, thresholded_image):
    
    slice_id=[np.argmax(thresholded_image.sum((0,1))), 1]
    
    # import matplotlib.pyplot as plt
    # # Display the original image and the segmented masks
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(np.concatenate((image_array[:,:, slice_id[0]],image_array[:,:, slice_id[1]])), cmap='gray')
    # axs[0].set_title('Original Image')
    # axs[0].axis('off')
    # axs[1].imshow(np.concatenate((thresholded_image[:,:, slice_id[0]],thresholded_image[:,:, slice_id[1]])), cmap='gray')
    # axs[1].set_title('mask')
    # axs[1].axis('off')
    # plt.tight_layout()
    # plt.show()


def calculate_local_prediction(image_data, hist_stats, title):
    image_flatened = image_data.flatten()
    hist, bins = np.histogram(image_flatened, bins=hist_stats['n_bins'], range=(0, 1))
    
    sub_hist = np.abs(hist - hist_stats['mean']) 
    sub_hist[sub_hist <= 4 * hist_stats['std']] = 0 
    
    # print(sub_hist[sub_hist>0])
    
    spike_inds = sub_hist.nonzero()[0]
    spike_inds.sort()
    if len(spike_inds) > 0  and spike_inds[0] == 0:
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

        show_image_mask(image_data, thresholded_image)
        return thresholded_image
    else:
        return thresholded_image


def predict_folder_abs(mode, input_folder, target_folder, hist_stats):
    for f in os.listdir(input_folder):
        source_file = os.path.join(input_folder, f)
        image_nib = nib.load(source_file)
        image_array = image_nib.get_fdata()
        
        
        pred_local = calculate_local_prediction(image_array, hist_stats, f)
        
        
        print(mode, f, pred_local.sum())
        
        if mode == "pixel":
            target_file = os.path.join(target_folder, f)
            final_nib_img = nib.Nifti1Image(pred_local, affine=image_nib.affine)
            nib.save(final_nib_img, target_file)
        elif mode == "sample":
            if pred_local.any()>0:
                abnomal_score = 1.0
            else:
                abnomal_score = 0.0
                
            
            with open(os.path.join(target_folder, f + ".txt"), "w") as write_file:
                write_file.write(str(abnomal_score))

        else:
            print("Mode not correctly defined. Either choose 'pixel' oder 'sample'")
        
        
        
if __name__ == "__main__":

    print()
    print('************* TEST GPU ********************')
    import torch
    print('CUDA AVAILABLE:', torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE:', device)
    test_gpu = torch.ones((1,)).to(device)
    print('TENSOR DEVICE', test_gpu.device)
    
    print('*******************************************')
    print()
    
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("-region", type=str, default="brain", help="can be either 'brain' or 'abdom'.", required=False)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    mode = args.mode

    hist_root = '/workspace/histogram'
    hist_stats = load_train_hist(hist_root, args.region, n_bins = 4096)
    predict_folder_abs(mode, input_dir, output_dir, hist_stats)
    

    
    
    # hist_root = 'histogram'
    # region = 'brain'
    # hist_stats = load_train_hist(hist_root, region, n_bins = 4096)    
    
    
    
    # predict_folder_abs('pixel', '/mnt/sda/Data/MOOD/brain_toy/toy', '.', hist_stats)
    # predict_folder_abs('pixel', '/mnt/sda/Data/MOOD/brain_val/brain_val_transformed', '../tmp_output', hist_stats)
    