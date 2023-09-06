# TODO : remove plot function
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torchio as tio
   
    
def create_qualitative_results(batch, recon, metric_map, thresholded_image, config):
    #%%
    name = Path(batch['path'][0]).name
    image_array = batch["image"][tio.DATA].float().squeeze().cpu().numpy()
    
    
    slice_nr=np.argmax(thresholded_image.sum((0,1)))
    if slice_nr == 0:
        slice_nr = image_array.shape[2]//2
    # Display the original image and the segmented masks
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))
    obj = axs[0].imshow(image_array[:,:, slice_nr].transpose(), cmap='gray', vmin=0, vmax=1)
    plt.colorbar(obj, ax=axs[0], fraction=0.045)
    axs[0].axis('off')
    
    if recon is not None:
        recon = recon.squeeze()
        obj = axs[1].imshow(recon[:,:, slice_nr].transpose(), cmap='gray')
        plt.colorbar(obj, ax=axs[1], fraction=0.045)
        obj = axs[2].imshow(metric_map[:,:, slice_nr].transpose(), vmin=0, vmax=1, cmap='RdYlGn')
        plt.colorbar(obj, ax=axs[2], fraction=0.045)
        
    axs[1].axis('off')
    axs[2].axis('off')
    
    obj = axs[3].imshow(thresholded_image[:,:, slice_nr].transpose(), cmap='gray', vmin=0, vmax=1)
    plt.colorbar(obj, ax=axs[3], fraction=0.045)
    axs[3].axis('off')
    plt.tight_layout()
    save_name= os.path.join(config.result_dir, name.split('.')[0] + '_visuals.svg')
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()

def create_histogram_plot_paper(bins,hist, hist_mean, hist_std, batch, image, thresholded_image, config):
    name = Path(batch['path'][0]).name
    toy_label = f"Toy example {name.split('.')[0].split('_')[1]}"
    
    sns.set_theme()
    fig, ax = plt.subplots(1,3, figsize=(13,3))
    
    slice_nr = np.argmax(thresholded_image.sum((0,1)))
    obj = ax[0].imshow(np.flipud(np.rot90(image.squeeze()[:,:,slice_nr])), cmap='gray', vmin=0, vmax=1)
    ax[0].axis('off')
    ax[0].set_title(toy_label)
    plt.colorbar(obj, ax=ax[0], fraction=0.05)
    
    
    xpos_bins = bins[:-1] + (bins[1] - bins[0])/2
    colors = sns.color_palette("Set2", 4)
    ax[1].plot(xpos_bins, hist, color=colors[1], label=toy_label)
    ax[1].plot(xpos_bins,hist_mean, color=colors[0],  label='Mean of training set\nwith $4\sigma$')
    ax[1].plot(xpos_bins,hist_mean+4*hist_std, color=colors[0], alpha=0.2) 
    ax[1].plot(xpos_bins,hist_mean-4*hist_std, color=colors[0], alpha=0.2) 
    ax[1].fill_between(xpos_bins, hist_mean-4*hist_std, hist_mean+4*hist_std, color=colors[0], alpha=0.1)
    ax[1].set_ylim([0, 10000])
    ax[1].set_xlim([0,1])
    ax[1].set_xlabel('Intensity value')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    ax[1].set_title('Histograms')
    
    
    sub_hist = np.abs(hist - hist_mean)
    ax[2].plot(xpos_bins, sub_hist, color=colors[1], label='Detected peak')
    
    
    y_an = np.where(sub_hist >  4*hist_std, sub_hist, 0)
    for x, y in zip(xpos_bins, y_an):
        if y >0:
            ax[2].annotate(f"{x:.2f}", (x+0.01, 4500), color='black',
                           bbox=dict(boxstyle='round, pad=0.3', edgecolor=colors[1], facecolor=colors[1]))

            xpos = np.argmax(thresholded_image[:,:,slice_nr].sum(1))
            ypos = np.argmax(thresholded_image[:,:,slice_nr].sum(0))
            # ax[0].annotate(f"{x:.2f}", (xpos-5, ypos-20), color='white')
            ax[0].annotate(f"{x:.2f}", (xpos-8, ypos+12), color='black',
                           bbox=dict(boxstyle='round, pad=0.3', edgecolor=colors[1], facecolor=colors[1]))
    
    
    ax[2].set_ylim([0, 10000])
    ax[2].set_xlim([0,1])
    ax[2].set_xlabel('Intensity value')
    ax[2].set_ylabel('Frequency')
    ax[2].set_title('Out of distribution detection')
    ax[2].legend(loc='center right')
       
    
    plt.tight_layout()
    save_name= os.path.join(config.result_dir, name + '.pdf')
    plt.savefig(save_name, bbox_inches='tight')
     
    plt.show()