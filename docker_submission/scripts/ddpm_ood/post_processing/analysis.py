# %%
# python analysis.py --run mood_simplex_ddp_07_12/nifti_modified/out/T100300600 --phase toy --combine_method mean,mask,ssim,blur,mask --mask_radius 5 --blur_sigma 5

import sys

sys.path.append('post_processing')
import torch
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np 
from scipy.ndimage import gaussian_filter
import argparse
from utils import dsc_score, segment, ssim_pad, calculate_challenge_metrics
import copy
import time
import json
from sklearn import metrics

# %%

def load_config():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--region', type=str, default='brain', help='[brain, abdom]')
    parser.add_argument('--phase', type=str, default='val', help='[toy, val, train]')
    parser.add_argument('--suffix', type=str, default='_transformed')
    parser.add_argument('--results_dir', type=str, default="/home/bme001/shared/mood/code/ddpm-ood/checkpoints")
    parser.add_argument('--run', type=str, default="mood_simplex_ddp_07_12/nifti_modified/val/T100")

    parser.add_argument('--combine_method', type=str, default="select,mask,ssim,blur,mask")
    parser.add_argument('--select_recon', type=int, default=0)
    parser.add_argument('--blur_sigma', type=int, default=15)
    parser.add_argument('--mask_radius', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--save_predictions', type=bool, default=True)
    parser.add_argument('--make_plots', type=bool, default=True)
    parser.add_argument('--add_evaluation', type=bool, default=True)

    args = parser.parse_args()
    if os.uname()[1] == 'TUE026169':
        args.dataroot="/mnt/sda/Data/MOOD"
    else:
        args.dataroot="/home/bme001/shared/mood/data"

    combine_id = ''
    for method in args.combine_method.split(','):
        if method == 'blur':
            combine_id += f'blur{args.blur_sigma}_'
        elif method == 'mask':
            combine_id += f"mask{args.mask_radius}_"
        elif method == 'select':
            combine_id += f"recon{args.select_recon}_"
        else:
            combine_id += f"{method}_"
    combine_id = combine_id[:-1]
    
    if args.phase == 'toy':
        args.real_root = os.path.join(args.dataroot, f"{args.region}_{args.phase}", args.phase)
        args.label_root = os.path.join(args.dataroot, f"{args.region}_{args.phase}", f"{args.phase}_label")
    else:
        args.real_root = os.path.join(args.dataroot, f"{args.region}_{args.phase}", f"{args.region}_{args.phase}{args.suffix}")
        args.label_root = os.path.join(args.dataroot, f"{args.region}_{args.phase}", f"{args.region}_{args.phase}{args.suffix}_label")

    args.save_folder = f"analysis_visuals/{args.run}/{combine_id}"
    if os.getcwd().split('/')[-1] == 'ddpm-ood':
        args.save_folder = os.path.join("post_processing" , args.save_folder )
    return args


def process_cases(args):

    predictions = {}
    scores = {}

    all_global_pred, all_global_gt = [], []
    all_local_pred, all_local_gt = [], []

    for fname in sorted(os.listdir(args.real_root)):
        ID = fname.split('.')[0]
        print(ID)
        
        result, binary_result, gt, avg_recon = process_one_case(args, fname)
        predictions[ID] = binary_result

        if args.add_evaluation == True or args.make_plots == True:
            score, global_label, label = evaluate_one_case(args, ID,  binary_result)
            print('\t global_label =', global_label, 'label_sum =', label.sum())
            
            all_global_pred.append(score['global_pred'])
            all_global_gt.append(global_label)

            all_local_gt.append(label.flatten())
            all_local_pred.append(binary_result.flatten())

            scores[ID] = score

            if args.make_plots == True:
                make_plot(args, ID, gt, global_label, label, avg_recon, result, binary_result, score)

    if args.add_evaluation == True:
        tp, fp, fn = 0, 0, 0
        for ID, score in scores.items():
            if 'F1_hull' in score.keys():
                tp += score['TP_hull']
                fp += score['FP_hull']
                fn += score['FN_hull']
        
        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_precision_global = metrics.average_precision_score(all_global_gt,all_global_pred )

        all_local_gt = np.concatenate(all_local_gt)
        all_local_pred = np.concatenate(all_local_pred)
        avg_precision_local = metrics.average_precision_score(all_local_gt,all_local_pred )

        overall_local_scores = {'Avg_precision_global': avg_precision_global,
                                'Avg_precision_local': avg_precision_local,
                                'Precision_hull': precision,
                                'Recall_hull': recall,
                                'F1_hull': f1_score,
                                }

        save_dict = {'overall_local_scores': overall_local_scores,
                     'score_per_patient': scores}
        with open(os.path.join(args.save_folder, f"scores_th{args.threshold}.json"), 'w') as outf:
            json.dump(save_dict, outf, indent=2)
    return predictions, scores


def process_one_case(args, fname):
    # Record the start time
    start_time = time.time()
    fname_recon = fname.replace('.nii.gz', '_recon.nii.gz')
    fname_pred = os.path.join(args.save_folder, "ssim_maps", "prediction_"+fname)
    
    gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.real_root, fname)))
    result = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.results_dir, args.run, fname_recon)))
    if result.ndim == 3:
            result = result[None]

    if 'mean' in args.combine_method:
        avg_recon = result.mean(0)
    elif 'select' in args.combine_method:
        avg_recon = result[args.select_recon]

    if os.path.isfile(fname_pred):
        result = sitk.GetArrayFromImage(sitk.ReadImage(fname_pred))
        end_time = time.time()
    else:
        print('recon shape', result.shape)
        if 'mask' in args.combine_method:
            mask = segment(gt, r=args.mask_radius)

        ssim_pos=100
        for step, method in enumerate(args.combine_method.split(',')):
            if method == 'mean':
                result = result.mean(0)[None]
                
            elif method == 'select':
                result = result[args.select_recon][None]
            elif method == 'blur':
                result = np.stack([gaussian_filter(r, args.blur_sigma) for r in result])
            elif method == 'ssim':
                result = np.stack([ssim_pad(ssim(gt, r, data_range=1, full=True)[1]) for r in result])
                ssim_pos = step
            elif method == 'mask':
                if step < ssim_pos:
                    result = np.stack([ mask*r for r in result])
                else:
                    result = np.stack([ np.where(mask==0, 1, r) for r in result])
            print('\t', method, result.shape)

        # Reduce shape
        if result.shape[0] != 1:
            print('Not reduced!')
        result = result[0]
        end_time = time.time()

        if args.save_predictions == True:
            os.makedirs(os.path.join(args.save_folder, "ssim_maps"), exist_ok=True)
            sitk.WriteImage(sitk.GetImageFromArray(result),fname_pred)

    # Binarize prediction
    binary_result = np.where(result > args.threshold, 0., 1.)

    # Calculate the elapsed time
    print(f"\t Elapsed time: {end_time - start_time:.1f} seconds")
    return result, binary_result, gt, avg_recon


def evaluate_one_case(args, ID, result):
    print('\t Evaluating ...')

    if args.phase == 'val' and args.suffix == '':
        global_label = 0
    else:
        with open(os.path.join(args.label_root, "sample", f"{ID}.nii.gz.txt"), 'r') as inf:
            global_label = int(inf.read().strip())

    sum_ratio = result.sum()/np.prod(result.shape)
    global_pred = int(result.sum() > 0)
    score = {'sum_ratio': sum_ratio,
             'global_label': global_label,
             'global_pred': global_pred}
    
    hull_objects_fname = os.path.join(args.label_root, 'hull', f"{ID}.nii.gz")
    if os.path.isfile(hull_objects_fname):
        hull_objects = sitk.GetArrayFromImage(sitk.ReadImage(hull_objects_fname))
        if hull_objects.ndim == 3:
            hull_objects = hull_objects[None]
        
        label_fname = os.path.join(args.label_root, 'pixel', f"{ID}.nii.gz")
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_fname))

        score['DSC'] = dsc_score(result, label)

        score['Avg_precision'] = metrics.average_precision_score(label.flatten(), result.flatten())

        tp, fp, fn, f1_score, precision, recall, sum_per_pred_obj = calculate_challenge_metrics(result, hull_objects, threshold=args.threshold)
        
        score['TP_hull'] = tp
        score['FP_hull'] = fp 
        score['FN_hull'] = fn
        score['F1_hull'] = f1_score
        score['Precision_hull'] = precision
        score['Recall_hull'] = recall
        score['sum_per_pred_obj'] = sum_per_pred_obj

    else:
        label = np.zeros(result.shape, dtype=np.float32)
                
    return score, global_label, label, 

def make_plot(args, ID, gt, global_label, label, avg_recon, result, binary_result, score):
    os.makedirs(os.path.join(args.save_folder, f"overviews_th{args.threshold}"), exist_ok=True)

    if label.sum() == 0:
        slices = [label.shape[0]//2, label.shape[1]//2, label.shape[2]//2]
    else:
        slices=[]
        slices.append(label.sum((1,2)).argmax())
        slices.append(label.sum((0,2)).argmax())
        slices.append(label.sum((0,1)).argmax())

    ncols = 4
    
    fig, ax = plt.subplots(3, ncols, figsize=(ncols*4, 3*4+1))

    obj = {}
    obj["00"] = ax[0,0].imshow(gt[slices[0]] , cmap='gray', vmin=0, vmax=1)
    obj["01"] = ax[0,1].imshow(avg_recon[slices[0]], cmap='gray', vmin=0, vmax=1)
    obj["02"] = ax[0,2].imshow(result[slices[0]], cmap='RdYlGn', vmin=0, vmax=1)
    obj["03"] = ax[0,3].imshow(binary_result[slices[0]], cmap='gray', vmin=0, vmax=1)

    obj["10"] = ax[1,0].imshow(gt[:,slices[1]] , cmap='gray', vmin=0, vmax=1)
    obj["11"] = ax[1,1].imshow(avg_recon[:,slices[1]], cmap='gray', vmin=0, vmax=1)
    obj["12"] = ax[1,2].imshow(result[:,slices[1]], cmap='RdYlGn', vmin=0, vmax=1)
    obj["13"] = ax[1,3].imshow(binary_result[:,slices[1]], cmap='gray', vmin=0, vmax=1)
   
    obj["20"] = ax[2,0].imshow(gt[:,:,slices[2]] , cmap='gray', vmin=0, vmax=1)
    obj["21"] = ax[2,1].imshow(avg_recon[:,:,slices[2]], cmap='gray', vmin=0, vmax=1)
    obj["22"] = ax[2,2].imshow(result[:,:,slices[2]], cmap='RdYlGn', vmin=0, vmax=1)
    obj["23"] = ax[2,3].imshow(binary_result[:,:,slices[2]], cmap='gray', vmin=0, vmax=1)

    ax[0,0].set_title(f'Input image (slice={slices[0]})')
    ax[1,0].set_title(f'Input image (slice={slices[1]})')
    ax[2,0].set_title(f'Input image (slice={slices[2]})')
    ax[0,1].set_title(f'mean/selected reconstruction')
    ax[0,2].set_title('Processed SSIM map')

    
    if 'DSC' in score.keys():
        this_title = f"Th={args.threshold} DSC={score['DSC']:.2f} F1={score['F1_hull']:.2f}"
    else:
        d = {k: v for k, v in score.items() if k != 'gt_to_hull'}

        this_title = f"Th={args.threshold}  {' '.join([f'{k}={v:.2f}' for k, v in d.items()])}"

    ax[0,3].set_title(this_title)

    for row in range(3):
        for col in range(ncols):    
            plt.colorbar(obj[f"{row}{col}"], ax=ax[row,col], fraction=0.04)
            ax[row,col].axis('off')

    global_label = 'healthy' if global_label==0 else 'ood'
    plt.suptitle(f"{ID} {global_label} {args.combine_method}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_folder, f"overviews_th{args.threshold}", f"{ID}.png"))
    # plt.show()
    plt.close()




if __name__ == '__main__':
    args = load_config()
    predictions, scores = process_cases(args)




       
# %%
