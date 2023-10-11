
# %%
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
import numpy as np
# %%

root = 'analysis_visuals/mood_simplex_ddp_07_12/nifti_modified'
region='brain'


label_roots = {'out': f'/home/bme001/shared/mood/data/{region}_toy/toy_label/sample',
               'in':  None,
               'val': f'/home/bme001/shared/mood/data/{region}_val/brain_val_transformed_label/sample'  }
all_results = {}
for group in os.listdir(root):
    all_results[group] = {}
    for T in os.listdir(os.path.join(root, group)):
        all_results[group][T] = {}
        for method in os.listdir(os.path.join(root, group, T)):
            print(group, T, method)
            with open(os.path.join(root, group, T, method, 'scores.json'), 'r') as inf:
                results = json.load(inf)

            metrics = {'DSC': [],
                       'FP/N ood': [],
                       'TP/n healhty': []}
            for ID, result in sorted(results.items()):
                if label_roots[group] is None:
                    global_label = 0
                    metrics['FP/N healhty'].append(result['FP/N'])
                else:
                    label_file = os.path.join(label_roots[group], f"{ID}.nii.gz.txt")
                    with open(label_file, 'r') as inf:
                        global_label = int(inf.read().strip())

                    if global_label == 0:
                        metrics['FP/N healhty'].append(result['FP/N'])
                    else:
                        if list(result.keys())[0] == 'DSC':
                            metrics['DSC'].append(result['DSC'])
                            prediction_map = os.path.join(root, group, T, method, 'ssim_maps',f"prediction_{ID}.nii.gz")

                            pred = sitk.GetArrayFromImage(sitk.ReadImage(prediction_map))
                            pred = np.where(pred > 0.5, 0., 1.)
                            pred =  pred.sum()/np.prod(pred.shape)
                            
                            metrics['TP/n ood'].append(pred)
                        else:
                            
                            metrics['TP/n ood'].append(result['FP/N'])

            all_results[group][T][method] = metrics
print('All results collected')
print()

# %%
sns.set_theme()
translator = {'out': 'toy', 'in': 'val (original)', 'val' : 'val (tranformed)'}

# %%
group='in'
T_values = list(all_results[group].keys())

plt.figure(figsize=(12, 6))  # Create a single figure with a larger size

method_names = sorted(all_results[group][T_values[0]].keys())
num_methods = len(method_names)
for t_index, T in enumerate(T_values):

    method_scores = [all_results[group][T][method]['DSC'] for method in method_names]
    positions = [i + t_index * (num_methods + 1) for i in range(1, num_methods + 1)]
    plt.boxplot(method_scores, positions=positions, labels=method_names)

plt.xlabel('Method')
plt.ylabel('DSC')
plt.title(f'DSC scores of local cases {translator[group]}')

plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
# plt.ylim([-0.05,1*1.05])
plt.show()


# %%
group='out'
T = list(all_results[group].keys())[0]

plt.figure(figsize=(12, 6))  # Create a single figure with a larger size
method_names = sorted(all_results[group][T].keys())
method_scores = [all_results[group][T][method]['DSC'] for method in method_names]
plt.boxplot(method_scores, positions=range(len(method_scores)), labels=method_names)

plt.xlabel('Method')
plt.ylabel('DSC')
plt.title(f'DSC scores of local cases {translator[group]}')

plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
# plt.ylim([-0.05,1*1.05])
plt.show()


# %% Classification

for group in all_results.keys():
    sens = []
    spec = []
    sens_names = []
    spec_names = []

    
    for T in sorted(all_results[group].keys()):
        for method, result in sorted(all_results[group][T].items()):


            ood_cases = np.array(result['FP/N ood'])
            ood_cases[ood_cases>0] = 1

            healthy_cases =  np.array(result['FP/N healhty'])
            healthy_cases[healthy_cases>0] = 1

            FP = healthy_cases.sum()
            TN = healthy_cases.shape[0] - FP

            TP = ood_cases.sum()
            FN = ood_cases.shape[0] - TP

            print(group, T, method, f"FP={FP}\tTN={TN}\tTP={TP}\tFN={FN}")
            if len(ood_cases) > 0:
                sens.append(TP/(TP+FN))
                sens_names.append(f"{T} {method} (n={len(ood_cases)})")

            if len(healthy_cases) > 0:
                spec.append(TN/(TN+FP))
                spec_names.append(f"{T} {method} (n={len(healthy_cases)})")       

    fig, ax =plt.subplots(1,2, figsize=(16,3))
    ax[0].bar(range(len(sens)), sens)
    ax[0].set_xticks(range(len(sens)), sens_names, rotation=-90)
    ax[0].set_title(f'Global detection sensitivity {translator[group]}')

    ax[1].bar(range(len(spec)), spec)
    ax[1].set_xticks(range(len(spec)), spec_names, rotation=-90)
    ax[1].set_title(f'Global detection specificity {translator[group]}')
    plt.show()     



            

            
            




# %%
