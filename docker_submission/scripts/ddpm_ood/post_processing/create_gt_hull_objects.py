# %%
import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay

# %%
def calculate_convex_hull(coords, shape):
    # Calculate the convex hull of the given coordinates using scipy.spatial.ConvexHull
    try:
        hull = ConvexHull(coords)
        

        deln = Delaunay(coords[hull.vertices])

        # Instead of allocating a giant array for all indices in the volume,
        # just iterate over the slices one at a time.
        idx_2d = np.indices(shape[1:], np.int16)
        idx_2d = np.moveaxis(idx_2d, 0, -1)

        idx_3d = np.zeros((*shape[1:], len(shape)), np.int16)
        idx_3d[:, :, 1:] = idx_2d
        
        mask = np.zeros(shape)
        for z in range(shape[0]):
            idx_3d[:,:,0] = z
            s = deln.find_simplex(idx_3d)
            mask[z, (s != -1)] = 1
    except Exception as e:
        print('\t Convex hull not created', e)
        for i in range(coords.shape[1]):
            print(i, coords[:,i].min(), coords[:,i].max())

        mask= None

    return mask

def process_ground_truths(root):
    # Define source and destination
    pixel_root = os.path.join(root, 'pixel')
    dest = os.path.join(root, 'hull')
    os.makedirs(dest, exist_ok=True)


    for fname in os.listdir(pixel_root):
        print(fname)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pixel_root, fname)))
        
        if gt.sum()>0:
    
            # Perform connected component analysis using labeling
            labeled_array, num_objects = ndimage.label(gt)

            if num_objects > 2:
                struct = np.ones((3,3,3)).astype(bool)
                x1 = ndimage.binary_dilation(gt, struct)
                x2 = ndimage.binary_erosion(x1, struct)

                labeled_array, num_objects = ndimage.label(x2)
    
            # Iterate over each labeled object and calculate its properties
            all_components = []
            for obj_id in range(1, num_objects + 1):
                obj_mask = labeled_array == obj_id
                coords = np.array(np.where(obj_mask)).T

                # Calculate the convex hull for the GT object
                hull_mask = calculate_convex_hull(coords, gt.shape)
                
                if hull_mask is None:
                    all_components.append(obj_mask.astype(gt.dtype))
                else:
                    all_components.append( hull_mask.astype(gt.dtype))

            all_components = np.stack(all_components)
        
            sitk.WriteImage(sitk.GetImageFromArray(all_components), os.path.join(dest, fname))

            
            print('\t Saved', os.path.join(dest, fname), all_components.shape)
            
    
if __name__ == '__main__':
    # root = '/home/bme001/shared/mood/data/brain_val/brain_val_transformed_label'
    # root = '/home/bme001/shared/mood/data/brain_toy/toy_label'

    root = '/home/bme001/shared/mood/data/abdom_val/abdom_val_transformed_label'
    # root = '/home/bme001/shared/mood/data/abdom_toy/toy_label'

    process_ground_truths(root)


    
# %%
#