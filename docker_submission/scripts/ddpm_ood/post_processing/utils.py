import numpy as np
import SimpleITK as sitk
from scipy.spatial import ConvexHull, Delaunay
from scipy import ndimage


#%%
def dsc_score(pred_array, target_array):
    """
    Calculate the Dice Similarity Coefficient (DSC) score between two 3D NumPy arrays.
    
    Parameters:
        pred_array (np.ndarray): The predicted binary segmentation mask (0 or 1).
        target_array (np.ndarray): The ground truth binary segmentation mask (0 or 1).
        
    Returns:
        float: The Dice Similarity Coefficient (DSC) score.
    """
    # Check if the input arrays have the same shape
    if pred_array.shape != target_array.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    # Flatten the 3D arrays to 1D
    pred_flat = pred_array.flatten()
    target_flat = target_array.flatten()
    
    # Calculate the intersection and union between the predicted and target masks
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)
    
    # Calculate the Dice Similarity Coefficient
    dice_coefficient = (2.0 * intersection) / (union + 1e-8)  # Add a small epsilon to avoid division by zero
    
    return dice_coefficient



def ssim_pad(array, pad=3):
    # Create a new array with the same shape and data type as the input array
    new_array = np.ones(array.shape, dtype=array.dtype)
    
    # Check the dimensionality of the input array
    if array.ndim == 3:
        # If the input array is 3-dimensional, copy the values from the input array to the new array
        # but leave a padding of size 'pad' on each side
        new_array[pad:-pad, pad:-pad, pad:-pad] = array[pad:-pad, pad:-pad, pad:-pad]
        
    elif array.ndim == 2:
        # If the input array is 2-dimensional, copy the values from the input array to the new array
        # but leave a padding of size 'pad' on each side
        new_array[pad:-pad, pad:-pad] = array[pad:-pad, pad:-pad]
        
    return new_array


def segment(input_image, r):
    # Step 1: Invert intensity to convert foreground to background and vice versa
    image = sitk.InvertIntensity(sitk.GetImageFromArray(input_image))

    # Step 2: Apply Otsu thresholding to obtain a binary mask
    mask = sitk.OtsuThreshold(image)
    
    # Step 3: Dilate the binary mask to merge nearby regions
    dil_mask = sitk.BinaryDilate(mask, (r, r, r))
    
    # Step 4: Apply connected component analysis to label connected regions
    component_image = sitk.ConnectedComponent(dil_mask)
    
    # Step 5: Sort the connected components by object size and re-label them
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    
    # Step 6: Keep only the largest component (background) and convert it to binary
    largest_component_binary_image = sorted_component_image == 1
    
    # Step 7: Perform binary morphological closing to fill gaps and smooth the mask
    mask_closed = sitk.BinaryMorphologicalClosing(largest_component_binary_image, (r, r, r))
    
    # Step 8: Dilate the closed mask to ensure the mask covers the full object
    dilated_mask = sitk.BinaryDilate(mask_closed, (r, r, r))
    
    # Step 9: Fill holes inside the mask
    filled_mask = sitk.BinaryFillhole(dilated_mask)
    
    # Convert the final filled mask to a numpy array and return
    return sitk.GetArrayFromImage(filled_mask)


def find_connected_components(binary_array, exclude_threshold=8):
    # Perform connected component analysis using labeling
    labeled_array, num_objects = ndimage.label(binary_array)
    
    # Initialize a list to store connected components (objects) and their properties
    connected_components = []
    
    # Iterate over each labeled object and calculate its properties
    for obj_id in range(1, num_objects + 1):
        obj_mask = labeled_array == obj_id
        size = obj_mask.sum()
        if size > exclude_threshold:
            coords = np.array(np.where(obj_mask)).T
            center_of_mass = coords.mean(axis=0)
            properties = {'center_of_mass': center_of_mass,
                          'size': obj_mask.sum()}
        
            connected_components.append(properties)
    return connected_components

def match_objects(gt_objects, pred_objects):
    # Initialize TP, FP, and FN counts to zero
    tp, fp, fn = 0, 0, 0
    for gt_object in gt_objects:
        matched = False
        for pred_object in pred_objects:
            c = np.round(pred_object['center_of_mass']).astype(int)
            in_hull = gt_object[c[0], c[1], c[2]].astype(bool)

            bigger_than_half = (pred_object['size'] > gt_object.sum()/2)
            smaller_than_double = (pred_object['size'] < gt_object.sum()*2)
            size_ok = (bigger_than_half and smaller_than_double)

            if in_hull == True and size_ok == True:
                tp += 1
                matched = True
                break
        
        if not matched:
            fn += 1
    # Count the remaining unmatched prediction objects as FPs
    fp = len(pred_objects) - tp

    
    return tp, fp, fn


def calculate_challenge_metrics(pred_array, gt_objects, threshold=0.5):
    # Determine the binarization threshold using the toy-dataset
    binary_pred_array = (pred_array >= threshold).astype(int)
    
    # Find connected components 
    prediction_objects = find_connected_components(binary_pred_array)

    sum_per_pred_obj = [float(p['size']) for p in prediction_objects]
    
    # Match GT and prediction objects to get TP, FP, FN for the whole dataset
    tp, fp, fn = match_objects(gt_objects, prediction_objects)
    
     # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return tp, fp, fn, f1_score, precision, recall, sum_per_pred_obj

# # Main script
# if __name__ == "__main__":
#     # Load the ground truth and prediction images
#     gt_folder = '/home/s144823/gt/'
#     pred_folder = '/home/s144823/pred/'
#     image_names = ['toy1.nii.gz', 'toy2.nii.gz', 'toy3.nii.gz']  # Replace with actual image names
#     threshold = 0.5


#     # Assuming the images are 3D, load them and calculate the F1 score
#     for image_name in image_names:
#         gt_image = sitk.ReadImage(gt_folder + image_name)
#         pred_image = sitk.ReadImage(pred_folder + image_name)
        
#         gt_array = sitk.GetArrayFromImage(gt_image)
#         pred_array = sitk.GetArrayFromImage(pred_image)
        
#         f1_score = calculate_f1_score(gt_array, pred_array)
#         print(f"F1 Score for {image_name}: {f1_score}")

# # %%
# a