"""Utility functions."""

import numpy as np
import pandas as pd

from PIL import Image

IMAGE_SIZE = (252, 378)

def load_mask(mask_path):
    """Loads the segmentation mask from the specified path.

    Inputs:
        mask_path (str): the path from which the segmentation mask will be read.
        It should have the format "/PATH/TO/LOAD/DIR/XXXX_mask.png".

    Outputs:
        mask (np.array): segmentation mask as a numpy array.
    """
    mask = np.asarray(Image.open(mask_path)).astype(int)
    if mask.max() > 1:
        mask = mask // 255
    return mask


def mask_to_rle(mask):
    """
    Convert a binary mask (2D numpy array) to RLE (column-major).
    Returns a string of space-separated values.
    """
    pixels = mask.flatten(order='F')  # Fortran order (column-major)
    pixels = np.concatenate([[0], pixels, [0]])  # pad with zeros to catch transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]  # calculate run lengths
    return ' '.join(str(x) for x in runs)


def compute_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Berechnet den Intersection over Union (IoU) Score für binäre Masken.
    Args:
        pred_mask (np.ndarray): Die vorhergesagte Maske (kann Float-Werte enthalten).
        gt_mask (np.ndarray): Die Ground Truth Maske (kann Float-Werte enthalten).
        threshold (float): Der Schwellenwert zur Binarisierung der Masken.
    Returns:
        float: Der IoU-Score.
    """
    # Ensure the masks are binary (True/False or 0/1) based on the threshold
    pred_mask_binary = pred_mask > threshold
    gt_mask_binary = gt_mask > threshold

    # Calculate intersection and union as boolean arrays
    intersection = np.logical_and(pred_mask_binary, gt_mask_binary)
    union = np.logical_or(pred_mask_binary, gt_mask_binary)

    # Sum the True values (which are 1 when summed)
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)

    # Handle the edge case where the union is zero (both masks are empty)
    if union_sum == 0:
        # If both masks are empty, they perfectly match, so IoU is 1.0
        return 1.0
    
    # Otherwise, calculate IoU normally
    iou_score = intersection_sum / union_sum

    return iou_score

# You would also keep your save_predictions function here if it's in utils.py
# def save_predictions(image_ids, pred_masks, save_path):
#     # ... your existing save_predictions implementation ...


def save_predictions(image_ids, pred_masks, save_path='submission.csv'):
    '''
    image_ids: list of image_ids [0000, 0001, ...]
    pred_masks: binary 2D numpy array
    '''
    assert len(image_ids) == len(pred_masks)
    predictions = {'ImageId': [], 'EncodedPixels': []}
    for i in range(len(image_ids)):
        mask = pred_masks[i]
        mask_rle = mask_to_rle(mask)
        predictions['ImageId'].append(image_ids[i])
        predictions['EncodedPixels'].append(f'{mask_rle}')

    pd.DataFrame(predictions).to_csv(save_path, index=False)