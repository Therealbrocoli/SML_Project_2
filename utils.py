"""Utility functions."""

import numpy as np
import pandas as pd

from PIL import Image

IMAGE_SIZE = (252, 378)

print("start")
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


import numpy as np

def compute_iou(pred_mask, gt_mask, threshold=0.5):
    # Ensure the masks are in the correct format
    pred_mask = pred_mask > threshold
    gt_mask = gt_mask > threshold

    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    # Calculate IoU
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score



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
print("stop")