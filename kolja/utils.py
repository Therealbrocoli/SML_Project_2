"""Utility functions."""

import numpy as np
import pandas as pd
import time
import os
import torch
from torchvision import transforms
from PIL import Image

IMAGE_SIZE = (252, 378)

def mean_std():
    t0 = time.perf_counter()
    # === PARAMETER ===
    print(f"[INFO]: start mean_std calculations for augmentation")
    t = time.perf_counter()
    rgb_folder = "datasets/train_data/rgb"
    IMAGE_SIZE = (252, 378)
    print(f"[TIME]: mean_std: variable defintion {time.per_counter()-t:.3f}s")

    # 1. Transform für Statistik: Resize → ToTensor (skaliert [0,255] → [0,1])
    t = time.perf_counter()
    stat_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),  # ergibt Tensor mit shape (3, H, W), Werte ∈ [0,1]
    ])
    print(f"[TIME]: mean_std: transformation for calculation {time.per_counter()-t:.3f}s")

    # 2. Arrays für Summen und Quadratsummen initialisieren
    t = time.perf_counter()
    sum_channels    = torch.zeros(3)
    sum_sq_channels = torch.zeros(3)
    total_pixels    = 0
    print(f"[TIME]: mean_std: Arrays für Summen und Quadratsummen initialisieren {time.per_counter()-t:.3f}s")

    # 3. Über alle Bilder iterieren und Statistiken aufsammeln
    t = time.perf_counter()
    for fname in os.listdir(rgb_folder):
        if not fname.lower().endswith(".jpg"):
            continue

        path = os.path.join(rgb_folder, fname)
        img  = Image.open(path).convert("RGB")
        tensor = stat_transform(img)  # Shape: (3, H, W)

        # Anzahl Pixel im Bild
        _, H, W = tensor.shape
        num_pix = H * W
        total_pixels += num_pix

        # Kanalsummen und -quadratsummen aufsummieren
        # tensor.sum(dim=(1,2)) ist ein 3-Tupel: [Summe_R, Summe_G, Summe_B]
        sum_channels    += tensor.sum(dim=(1, 2))
        sum_sq_channels += (tensor ** 2).sum(dim=(1, 2))
    print(f"[TIME]: mean_std: über alle Bilder iteriert {time.per_counter()-t:.3f}s")

    # 4. Mittelwert und Standardabweichung berechnen
    t = time.perf_counter()
    mu    = sum_channels    / total_pixels                  # Tensor der Länge 3
    var   = (sum_sq_channels / total_pixels) - (mu ** 2)    # Kanal-Varianz
    sigma = torch.sqrt(var)                                 # Kanal-StdDev
    print(f"[TIME]: mean_std: mean und std fertig berechnet in torch format{time.per_counter()-t:.3f}s")

    # 5. In Python-Listen für Normalize umwandeln
    t = time.perf_counter()
    mean_values = mu.tolist()       # z.B. [0.48, 0.43, 0.39]
    std_values  = sigma.tolist()    # z.B. [0.24, 0.25, 0.23]
    print(f"[TIME]: mean_std: ready für die Class ETHMugsDataset transformiert {time.per_counter()-t:.3f}s")

    print("Berechnetes mean:", mean_values)
    print("Berechnete std: ", std_values)
    print(f"[TIME]: mean_std: calculation dauer{time.per_counter()-t:.3f}s")


    return mean_values, std_values

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

print(mean_std())

