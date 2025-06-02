"""utils.py"""

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_SIZE = (252, 378)  # (width, height), nur als Orientierung

def load_mask(mask_path: str) -> np.ndarray:
    """
    Lädt eine Segmentierungsmaske (PNG) und gibt ein 2D-array (0 oder 1) zurück.
    """
    mask = np.asarray(Image.open(mask_path)).astype(int)
    if mask.max() > 1:
        mask = mask // 255
    return mask

def mask_to_rle(mask: np.ndarray) -> str:
    """
    Konvertiert eine binäre Maske (2D) in RLE (Run-Length-Encoding), spaltenweise (Fortran-Order).
    Rückgabe: Space-separierter String (z.B. "3 5 10 2 …").
    """
    pixels = mask.flatten(order='F')  # Spaltenweise flatten
    pixels = np.concatenate([[0], pixels, [0]])  # Padding
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]
    return ' '.join(str(x) for x in runs)

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    """
    Berechnet IoU (Intersection over Union) für binäre Masken (0/1).
    """
    intersection = (pred_mask & gt_mask).astype(float).sum()
    union = (pred_mask | gt_mask).astype(float).sum()
    iou = (intersection + eps) / (union + eps)
    return iou

def save_predictions(image_ids: list[str], pred_masks: list[np.ndarray], save_path: str = 'submission.csv'):
    """
    Schreibt eine CSV für Kaggle-Submission:
      - ImageId: z.B. "0001"
      - EncodedPixels: RLE-String
    """
    assert len(image_ids) == len(pred_masks), "Anzahl IDs und Masken muss gleich sein."
    predictions = {'ImageId': [], 'EncodedPixels': []}
    for img_id, mask in zip(image_ids, pred_masks):
        rle = mask_to_rle(mask)
        predictions['ImageId'].append(img_id)
        predictions['EncodedPixels'].append(rle)
    pd.DataFrame(predictions).to_csv(save_path, index=False)
    print(f"[INFO] Submission-CSV gespeichert: {save_path}")
