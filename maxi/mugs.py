"""mugs.py"""

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ETHMugsDataset(Dataset):
    """
    Lädt ETH-Mug-Bilder und ihre Masken.
    - maxi/train_data/rgb   (RGB-Bilder, z.B. 0001_rgb.jpg)
    - maxi/train_data/masks (Masken,   z.B. 0001_mask.png)
    - maxi/test_data/rgb    (Nur RGB,  z.B. 1001_rgb.jpg)
    """
    def __init__(self, root_dir: str, mode="train", use_aug: bool = True):
        """
        Args:
            root_dir (str): Pfad zum Ordner “train_data” oder “test_data” (direkt unter /maxi).
            mode (str): “train”, “val” oder “test”.
            use_aug (bool): Ob bei “train” Augmentationen angewendet werden sollen.
        """
        self.mode = mode
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(self.root_dir, "rgb")
        self.mask_dir = os.path.join(self.root_dir, "masks") if mode != "test" else None

        # Alle RGB-Dateipfade sortiert einlesen („*_rgb.jpg“)
        self.image_paths = sorted([
            os.path.join(self.rgb_dir, f)
            for f in os.listdir(self.rgb_dir)
            if f.endswith("_rgb.jpg")
        ])

        if self.mode != "test":
            # Masken nur im Trainings-/Validierungssatz
            self.mask_paths = sorted([
                os.path.join(self.mask_dir, f)
                for f in os.listdir(self.mask_dir)
                if f.endswith("_mask.png")
            ])
            assert len(self.image_paths) == len(self.mask_paths), \
                f"Anzahl Bilder ({len(self.image_paths)}) != Anzahl Masken ({len(self.mask_paths)})"
        else:
            self.mask_paths = None

        # Definiere Albumentations-Transform:
        # 1) Resize auf 378×252, 2) optional Augmentations, 3) Normalize + ToTensorV2
        if self.mode == "train" and use_aug:
            self.transform = A.Compose([
                A.Resize(378, 252),                   # Höhe=378, Breite=252
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=20,
                    border_mode=0, p=0.7
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(p=0.3),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                ToTensorV2(),
            ])
        else:
            # Kein Augmentieren (Val/Test)
            self.transform = A.Compose([
                A.Resize(378, 252),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                ToTensorV2(),
            ])

        print(f"[INFO] Dataset mode: {mode}  |  Anzahl Bilder: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.mode == "test":
            # Nur das Bild zurückgeben + Bild-ID (Prefix vor "_rgb.jpg")
            augmented = self.transform(image=image)
            img_id = os.path.basename(img_path).split("_")[0]
            return augmented["image"], img_id

        # Im Train/Val-Modus: lade Maske, konvertiere in 0/1, float32
        mask_path = self.mask_paths[idx]
        mask = np.array(Image.open(mask_path))
        if mask.max() > 1:
            mask = mask // 255
        mask = mask.astype("float32")

        augmented = self.transform(image=image, mask=mask)
        img_tensor = augmented["image"]         # Shape: [3, 378, 252]
        mask_tensor = augmented["mask"].unsqueeze(0)  # Shape: [1, 378, 252]
        return img_tensor, mask_tensor
