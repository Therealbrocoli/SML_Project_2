"""ETH Mugs Dataset."""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import IMAGE_SIZE, load_mask

class ETHMugsDataset(Dataset):
    def __init__(self, root_dir, mode="train", use_aug=True):
        """
        Args:
            root_dir (str): Path to the root directory of the dataset (contains rgb/ und masks/).
            mode (str): 'train', 'val' oder 'test'.
            use_aug (bool): Ob Datenaugmentation verwendet werden soll (nur sinnvoll bei train).
        """
        self.mode = mode
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(self.root_dir, "rgb")
        self.mask_dir = os.path.join(self.root_dir, "masks")
        self.image_paths = sorted([os.path.join(self.rgb_dir, f) for f in os.listdir(self.rgb_dir) if f.endswith("_rgb.jpg")])

        if self.mode != "test":
            self.mask_paths = sorted([os.path.join(self.mask_dir, f) for f in os.listdir(self.mask_dir) if f.endswith("_mask.png")])
            assert len(self.image_paths) == len(self.mask_paths)
        else:
            self.mask_paths = None

        # Augmentations und Transforms
        if mode == "train" and use_aug:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, border_mode=0, p=0.7),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(p=0.3),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])
        print("[INFO] Dataset mode:", mode)
        print("[INFO] Number of images in ETHMugsDataset:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        if self.mode == "test":
            transformed = self.transform(image=image)
            return transformed["image"]
        else:
            mask = np.array(Image.open(self.mask_paths[idx]))
            if mask.max() > 1:
                mask = mask // 255
            mask = mask.astype("float32")
            transformed = self.transform(image=image, mask=mask)
            # Maske ist shape [H,W] -> [1,H,W]
            mask_tensor = transformed["mask"].unsqueeze(0)
            return transformed["image"], mask_tensor
