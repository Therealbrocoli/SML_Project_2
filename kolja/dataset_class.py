import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms, transforms as T
import torchvision.transforms.functional as F

from utils import IMAGE_SIZE  #(W, H)


class ETHMugsDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        # 1. Standards werden definiert im Dataset
        self.mode = mode
        self.root_dir = root_dir
        self.mean = [0.427, 0.419, 0.377]
        self.std = [0.234, 0.225, 0.236]

        # 2. Was wird gemacht wenn man von "train" data redet.
        if self.mode == "train":
            self.rgb_dir = os.path.join(self.root_dir, "rgb")
            self.mask_dir = os.path.join(self.root_dir, "masks")
            self.image_paths = [f for f in os.listdir(self.rgb_dir) if f.endswith(".jpg")]

            self.transform = transforms.Compose([
                transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),              # Wandelt das PIL-Image in einen PyTorch-Tensor um.
                transforms.Normalize(self.mean, self.std), # Normalisiert die Bildkanäle.
            ])

        if self.mode == "test":
            self.rgb_dir = os.path.join(self.root_dir, "rgb")
            self.image_paths = [f for f in os.listdir(self.rgb_dir) if f.endswith(".jpg")]
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

        print("[INFO] Dataset mode:", mode)  # Gibt aus, welcher Mode genutzt wird (train/test).
        print("[INFO] Number of images in the ETHMugDataset:", len(self.image_paths))  # Gibt die Anzahl der Bilder im Datensatz aus.

    def __len__(self):  # Ermöglicht len(dataset); gibt die Anzahl der Bilder zurück.
        return len(self.image_paths)  # Gibt die Länge der Bildliste zurück.

    def __getitem__(self, idx: int):

        IMG_NAME = os.path.join(self.rgb_dir, self.image_paths[idx])  # Baut Pfad zum Bild auf.
        image = Image.open(IMG_NAME).convert('RGB')  # Öffnet das Bild als RGB.

        if self.mode == "train":  # Nur im Training werden echte Masken geladen.

            img_base_name = os.path.splitext(self.image_paths[idx])[0].split('_rgb')[0]  # Extrahiert den Basisnamen ohne Suffix und Extension.

            MASK_NAME = os.path.join(self.mask_dir, f"{img_base_name}_mask.png")  # Sucht zugehörigen Maskenpfad.

            if not os.path.exists(MASK_NAME):  # Prüft, ob die Maske existiert.
                raise FileNotFoundError(f"The mask file {MASK_NAME} does not exist.")  # Fehler, falls Maske fehlt.
            mask = Image.open(MASK_NAME).convert('L')  # Öffnet die Maske als Graustufenbild.

            image = self.transform(image)

        elif self.mode == "val":
            mask = Image.open(MASK_NAME).convert('L')  # Öffnet die Maske als Graustufenbild.
        else:
            mask = torch.zeros((1, 252, 376), dtype=torch.float32)  # Gibt Dummy-Maske zurück, falls nicht train.

        return image, mask  # Gibt Bild und Maske (beides als Tensor) zurück.