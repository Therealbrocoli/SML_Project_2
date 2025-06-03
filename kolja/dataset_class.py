"""ETH Mugs Dataset."""  
import os  
from PIL import Image  # Importiert PIL, um Bilder zu öffnen und zu bearbeiten.
import torch  # Importiert PyTorch, das Framework für Deep Learning.
import random

from torch.utils.data import Dataset  # Importiert die Dataset-Basisklasse für eigene Datensätze.
from torchvision import transforms  # Importiert Bildtransformationen und Augmentationen.

from utils import * #Importiert eine Bildgrößen-Konstante und eine Funktion (hier nicht genutzt).

"""
root_dir = "dataset/train_data"
root_dir = "dataset/test_data"
"""

class ETHMugsDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        # 1. Standards werden definiert im Dataset
        self.mode = mode
        self.root_dir = root_dir

        # 2. Was wird gemacht wenn man von "train" data redet.
        if self.mode == "train":
            self.rgb_dir = os.path.join(self.root_dir, "rgb")
            self.mask_dir = os.path.join(self.root_dir, "masks")
            self.mean, self.std = [0.427, 0.419, 0.377],[0.234, 0.225, 0.236] # = mean_std()
            self.image_paths = [f for f in os.listdir(self.rgb_dir) if f.endswith(".jpg")]

            self.transform1 = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),              # Wandelt das PIL-Image in einen PyTorch-Tensor um.
                transforms.Normalize(self.mean, self.std), # Normalisiert die Bildkanäle.
            ])
            self.transform2 = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),      # Zufällige Drehung (max ±10 Grad).
                transforms.Resize(IMAGE_SIZE),      
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
            self.transform3 = transforms.Compose([
                transforms.ColorJitter(contrast=0.5, saturation=0.5, hue=0.1),
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
            self.transform4 = transforms.Compose([
                transforms.RandomSolarize(threshold=128, p=0.3),
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
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

    def __getitem__(self, idx: int):  # Holt das Bild + Maske zum gegebenen Index (dataset[idx]).
        img_name = os.path.join(self.rgb_dir, self.image_paths[idx])  # Baut Pfad zum Bild auf.
        image = Image.open(img_name).convert('RGB')  # Öffnet das Bild als RGB.

        if self.mode == "train":  # Nur im Training werden echte Masken geladen.
            img_base_name = os.path.splitext(self.image_paths[idx])[0].split('_rgb')[0]  # Extrahiert den Basisnamen ohne Suffix und Extension.
            mask_name = os.path.join(self.mask_dir, f"{img_base_name}_mask.png")  # Sucht zugehörigen Maskenpfad.
            if not os.path.exists(mask_name):  # Prüft, ob die Maske existiert.
                raise FileNotFoundError(f"The mask file {mask_name} does not exist.")  # Fehler, falls Maske fehlt.
            mask = Image.open(mask_name).convert('L')  # Öffnet die Maske als Graustufenbild.
            mask = self.mask_transform(mask)  # Transformiert die Maske.
        else:
            mask = torch.zeros((1, 252, 376), dtype=torch.float32)  # Gibt Dummy-Maske zurück, falls nicht train.

        if self.transform:  # Wenn Transformationen gesetzt sind.
            image = self.transform(image)  # Transformiert das Bild.

        return image, mask  # Gibt Bild und Maske (beides als Tensor) zurück.