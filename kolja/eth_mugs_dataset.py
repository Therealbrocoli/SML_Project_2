"""ETH Mugs Dataset."""  
import os  
from PIL import Image  # Importiert PIL, um Bilder zu öffnen und zu bearbeiten.
import torch  # Importiert PyTorch, das Framework für Deep Learning.

from torch.utils.data import Dataset  # Importiert die Dataset-Basisklasse für eigene Datensätze.
from torchvision import transforms  # Importiert Bildtransformationen und Augmentationen.

from utils import IMAGE_SIZE, load_mask  # Importiert eine Bildgrößen-Konstante und eine Funktion (hier nicht genutzt).

class ETHMugsDataset(Dataset):  # Definiert eine neue Dataset-Klasse für ETH Mugs, basierend auf PyTorchs Dataset.
    def __init__(self, root_dir, mode="train"):  # Initialisierungsmethode, setzt Pfade und Mode (train/test).
        self.mode = mode  # Merkt sich ob train oder test.
        self.root_dir = root_dir  # Pfad zum Wurzelordner für die Daten (train oder test).

        self.rgb_dir = os.path.join(self.root_dir, "rgb")  # Setzt den Pfad zum RGB-Bildordner.
        self.mask_dir = os.path.join(self.root_dir, "masks")  # Setzt den Pfad zum Maskenordner.

        self.image_paths = [f for f in os.listdir(self.rgb_dir) if f.endswith(".jpg")]  # Liste aller .jpg-Bilder im RGB-Ordner.
        self.image_paths.sort()  # Sortiert die Bildnamen alphabetisch für stabile Reihenfolge.

        # Definiert die Transformationen für die Bilder.
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Zufälliges Spiegeln zur Augmentation.
            transforms.RandomRotation(10),      # Zufällige Drehung (max ±10 Grad).
            transforms.Resize(IMAGE_SIZE),      # Skaliert das Bild auf die gewünschte Zielgröße.
            transforms.ToTensor(),              # Wandelt das PIL-Image in einen PyTorch-Tensor um.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisiert die Bildkanäle.
        ])

        # Definiert die Transformationen für die Masken.
        self.mask_transform = transforms.Compose([
            transforms.Resize((252, 376)),  # Setzt die Maske auf einheitliche Zielgröße.
            transforms.ToTensor(),          # Wandelt die Maske in einen Tensor um (Werte 0–1).
        ])

        print("[INFO] Dataset mode:", mode)  # Gibt aus, welcher Mode genutzt wird (train/test).
        print("[INFO] Number of images in the ETHMugDataset:", len(self.image_paths))  # Gibt die Anzahl der Bilder im Datensatz aus.
###
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
