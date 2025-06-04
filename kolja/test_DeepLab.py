import argparse
import os
import random
import numpy as np
from datetime import datetime
import torch
import time
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from dataset_DeepLab import ETHMugsDataset
from utils import *
from DeepLab import DeepLab

import glob

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Laden Sie die Konfiguration und die Gewichte
def load_config(config_path):
    """Lädt die Konfiguration aus einer YAML-Datei."""
    t = time.perf_counter()
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"[TIME]: load_config: Config is loaded in {time.perf_counter()-t:.3f} s")
    return config
config = load_config('config_DeepLab.yaml')  # Stellen Sie sicher, dass der Pfad korrekt ist
model = DeepLab()  # Erstellen Sie eine Instanz Ihres Modells

#Lade die Checkpoints

checkpoint_root = "checkpoints"
subdirs = sorted([
    d for d in os.listdir(checkpoint_root)
    if os.path.isdir(os.path.join(checkpoint_root, d))
])

if not subdirs:
    raise FileNotFoundError("Keine Unterordner in 'checkpoints/' gefunden.")

# 2. Wähle das zuletzt erzeugte Verzeichnis (zeitlich sortiert)
########## Vorletzten Checkpoint
latest_dir = os.path.join(checkpoint_root, subdirs[-2])
################# 

checkpoint_files = sorted(
    glob.glob(os.path.join(latest_dir, "epoch_*.pth")),
    key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
)
if not checkpoint_files:
    raise FileNotFoundError(f"Keine epoch_*.pth Dateien gefunden in {latest_dir}")

checkpoint_path = checkpoint_files[-1]
print(f"[INFO] Lade Checkpoint: {checkpoint_path}")


model.load_state_dict(torch.load(checkpoint_path, weights_only=True))  # Load the weights  
model.eval()  # Setzen Sie das Modell in den Evaluierungsmodus

# Erstellen Sie den Test-Datensatz und den DataLoader
test_dataset = ETHMugsDataset(root_dir=config['paths']['data_root'], mode="test")
test_loader = DataLoader(test_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

# Durchlaufen Sie die Testdaten und führen Sie Vorhersagen durch
image_ids = []
pred_masks = []

with torch.no_grad():
    for i, (image, _) in enumerate(test_loader):
        image = image.to(device)
        test_output = model(image)
        test_output = torch.sigmoid(test_output)  # Stellen Sie sicher, dass die Ausgabe im Bereich [0, 1] liegt

        pred_mask = (test_output > 0.5).squeeze().cpu().numpy()

        # Stellen Sie sicher, dass pred_mask 2D ist
        if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
            pred_mask = pred_mask.squeeze(0)

        # Konvertieren Sie die Binärmaske in ein PIL-Bild
        pred_mask_image = Image.fromarray((pred_mask * 255).astype('uint8'))
        pred_mask_image.save(os.path.join(config['paths']['out_dir'], f"{str(i).zfill(4)}_mask.png"))

        image_ids.append(str(i).zfill(4))
        pred_masks.append(pred_mask)

# Speichern Sie die Vorhersagen im gewünschten Format
save_predictions(image_ids=image_ids, pred_masks=pred_masks, save_path=os.path.join(config['paths']['out_dir'], 'submission.csv'))
