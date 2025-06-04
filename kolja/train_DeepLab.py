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
from dataset_class import ETHMugsDataset
from utils import *
from DeepLab import DeepLab

def load_config(config_path):
    """Lädt die Konfiguration aus einer YAML-Datei."""
    t = time.perf_counter()
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"[Time]: load_config: Config is loaded in {time.perf_counter()-t:.3f} s")
    return config

def plot_training_progress(train_losses, val_ious):
    """Plottet den Trainingsverlauf."""
    t = time.perf_counter()
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_ious, label='Validation IoU')
    plt.title('Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    print(f"[Time]: plot_training_progress: plot is loaded in  {time.perf_counter()-t:.3f} s")

# Definiert eine Funktion, die das Modell erzeugt.
def build_model():
    # Beschreibt in einem Docstring, dass hier das Modell gebaut wird.
    """Build the model."""
    print(f"[INFO]: build_model: has been started")
    return DeepLab()

# Definiert die Trainingsfunktion mit Speicherorten für Checkpoints und Daten als Argumente.
def train(ckpt_dir: str, train_data_root: str, val_data_root: str, config: dict):
    # === ANSI TERMINAL==
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    t0 = time.perf_counter()
    print(f"[INFO]: train: has been started")

    #1. Prüft, ob eine GPU verfügbar ist und wählt das richtige Device.
    t = time.perf_counter()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[Time]: train: train devive {BOLD}{GREEN}{device}{RESET} is choosen {time.perf_counter()-t:.3f} s")

    #2. Lade das vollständige Trainingsdataset
    t = time.perf_counter()
    full_train_dataset = ETHMugsDataset(root_dir=train_data_root, mode="train")
    print(f"[Time]: train: full_train_dataset is loaded as class ETHMugsDataset {time.perf_counter()-t:.3f} s")


    #3. Train-Validation Split
    t = time.perf_counter()
    train_len = int(0.8 * len(full_train_dataset))
    val_len = len(full_train_dataset) - train_len
    train_dataset, val_dataset = random_split(full_train_dataset, [train_len, val_len])
    print(f"[Time]: train: full_train_dataset has been splitted into train and val {time.perf_counter()-t:.3f} s")

    #4. Traindata Augmentation
    #====================================================================================
    print(f"{GREEN}[ATTENTION]: train: The Augmentation is still not implemented{RESET}")
    #====================================================================================

    #5. Erstelle DataLoader für Trainings- und Validierungsdaten
    t = time.perf_counter()
    train_loader = DataLoader(train_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    print(f"[Time]: train: train & val Dataloading is done {time.perf_counter()-t:.3f} s")

     #6. Traindata Augmentation
    #====================================================================================
    print(f"{GREEN}[ATTENTION]: train: data flattening is still not implemented{RESET}")
    """
    for images_batch, masks_batch in train_loader:
    B, V, C, H, W = images_batch.shape
    images_flat = images_batch.view(B * V, C, H, W)
    masks_flat  = masks_batch.view (B * V, 1, H, W)
    """
    #====================================================================================

    #7. Lade Testdaten
    t = time.perf_counter()
    test_dataset = ETHMugsDataset(root_dir=val_data_root, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    print(f"[Time]: train: test_dataset is loaded and ready to use {time.perf_counter()-t:.3f} s")

    #8. Erstellt den Ausgabeordner, falls dieser noch nicht existiert.
    t = time.perf_counter()
    os.makedirs(config['paths']['out_dir'], exist_ok=True)
    # Gibt aus, wohin die Masken gespeichert werden.
    print(f"[INFO]: train: will save the predicted segmentation masks to {config['paths']['out_dir']} {time.perf_counter()-t:.3f} s")

    #9. Model
    t = time.perf_counter()
    model = build_model()
    model.to(device) # GPU oder CPU
    print(f"[Time]: train: model is build and transferred to the {BOLD}{GREEN}{device}{RESET} {time.perf_counter()-t:.3f} s")

    # Definiert die Loss-Funktion für binäre Klassifikation (Segmentierung).
    t = time.perf_counter()
    criterion = torch.nn.BCELoss()
    print(f"[Time]: train: loss function is defined as 'criterion' {time.perf_counter()-t:.3f} s")

    # Initialisiert den Adam-Optimizer mit Lernrate.
    t = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['hyperparameters']['learning_rate'])
    print(f"[Time]: train: Adam_Optimizer is defined as 'optimizer' {time.perf_counter()-t:.3f} s")

    # Erstellt Scheduler, der die Lernrate basierend auf der Validierungs-IoU anpasst.
    t = time.perf_counter()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    print(f"[Time]: train: Adam_Optimizer is defined as 'optimizer' {time.perf_counter()-t:.3f} s")

    train_losses = []
    val_ious = []

    # Schleife über alle Trainingsepochen.
    print("[INFO]: Starting training...")
    for epoch in range(config['hyperparameters']['num_epochs']):
        model.train() # Setzt das Modell in den Trainingsmodus

        print('****************************')
        print(epoch)
        print('****************************')

        epoch_loss = 0
        # Schleife über alle Trainings-Batches.
        for i, (image, gt_mask) in enumerate(train_loader):
            image = image.to(device)
            gt_mask = gt_mask.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, gt_mask.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_losses.append(loss.item())

            print("Training Loss: {}".format(loss.data.cpu().numpy()),
                  "- IoU: {}".format(compute_iou(output.data.cpu().numpy() > 0.5, gt_mask.data.cpu().numpy())))

        avg_epoch_loss = epoch_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_iou = 0
        with torch.no_grad():
            for image, gt_mask in val_loader:
                image = image.to(device)
                gt_mask = gt_mask.to(device)
                output = model(image)
                val_iou += compute_iou((output > 0.5).cpu().numpy(), gt_mask.cpu().numpy())

        val_iou /= len(val_loader)
        val_ious.append(val_iou)
        print(f"Validation IoU: {val_iou}")

        lr_scheduler.step(val_iou)

        # Speichert das Modell nach jeder Epoche als Checkpoint.
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pth"))

    plot_training_progress(train_losses, val_ious)

    # Train the model on the full dataset after determining the parameters
    print("[INFO]: Training the model on the full dataset...")
    full_train_loader = DataLoader(full_train_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(config['hyperparameters']['num_epochs']):
        model.train()

        print('****************************')
        print(f"Full dataset training - Epoch {epoch}")
        print('****************************')

        for image, gt_mask in full_train_loader:
            image = image.to(device)
            gt_mask = gt_mask.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, gt_mask.float())
            loss.backward()
            optimizer.step()

            print("Training Loss: {}".format(loss.data.cpu().numpy()),
                  "- IoU: {}".format(compute_iou(output.data.cpu().numpy() > 0.5, gt_mask.data.cpu().numpy())))

    # Test Daten
    model.eval() # Setzt das Modell in den Evaluierungsmodus.
    image_ids = [] # Initialisiert eine Liste für Bild-IDs.
    pred_masks = [] # Initialisiert eine Liste für vorhergesagte Masken.

    with torch.no_grad():
        # Schleife über alle Testbilder im DataLoader.
        for i, (image, _) in enumerate(test_loader):
            image = image.to(device)
            test_output = model(image)
            test_output = torch.nn.Sigmoid()(test_output) # binäre Segmentierung in ETH Tasse und Hintergrund

            # Schwellenwert auf 0.5: Erzeugt Binärmaske als NumPy-Array.
            pred_mask = (test_output > 0.5).squeeze().cpu().numpy()
            # Wandelt die Binärmaske in ein Graustufenbild (PIL Image) um.
            pred_mask_image = Image.fromarray((pred_mask * 255).astype('uint8'))
            pred_mask_image.save(os.path.join(config['paths']['out_dir'], f"{str(i).zfill(4)}_mask.png"))

            image_ids.append(str(i).zfill(4))
            pred_masks.append(pred_mask)

    # Speichert alle Vorhersagen im Submission-Format als CSV-Datei.
    save_predictions(image_ids=image_ids, pred_masks=pred_masks, save_path=os.path.join(config['paths']['out_dir'], 'submission.csv'))
    print(f"[INFO]: Predictions saved to {os.path.join(config['paths']['out_dir'], 'submission.csv')}")

if __name__ == "__main__":
    # Erstellt einen Argumentparser für Kommandozeilenargumente.
    parser = argparse.ArgumentParser(description="SML Project 2.")
    # Fügt Argument für den Konfigurationspfad hinzu.
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to the config file.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S") # Formatiert das Datum als String für den Ordnernamen.
    ckpt_dir = os.path.join(config['paths']['ckpt_dir'], dt_string) # Verbindet Checkpoint-Pfad und Zeitstring zu neuem Verzeichnis.
    os.makedirs(ckpt_dir, exist_ok=True) # Erstellt das Checkpoint-Verzeichnis, falls nicht vorhanden.
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)
    print("PLEASE ARCHIVE PREDICTIONS FOLDER AND RENAME THE FOLDER TO predictions{number}.csv")

    # Setzt den Trainingsdaten-Ordner.
    train_data_root = os.path.join(config['paths']['data_root'], "train_data")
    print(f"[INFO]: Train data root: {train_data_root}")

    val_data_root = os.path.join(config['paths']['data_root'], "test_data")
    print(f"[INFO]: Test data root: {val_data_root}")

    train(ckpt_dir, train_data_root, val_data_root, config) # Startet das Training mit den definierten Pfaden.
