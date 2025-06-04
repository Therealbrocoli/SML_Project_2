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
from dataset_DeepLab_augmentiert import ETHMugsDataset
from utils import *
from DeepLab import DeepLab

def load_config(config_path):
    """Lädt die Konfiguration aus einer YAML-Datei."""
    t = time.perf_counter()
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"[TIME]: load_config: Config is loaded in {time.perf_counter()-t:.3f} s")
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
    print(f"[TIME]: plot_training_progress: plot is loaded in  {time.perf_counter()-t:.3f} s")

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
    CYAN = "\033[96m"
    RESET = "\033[0m"

    t0 = time.perf_counter()
    print(f"[INFO]: train: has been started")

    #1. Prüft, ob eine GPU verfügbar ist und wählt das richtige Device.
    t = time.perf_counter()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[TIME]: train: train devive {BOLD}{GREEN}{device}{RESET} is choosen {time.perf_counter()-t:.3f} s")

    #2. Lade das vollständige Trainingsdataset
    t = time.perf_counter()
    full_train_dataset = ETHMugsDataset(root_dir=train_data_root, mode="train")
    print(f"[TIME]: train: full_train_dataset is loaded as class ETHMugsDataset {time.perf_counter()-t:.3f} s")


    #3. Train-Validation Split
    t = time.perf_counter()
    train_len = int(0.8 * len(full_train_dataset))
    val_len = len(full_train_dataset) - train_len
    train_dataset, val_dataset = random_split(full_train_dataset, [train_len, val_len])
    print(f"[TIME]: train: full_train_dataset has been splitted into train and val {time.perf_counter()-t:.3f} s")

    #5. Erstelle DataLoader für Trainings- und Validierungsdaten
    t = time.perf_counter()
    train_loader = DataLoader(train_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    print(f"[TIME]: train: train & val Dataloading is done {time.perf_counter()-t:.3f} s")

    #7. Lade Testdaten
    t = time.perf_counter()
    test_dataset = ETHMugsDataset(root_dir=val_data_root, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    print(f"[TIME]: train: test_dataset is loaded and ready to use {time.perf_counter()-t:.3f} s")

    #8. Erstellt den Ausgabeordner, falls dieser noch nicht existiert.
    t = time.perf_counter()
    os.makedirs(config['paths']['out_dir'], exist_ok=True)
    # Gibt aus, wohin die Masken gespeichert werden.
    print(f"[INFO]: train: will save the predicted segmentation masks to {config['paths']['out_dir']} {time.perf_counter()-t:.3f} s")

    #9. Model
    t = time.perf_counter()
    model = build_model()
    model.to(device) # GPU oder CPU
    print(f"[TIME]: train: model is build and transferred to device {BOLD}{GREEN}{device}{RESET} {time.perf_counter()-t:.3f} s")

    #10. Definiert die Loss-Funktion für binäre Klassifikation (Segmentierung).
    t = time.perf_counter()
    criterion = torch.nn.BCELoss()
    print(f"[TIME]: train: binary loss function is defined as 'criterion' for segmentation {time.perf_counter()-t:.3f} s")

    #11. Initialisiert den Adam-Optimizer mit Lernrate.
    t = time.perf_counter()
    lr=float(config['hyperparameters']['learning_rate'])
    optimizer = torch.optim.Adam(model.parameters(), lr)
    print(f"[TIME]: train: Adam_Optimizer is defined as 'optimizer' with learning rate {BOLD}{lr}{RESET}  {time.perf_counter()-t:.3f} s")

    #12. Erstellt Scheduler, der die Lernrate basierend auf der Validierungs-IoU anpasst.
    t = time.perf_counter()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    print(f"[TIME]: train: Adam_Optimizer is defined as 'optimizer' {time.perf_counter()-t:.3f} s")

    #13. Schleife über alle Trainingsepochen.
    t0 = time.perf_counter()
    train_losses = []
    val_ious = []
    epochs = config['hyperparameters']['num_epochs']
    print(f"[INFO]: train: Starting training...")
    for epoch in range(epochs):

        #13.1 Setzt das Modell in den Trainingsmodus
        t = time.perf_counter()
        model.train() 
        print(f"[TIME]: train: trainingsloop: modell wurde in den Trainingsmodus geschaltet{time.perf_counter()-t:.3f} s")

        print('-'*40)
        print(f"{BOLD}EPOCH {epoch}, LR {lr_scheduler.get_last_lr()}{RESET}")
        print('-'*40)

        #13.2 Schleife über alle Trainings-Batches.
        t = time.perf_counter()
        epoch_loss = 0
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

            print(f"{BOLD}[INFO] -> Training Loss: {loss.data.cpu().numpy()} - IoU: {compute_iou(output.data.cpu().numpy() > 0.5, gt_mask.data.cpu().numpy())}{RESET}")
        print(f"[TIME]: train: trainingsloop: training batches done {time.perf_counter()-t:.3f} s")

        #13.2 Validation Loop
        t = time.perf_counter()
        print(f"{BOLD}[INFO]: train: trainingsloop {CYAN}VALDIATION STARTS{RESET}")
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
        print(f"{BOLD}[INFO] -> Validation IoU: {CYAN}{val_iou}{RESET}")
        print(f"[TIME]: train: trainingsloop: Schleife valdidation loop is done{time.perf_counter()-t:.3f} s")
        
        #13.3 Learnig Rate updaten
        t = time.perf_counter()
        lr_scheduler.step(val_iou)
        print(f"[TIME]: train: trainingsloop: learning rate is updated now {time.perf_counter()-t:.3f} s")

        #13.4 Speichert das Modell nach jeder Epoche als Checkpoint.
        t = time.perf_counter()
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pth"))
        print(f"[TIME]: train: trainingsloop: checkpoints has been saved {time.perf_counter()-t:.3f} s")

    print(f"[TIME]: train: TOTAL Training endurance {BOLD}{time.perf_counter()-t0:.3f} s{RESET}")

    #14. Plotting of the training and valdiation
    t = time.perf_counter()
    print(f"[INFO]: train: plotting starts {time.perf_counter()-t:.3f} s")
    plot_training_progress(train_losses, val_ious)
    print(f"[TIME]: train: plot is loaded in {time.perf_counter()-t:.3f} s")

    
    """
    #15. Train the model on the full dataset after determining the parameters
    t = time.perf_counter()
    print(f"[INFO]: Training the model on the full dataset...")
    full_train_loader = DataLoader(full_train_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    epochs = config['hyperparameters']['num_epochs']
    for epoch in range(epochs): 
        t = time.perf_counter()
        model.train()

        print('-'*40)
        print(f"Full dataset training - Epoch {epoch}")
        print('-'*40)

        for image, gt_mask in full_train_loader:
            image = image.to(device)
            gt_mask = gt_mask.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, gt_mask.float())
            loss.backward()
            optimizer.step()

            print(f"{BOLD}[INFO]: -> Training Loss: {loss.data.cpu().numpy()} - IoU: {compute_iou(output.data.cpu().numpy() > 0.5, gt_mask.data.cpu().numpy())}{RESET}")
        print(f"[TIME]: train: trainingsloop: training full batches done {time.perf_counter()-t:.3f} s")
    print(f"[TIME]: train: TOTAL full training endurance {BOLD}{time.perf_counter()-t0:.3f} s{RESET}")

    """
    #16. Test Daten
    t = time.perf_counter()
    model.eval() # Setzt das Modell in den Evaluierungsmodus.
    image_ids = [] # Initialisiert eine Liste für Bild-IDs.
    pred_masks = [] # Initialisiert eine Liste für vorhergesagte Masken.
    print(f"[TIME]: train: initiliasierungen für Test data prediction {time.perf_counter()-t0:.3f} s")

    out_dir = os.path.join('prediction')
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO]: Saving the predicted segmentation masks to {out_dir}")

    #17. Schleife über alle Testbilder im DataLoader.
    t = time.perf_counter()
    with torch.no_grad():
        for i, (image, _) in enumerate(test_loader):
            image = image.to(device)
            # es sind jeweils 32 Bilder #print(f"[DEBUG_IMAGE] Shape: {image.shape}")
            # 32,3,252,378
            test_output = model(image)  # 32,1,252,378
            test_output = torch.nn.Sigmoid()(test_output) # # 32,1,252,378 # binäre Segmentierung in ETH Tasse und Hintergrund

            pred_mask = (test_output > 0.5).squeeze().cpu().numpy()  # reduziert Dimension  # 32,252,378

            # Konvertieren zu uint8 für PIL (0 oder 255)
            pred_mask = (pred_mask * 255).astype(np.uint8) # 32, 252, 378

            for idx in range(pred_mask.shape[0]):
                #print(f"[DEBUG_pred_mask_idx] Shape: {pred_mask[idx].shape}")
                # 4. In PIL-Bild umwandeln und speichern
                pred_mask_image = Image.fromarray(pred_mask[idx])      

                pfad = os.path.join(config['paths']['out_dir'], f"{str(i).zfill(4)}_mask.png")
                pred_mask_image.save(pfad)

                image_ids.append(str(i).zfill(4))
                pred_masks.append(pred_mask)
    print(f"[TIME]: train: erstellen Test Masken DONE {time.perf_counter()-t:.3f} s")

    # Speichert alle Vorhersagen im Submission-Format als CSV-Datei.
    t = time.perf_counter()
    save_predictions(image_ids=image_ids, pred_masks=pred_masks, save_path=os.path.join(config['paths']['out_dir'], 'submission.csv'))
    print(f"[INFO]: train: Predictions saved to {os.path.join(config['paths']['out_dir'], 'submission.csv')}")


if __name__ == "__main__":
     # === ANSI TERMINAL==
    BOLD = "\033[1m"
    RESET = "\033[0m"

    t = time.perf_counter()
    print(f"{BOLD}Meine Lieben es ist mir eine Freude sie begrüssen zu dürfen wir beginnen...{RESET}")
    # Erstellt einen Argumentparser für Kommandozeilenargumente.
    
    parser = argparse.ArgumentParser(description="SML Project 2.")
    print(f"[TIME]: Erstellen eines Argumentparser für Kommandozeilenargumente. {time.perf_counter()-t:.3f} s")


    #2. Fügt Argument für den Konfigurationspfad hinzu.
    t = time.perf_counter()
    parser.add_argument(
        "-c",
        "--config",
        default="config_DeepLab.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S") # Formatiert das Datum als String für den Ordnernamen.
    ckpt_dir = os.path.join(config['paths']['ckpt_dir'], dt_string) # Verbindet Checkpoint-Pfad und Zeitstring zu neuem Verzeichnis.
    os.makedirs(ckpt_dir, exist_ok=True) # Erstellt das Checkpoint-Verzeichnis, falls nicht vorhanden.
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)
    print(f"[TIME]: Weitere Konfigurationen done für den checkpoints Ordner {time.perf_counter()-t:.3f} s")

    #3. Setzt den Trainingsdaten-Ordner.
    t = time.perf_counter()
    train_data_root = os.path.join(config['paths']['data_root'], "train_data")
    print(f"[INFO]: Train data root: {train_data_root}")
    val_data_root = os.path.join(config['paths']['data_root'], "test_data")
    print(f"[INFO]: Test data root: {val_data_root}")
    print(f"[TIME]: Roots are defined now... {time.perf_counter()-t:.3f} s")

    #4. Starten des Trainings
    t = time.perf_counter()
    print(f"[INFO]: Training starts now...")
    train(ckpt_dir, train_data_root, val_data_root, config) # Startet das Training mit den definierten Pfaden.
    print(f"[TIME]:  now... {time.perf_counter()-t:.3f} s")
