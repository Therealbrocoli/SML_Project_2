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
from torchvision import transforms  ####
from PIL import Image
from dataset_DeepLab_augmentiert import ETHMugsDataset
from utils import *
from DeepLab import DeepLabUnet  ### Neu: importiert das erweiterte Modell

def compute_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits).view(-1)
    t     = targets.view(-1)
    inter = (probs * t).sum()
    return 1 - (2 * inter + eps) / (probs.sum() + t.sum() + eps)

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

def train(ckpt_dir: str, train_data_root: str, val_data_root: str, config: dict):
    # === ANSI TERMINAL==
    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    CYAN  = "\033[96m"
    RED   = "\033[91m"
    RESET = "\033[0m"

    t0 = time.perf_counter()
    print(f"[INFO]: train: has been started")

    # 1. Device wählen
    t = time.perf_counter()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[TIME]: train: train device {BOLD}{GREEN}{device}{RESET} is chosen {time.perf_counter()-t:.3f} s")

    # 2. Load Full Train Dataset
    t = time.perf_counter()
    full_train_dataset = ETHMugsDataset(root_dir=train_data_root, mode="train")
    print(f"[TIME]: train: full_train_dataset is loaded as class ETHMugsDataset {time.perf_counter()-t:.3f} s")

    # 3. Train/Val Split
    t = time.perf_counter()
    train_len = int(0.8 * len(full_train_dataset))
    val_len = len(full_train_dataset) - train_len
    train_dataset, val_dataset = random_split(full_train_dataset, [train_len, val_len])
    print(f"[TIME]: train: full_train_dataset has been splitted into train and val {time.perf_counter()-t:.3f} s")

    # 4. DataLoader erstellen
    t = time.perf_counter()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['hyperparameters']['batch_size'], 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['hyperparameters']['val_batch_size'], 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    print(f"[TIME]: train: train & val Dataloading is done {time.perf_counter()-t:.3f} s")

    # 5. Ausgabeordner erstellen
    t = time.perf_counter()
    save_dir = config['paths']['out_dir']
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO]: train: will save the predicted segmentation masks to {save_dir} {time.perf_counter()-t:.3f} s")

    # 6. Modell initialisieren
    t = time.perf_counter()
    model = DeepLabUnet(num_classes=1).to(device)  ### Modell ist jetzt DeepLabUnet
    print(f"[TIME]: train: model is built and transferred to device {BOLD}{GREEN}{device}{RESET} {time.perf_counter()-t:.3f} s")

    # 7. Loss-Funktion definieren (BCEWithLogits + Dice)
    t = time.perf_counter()
    criterion_bce = torch.nn.BCEWithLogitsLoss()  ### Nutzt Logits direkt
    print(f"[TIME]: train: BCEWithLogitsLoss defined {time.perf_counter()-t:.3f} s")

    # 8. Optimizer initialisieren
    t = time.perf_counter()
    lr = float(config['hyperparameters']['learning_rate'])
    optimizer = torch.optim.Adam(model.parameters(), lr)
    print(f"[TIME]: train: Adam optimizer defined with lr {BOLD}{lr}{RESET} {time.perf_counter()-t:.3f} s")

    # 9. LR-Scheduler (abgestimmt auf Validation-IoU)
    t = time.perf_counter()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        factor=config['hyperparameters']['factor'],
        patience=config['hyperparameters']['patience']
    )
    print(f"[TIME]: train: ReduceLROnPlateau-Scheduler defined {time.perf_counter()-t:.3f} s")

    # 10. Training-Loop
    t0 = time.perf_counter()
    train_losses = []
    val_ious = []
    epochs = config['hyperparameters']['num_epochs']
    print(f"[INFO]: train: Starting training...")

    for epoch in range(epochs):
        # 10.1 Train-Modus
        t = time.perf_counter()
        model.train()
        print(f"[TIME]: train: trainingsloop: model set to train mode {time.perf_counter()-t:.3f} s")

        print('-' * 40)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{BOLD}EPOCH {epoch}  |  LR {current_lr:.6f}{RESET}")
        print('-' * 40)

        # 10.2 Trainingsbatches
        t = time.perf_counter()
        epoch_loss = 0.0
        for i, (image, gt_mask) in enumerate(train_loader):
            image = image.to(device)
            gt_mask = gt_mask.to(device)  # shape = (B,1,H,W), binär {0,1}

            optimizer.zero_grad()
            logits = model(image)        # shape = (B,1,H,W)

            # Loss = BCEWithLogits + 0.5 * DiceLoss
            loss_bce = criterion_bce(logits, gt_mask)
            loss_dice = compute_dice_loss(logits, gt_mask)  # eigene Dice-Loss-Funktion
            loss = loss_bce + 0.5 * loss_dice  ###

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_losses.append(loss.item())

            if (i + 1) % config['hyperparameters']['log_frequency'] == 0:
                print(f"{BOLD}[INFO] -> Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}{RESET}")

        print(f"[TIME]: train: trainingsloop: {len(train_loader)} training batches done {time.perf_counter()-t:.3f} s")

        # 10.3 Validation-Loop
        t = time.perf_counter()
        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for image, gt_mask in val_loader:
                image = image.to(device)
                gt_mask = gt_mask.to(device)
                logits = model(image)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_iou += compute_iou(preds.cpu().numpy(), gt_mask.cpu().numpy())

        val_iou /= len(val_loader)
        val_ious.append(val_iou)
        print(f"{BOLD}[INFO] -> Validation IoU: {CYAN}{val_iou:.4f}{RESET}")
        print(f"[TIME]: train: trainingsloop: validation loop done {time.perf_counter()-t:.3f} s")

        # 10.4 Learning Rate aktualisieren
        t = time.perf_counter()
        lr_scheduler.step(val_iou)
        print(f"[TIME]: train: trainingsloop: learning rate updated {time.perf_counter()-t:.3f} s")

        # 10.5 Checkpoint speichern
        t = time.perf_counter()
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[TIME]: train: trainingsloop: checkpoint saved at {ckpt_path} {time.perf_counter()-t:.3f} s")

    print(f"[TIME]: train: TOTAL Training time {BOLD}{time.perf_counter()-t0:.3f} s{RESET}")

    # 11. Plotting von Loss und Validation-IoU
    t = time.perf_counter()
    print(f"[INFO]: train: plotting starts {time.perf_counter()-t:.3f} s")
    plot_training_progress(train_losses, val_ious)
    print(f"[TIME]: train: plot done {time.perf_counter()-t:.3f} s")

    # 12. Test-Inferenz & CSV-Erzeugung
    t = time.perf_counter()
    model.eval()
    image_ids = []
    pred_masks = []
    print(f"[TIME]: train: init for Test data prediction {time.perf_counter()-t:.3f} s")

    td = os.path.join(val_data_root, "rgb")
    to_tensor = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std())    # Nutzt zuvor berechnetes mean/std
    ])  ###

    for fname in sorted(os.listdir(td)):
        if not fname.endswith((".jpg", ".png")):
            continue
        n = fname.replace("_rgb","")[:-4]
        p_jpg = os.path.join(td, f"{n}_rgb.jpg")
        p_png = os.path.join(td, f"{n}_rgb.png")
        img_path = p_jpg if os.path.exists(p_jpg) else p_png
        img = Image.open(img_path).convert("RGB")

        img_t = to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_t)
            pm = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0,0].astype(np.uint8)

        out_path = os.path.join(save_dir, f"{n}_mask.png")
        Image.fromarray((pm * 255).astype(np.uint8)).save(out_path)

        image_ids.append(n)
        pred_masks.append(pm)

    print(f"[TIME]: train: Test masks created {time.perf_counter()-t:.3f} s")

    # 13. Speichern der Predictions als CSV
    t = time.perf_counter()
    save_predictions(image_ids=image_ids, pred_masks=pred_masks, save_path=os.path.join(save_dir, 'submission.csv'))
    print(f"[INFO]: train: Predictions saved to {os.path.join(save_dir, 'submission.csv')}")

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

