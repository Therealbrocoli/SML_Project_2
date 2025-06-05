#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import pandas as pd                     ### updaten: pandas wird für RLE-Checks benötigt
import torch.nn.functional as F          ### updaten: F.interpolate

from dataset import ETHMugsDataset        # NICHT ÄNDERN
from DeepLabUnet1 import DeepLabUnet      # NICHT ÄNDERN
from utils import IMAGE_SIZE, mean_std, mask_to_rle, compute_iou

# === ANSI TERMINAL Farben ===
BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
RED   = "\033[91m"
RESET = "\033[0m"


def compute_dice_loss(logits, targets, eps=1e-6):
    """
    Eigene Dice‐Loss‐Funktion, analog zu deiner Vorlage.
    Nimmt rohe Logits (ohne Sigmoid) und binäre GT-Masken.
    """
    probs = torch.sigmoid(logits).view(-1)
    t     = targets.view(-1)
    inter = (probs * t).sum()
    return 1 - (2 * inter + eps) / (probs.sum() + t.sum() + eps)


def load_config(config_path):
    """
    Lädt eine YAML-Konfigurationsdatei.
    """
    t0 = time.perf_counter()
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"[TIME]: load_config: Config is loaded in {time.perf_counter()-t0:.3f} s")
    return config


def plot_training_progress(train_losses, val_ious):
    """
    Zeichnet den Trainings-Loss (über alle Iterationen) und die Validierungs-IoU (pro Epoche).
    """
    t0 = time.perf_counter()
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
    plt.savefig('prediction/training_progress.png')
    plt.show()
    print(f"[TIME]: plot_training_progress: plot is loaded in {time.perf_counter()-t0:.3f} s")

@torch.no_grad()
def inference_and_save_csv(model, device, out_dir="prediction"):
    """
    Führt Inferenz auf allen Testbildern durch und speichert:
      1) Einzelne Prediktionsbilder in `out_dir`
      2) Eine submission.csv im out_dir mit Spalten [ImageId, EncodedPixels]
         (geleert durch "1 1", falls die RLE leer oder NaN ist).
    """
    model.eval()
    ids, rles = [], []

    # DATA_DIR wurde global erst im main gesetzt
    td = os.path.join(DATA_DIR, "test_data", "rgb")
    os.makedirs(out_dir, exist_ok=True)

    # Dieselben Normalisierungswerte wie im Dataset verwenden
    to_tensor = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.427, 0.419, 0.377], [0.234, 0.225, 0.236])
    ])

    # Alle Testdateien (nach Name sortiert) laden
    files = sorted(f.replace("_rgb", "")[:-4]
                   for f in os.listdir(td)
                   if f.endswith((".jpg", ".png")))

    print(f"{CYAN}Starte Inferenz für Testdaten ...{RESET}")
    t0 = time.time()

    for n in files:
        p_jpg = os.path.join(td, f"{n}_rgb.jpg")
        p_png = os.path.join(td, f"{n}_rgb.png")
        img_path = p_jpg if os.path.exists(p_jpg) else p_png

        img = Image.open(img_path).convert("RGB")
        img_t = to_tensor(img).unsqueeze(0).to(device)

        logit = model(img_t)
        pm = (torch.sigmoid(logit) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)

        # 1) Einzelbild speichern
        out_path = os.path.join(out_dir, f"{n}_pred.png")
        Image.fromarray((pm * 255).astype(np.uint8)).save(out_path)

        # 2) RLE enkodieren (für CSV)
        rle = mask_to_rle(pm)
        if (rle is None) or (rle == "") or (pd.isna(rle)):
            rle = "1 1"
        ids.append(n)
        rles.append(rle)

    t1 = time.time()
    print(f"{CYAN}Inferenz-Zeit für {len(files)} Bilder: {t1 - t0:.2f} Sekunden{RESET}")

    # DataFrame bauen
    df = pd.DataFrame({"ImageId": ids, "EncodedPixels": rles})
    if df.isnull().values.any():
        print(f"{RED}Warnung: Submission hätte Nullwerte!{RESET}")
        df = df.fillna("1 1")

    # CSV speichern (im out_dir)
    outpath = os.path.join(out_dir, "submission.csv")
    df.to_csv(outpath, index=False)
    print(f"{GREEN}Submission gespeichert unter: {outpath}{RESET}")
    print(f"{GREEN}Prediction-Bilder gespeichert im Ordner: {out_dir}{RESET}")


def train(ckpt_dir: str, train_data_root: str, val_data_root: str, config: dict):
    """
    Haupt‐Trainingsschleife mit:
      • Train/Val‐Split
      • DataLoader
      • DeepLabUnet‐Initialisierung
      • BCE+Dice‐Loss
      • Adam + ReduceLROnPlateau
      • Early Stopping
      • Speicherung des jeweils besten Modells (best_model.pth)
    """
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[TIME]: train: train device {BOLD}{GREEN}{DEVICE}{RESET} is chosen")

    # --- 1) Kompletter Trainingsdatensatz laden ---
    full_train_dataset = ETHMugsDataset(root_dir=train_data_root, mode="train")
    print(f"[TIME]: train: full_train_dataset is loaded as class ETHMugsDataset")

    # --- 2) Train/Val Split (80/20) ---
    train_len = int(0.8 * len(full_train_dataset))
    val_len = len(full_train_dataset) - train_len
    train_dataset, val_dataset = random_split(full_train_dataset, [train_len, val_len])
    print(f"[TIME]: train: full_train_dataset has been split into train ({train_len}) and val ({val_len})")

    # --- 3) DataLoader erstellen ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['hyperparameters']['train_batch_size'],   # train_batch_size statt batch_size ### updaten
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['hyperparameters']['val_batch_size'],     # val_batch_size ### updaten
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    print(f"[TIME]: train: train & val DataLoader erstellt")

    # --- 4) Ausgabeordner erzeugen ---
    save_dir = config['paths']['out_dir']
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO]: train: will save the predicted segmentation masks to {save_dir}")

    # --- 5) Modell initialisieren ---
    model = DeepLabUnet(num_classes=1).to(DEVICE)
    print(f"[TIME]: train: DeepLabUnet gebaut und nach {DEVICE} verschoben")

    # --- 6) Loss‐Funktionen definieren ---
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    #criterion_bce = torch.nn.MultiLabelSoftMarginLoss() 

    # --- 7) Optimizer ---
    lr = float(config['hyperparameters']['learning_rate'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"[TIME]: train: Adam‐Optimizer definiert mit lr={lr}")

    # --- 8) LR‐Scheduler (ReduceLROnPlateau, überwacht Val‐IoU) ---
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['hyperparameters']['factor'],
        patience=config['hyperparameters']['patience']
    )
    print(f"[TIME]: train: ReduceLROnPlateau‐Scheduler definiert")

    # --- 9) Early‐Stopping – Variablen ---
    best_val_iou = -1.0
    epochs_no_improve = 0
    es_patience = config['hyperparameters']['es_patience']

    # --- 10) Trainings‐Loop über Epochen ---
    train_losses = []
    val_ious = []
    epochs = config['hyperparameters']['num_epochs']
    print(f"[INFO]: train: Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # 10.1) Train‐Modus
        model.train()
        print(f"[TIME]: train: trainingsloop: model set to train mode")

        print('-' * 40)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{BOLD}EPOCH {epoch}  |  LR {current_lr:.6f}{RESET}")
        print('-' * 40)

        # 10.2) Trainingsbatches
        for i, (image, gt_mask) in enumerate(train_loader):
            image = image.to(DEVICE)
            gt_mask = gt_mask.to(DEVICE)  # (B,1,H_orig,W_orig)

            optimizer.zero_grad()
            logits = model(image)        # (B,1, H_resized, W_resized)

            # Loss = BCEWithLogits + 0.5 * DiceLoss
            loss_bce = criterion_bce(logits, gt_mask)   # Mask wird hier automatisch aufgelöst/broadcasted
            loss_dice = compute_dice_loss(logits, gt_mask)
            loss = 1* loss_bce +  0.5* loss_dice

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if (i + 1) % config['hyperparameters']['log_frequency'] == 0:
                print(f"{BOLD}[INFO] -> Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}{RESET}")

        print(f"[TIME]: train: trainingsloop: {len(train_loader)} training batches done")

        # 10.3) Validation‐Loop
        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for image, gt_mask in val_loader:
                image = image.to(DEVICE)
                gt_mask = gt_mask.to(DEVICE)  # (B,1,H_orig,W_orig)

                # ### Fix für IoU: GT-Maske auf Modell-Ausgabegröße (IMAGE_SIZE) interpolieren ###
                gt_mask_resized = F.interpolate(gt_mask, size=IMAGE_SIZE, mode='nearest')  ### updaten: GT-Resizing
                logits = model(image)
                preds = torch.sigmoid(logits).float()

                # Pro Batch‐Element IoU berechnen
                for b in range(preds.shape[0]):
                    pred_np = (preds[b, 0].cpu().numpy() > 0.5).astype(np.uint8)
                    gt_np   = gt_mask_resized[b, 0].cpu().numpy().astype(np.uint8)  ### updaten: resized GT verwenden

                    # ### Union‐Guard: falls Union == 0, IoU = 0 ###
                    intersection = np.logical_and(pred_np, gt_np)
                    union = np.logical_or(pred_np, gt_np)
                    if union.sum() == 0:
                        curr_iou = 0.0
                    else:
                        curr_iou = intersection.sum() / union.sum()
                    val_iou += curr_iou                                              ### updaten: kein NaN mehr

        # Auf Anzahl aller Val‐Examples normalisieren
        val_iou /= len(val_loader.dataset)
        val_ious.append(val_iou)
        print(f"{BOLD}[INFO] -> Validation IoU: {CYAN}{val_iou:.4f}{RESET}")

        # 10.4) LR‐Scheduler updaten (mit Val‐IoU)
        lr_scheduler.step(val_iou)

        # 10.5) Best Model & Early‐Stopping prüfen
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_no_improve = 0
            # Bestes Modell speichern
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"{GREEN}Neues bestes Modell gespeichert! Val IoU: {val_iou:.4f}{RESET}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= es_patience:
            print(f"{RED}[INFO]: Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {es_patience} epochs).{RESET}")
            break

        # 10.6) Regelmäßiges Checkpointen (optional jede 3. Epoche)
        if (epoch % 3) == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[TIME]: train: trainingsloop: checkpoint saved at {ckpt_path}")

    print(f"[TIME]: train: TOTAL Training time {(time.perf_counter() - t0):.3f} s")

    # --- 11) Plotting von Loss und Validation‐IoU ---
    print(f"[INFO]: train: plotting starts")
    plot_training_progress(train_losses, val_ious)

    # Fertiges Training: best_model.pth im ckpt_dir liegt nun bereit

def seed_everything(seed: int = 42):
    # ---------- Python & NumPy ----------
    random.seed(seed)
    np.random.seed(seed)

    # ---------- PyTorch ----------
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)      # alle GPUs, falls du mehrere hast

    # ---------- CUDNN: deterministisch statt Speed-Rally ----------
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---------- PyTorch 1.10+ Hardcore-Mode ----------
    torch.use_deterministic_algorithms(True)

    # ---------- CuBLAS (CUDA ≥ 10.2) ----------
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if __name__ == "__main__":
    t0 = time.perf_counter()
    
    print(f"{BOLD}Meine Lieben es ist mir eine Freude sie begrüßen zu dürfen – wir beginnen...{RESET}")

    # 1) Argumentparser für das Config‐File
    parser = argparse.ArgumentParser(description="SML Project 2.")
    parser.add_argument(
        "-c",
        "--config",
        default="config_DeepLab.yaml",
        help="Path to the config file."
    )
    args = parser.parse_args()

    # 2) Config laden
    config = load_config(args.config)

    # 3) Checkpoint‐Ordner erstellen mit Zeitstring
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    CHECKPOINT_DIR = os.path.join(config['paths']['ckpt_dir'], dt_string)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"[INFO]: Model checkpoints will be saved to: {CHECKPOINT_DIR}")

    # 4) Daten‐Wurzelverzeichnisse
    DATA_DIR = config['paths']['data_root']          # z.B. "./datasets"
    train_data_root = os.path.join(DATA_DIR, "train_data")
    val_data_root   = os.path.join(DATA_DIR, "test_data")
    print(f"[INFO]: Train data root: {train_data_root}")
    print(f"[INFO]: Test data root:  {val_data_root}")

    # 5) Training aufrufen
    print(f"[INFO]: Training starts now...")
    train(CHECKPOINT_DIR, train_data_root, val_data_root, config)

    # 6) Nach dem Training: Bestes Modell für Inferenz laden und Submission erzeugen
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    net = DeepLabUnet(num_classes=1).to(DEVICE)
    print("hallo")
    net.load_state_dict(torch.load(best_ckpt, map_location=DEVICE, weights_only = True))
    print(f"{BOLD}{CYAN}Lade bestes Modell für Inferenz ...{RESET}")

    inference_and_save_csv(net, DEVICE, out_dir=config['paths']['out_dir'])
    print(f"{BOLD}{GREEN}Fertig – submission.csv und Prediktionsbilder sind erstellt!{RESET}")