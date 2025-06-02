#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import time

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

from utils import load_mask, compute_iou, mask_to_rle

# ANSI Colors für Terminal-Ausgabe
BOLD = "\033[1m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# ----------------- Einstellungen -----------------
DATA_DIR       = "datasets"
CHECKPOINT_DIR = "checkpoints"
PREDICT_DIR    = "prediction"
BATCH_SIZE     = 8
LR             = 1e-4
EPOCHS         = 10
VAL_RATIO      = 0.2
BASE_CH        = 64
SEED           = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

NORM = T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])

# ----------------- Dataset -----------------
class EthMugsDataset(Dataset):
    def __init__(self, root, file_list=None, augment=False):
        self.rgb_dir  = os.path.join(root, "rgb")
        self.mask_dir = os.path.join(root, "masks")
        all_imgs = [f for f in os.listdir(self.rgb_dir) if f.endswith((".jpg", ".png"))]
        names = [fn.replace("_rgb","")[:-4] for fn in all_imgs]
        self.names = file_list or names
        self.augment = augment
        self.to_tensor = T.ToTensor()
        self.norm = NORM
        self.color_jitter = T.ColorJitter(0.2, 0.2, 0.2, 0.05)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        n = self.names[idx]
        p_jpg = os.path.join(self.rgb_dir, f"{n}_rgb.jpg")
        img_path = p_jpg if os.path.exists(p_jpg) else os.path.join(self.rgb_dir, f"{n}_rgb.png")
        mask_path = os.path.join(self.mask_dir, f"{n}_mask.png")
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)
                img = img.rotate(angle, resample=Image.BILINEAR)
                mask = mask.rotate(angle, resample=Image.NEAREST)
            if random.random() < 0.5:
                img = self.color_jitter(img)
        img_t = self.to_tensor(img)
        img_t = self.norm(img_t)
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)
        mask_t = (mask_t > 0.5).float()
        return img_t, mask_t

def train_val_split(train_data_dir, val_ratio=VAL_RATIO, seed=SEED):
    rgb_dir = os.path.join(train_data_dir, "rgb")
    all_imgs = [f for f in os.listdir(rgb_dir) if f.endswith((".jpg", ".png"))]
    names = [fn.replace("_rgb","")[:-4] for fn in all_imgs]
    random.seed(seed)
    random.shuffle(names)
    val_count = int(len(names) * val_ratio)
    val_names = names[:val_count]
    train_names = names[val_count:]
    return train_names, val_names

# ----------------- U-Net -----------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c_in, c_out))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, c_in, c_out, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ConvBlock(c_in, c_out)
        else:
            self.up = nn.ConvTranspose2d(c_in//2, c_in//2, 2, stride=2)
            self.conv = ConvBlock(c_in, c_out)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet(nn.Module):
    def __init__(self, base=BASE_CH, bilinear=True):
        super().__init__()
        self.inc    = ConvBlock(3, base)
        self.down1  = Down(base, base*2)
        self.down2  = Down(base*2, base*4)
        self.down3  = Down(base*4, base*8)
        self.down4  = Down(base*8, base*8)
        self.bottleneck = ConvBlock(base*8, base*8)
        self.up1   = Up(base*8 + base*8, base*8, bilinear)
        self.up2   = Up(base*8 + base*4, base*4, bilinear)
        self.up3   = Up(base*4 + base*2, base*2, bilinear)
        self.up4   = Up(base*2 + base,   base,   bilinear)
        self.outc  = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottleneck(x5)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)

# ----------------- Loss (BCE + Dice) -----------------
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits).view(-1)
    t     = targets.view(-1)
    inter = (probs * t).sum()
    return 1 - (2*inter + eps) / (probs.sum() + t.sum() + eps)

# ----------------- Training & Validation -----------------
def train_one_epoch(model, loader, opt):
    model.train()
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss()
    t_start = time.time()
    for img, m in loader:
        img, m = img.to(DEVICE), m.to(DEVICE)
        opt.zero_grad()
        logits = model(img)
        loss = bce(logits, m) + 0.5 * dice_loss(logits, m)
        loss.backward()
        opt.step()
        total_loss += loss.item() * img.size(0)
    t_end = time.time()
    print(f"    {CYAN}Train-Durchlauf Zeit: {t_end-t_start:.2f} Sekunden{RESET}")
    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate_one_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    batches = 0
    bce = nn.BCEWithLogitsLoss()
    t_start = time.time()
    for img, m in loader:
        img, m = img.to(DEVICE), m.to(DEVICE)
        logits = model(img)
        loss = bce(logits, m) + 0.5 * dice_loss(logits, m)
        total_loss += loss.item() * img.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        iou_batch = 0.0
        for b in range(preds.size(0)):
            p = preds[b,0].cpu().numpy().astype(np.uint8)
            g = m[b,0].cpu().numpy().astype(np.uint8)
            iou_batch += compute_iou(p, g)
        total_iou += iou_batch / preds.size(0)
        batches += 1
    t_end = time.time()
    print(f"    {CYAN}Validation-Durchlauf Zeit: {t_end-t_start:.2f} Sekunden{RESET}")
    return total_loss / len(loader.dataset), total_iou / batches

# ----------------- Inferenz & Submission & Einzelbildspeicherung -----------------
@torch.no_grad()
def inference(model):
    print(f"{YELLOW}Starte Inferenz für Testdaten ...{RESET}")
    model.eval()
    ids, rles = [], []
    td = os.path.join(DATA_DIR, "test_data", "rgb")
    os.makedirs(PREDICT_DIR, exist_ok=True)
    files = sorted(f.replace("_rgb","")[:-4] for f in os.listdir(td) if f.endswith((".jpg",".png")))
    to_t = T.ToTensor()
    t_start = time.time()
    for n in files:
        p_jpg = os.path.join(td, f"{n}_rgb.jpg")
        p_png = os.path.join(td, f"{n}_rgb.png")
        img = Image.open(p_jpg if os.path.exists(p_jpg) else p_png).convert("RGB")
        t = to_t(img)
        t = NORM(t).unsqueeze(0).to(DEVICE)
        logit = model(t)
        pm = (torch.sigmoid(logit) > 0.5).cpu().numpy()[0,0].astype(np.uint8)

        # Einzelbild speichern
        out_path = os.path.join(PREDICT_DIR, f"{n}_pred.png")
        Image.fromarray((pm * 255).astype(np.uint8)).save(out_path)

        # Robust gegen Nulls:
        rle = mask_to_rle(pm)
        if rle is None or rle == '' or pd.isna(rle):
            rle = '1 1'  # Default: "leere" Maske im RLE-Format
        ids.append(n)
        rles.append(rle)
    t_end = time.time()
    print(f"{CYAN}Inferenz-Zeit für {len(files)} Bilder: {t_end-t_start:.2f} Sekunden{RESET}")
    df = pd.DataFrame({"ImageId": ids, "EncodedPixels": rles})
    if df.isnull().values.any():
        print(f"{RED}Warnung: Submission hätte Nullwerte!{RESET}")
        df = df.fillna('1 1')
    outpath = os.path.join(CHECKPOINT_DIR, "submission.csv")
    df.to_csv(outpath, index=False)
    print(f"{GREEN}Submission gespeichert unter:{RESET} {outpath}")
    print(f"{GREEN}Prediction-Bilder gespeichert im Ordner:{RESET} {PREDICT_DIR}")

# ----------------- Hauptprogramm -----------------
if __name__ == "__main__":
    print(f"{BOLD}{CYAN}{'='*48}{RESET}")
    print(f"{BOLD}{CYAN} ETH MUGS SEGMENTATION TRAINING{RESET}")
    print(f"{BOLD}{CYAN}{'='*48}{RESET}")
    print(f"{BOLD}Gerät:{RESET} {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"{GREEN}CUDA Name: {torch.cuda.get_device_name(0)}{RESET}")
    print(f"{BOLD}Random Seed:{RESET} {SEED}")
    print(f"{BOLD}Epochen:{RESET} {EPOCHS}")
    print(f"{BOLD}Datenpfad:{RESET} {DATA_DIR}")
    print(f"{BOLD}Checkpoint-Verzeichnis:{RESET} {CHECKPOINT_DIR}")
    print(f"{BOLD}Prediction-Ordner:{RESET} {PREDICT_DIR}")
    print(f"{BOLD}{'-'*48}{RESET}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    t_split = time.time()
    train_names, val_names = train_val_split(os.path.join(DATA_DIR, "train_data"))
    t_split = time.time() - t_split
    print(f"{BOLD}Trainings-/Val-Split:{RESET} {len(train_names)} / {len(val_names)} (Dauer: {t_split:.2f}s)")
    print(f"{BOLD}Lade Datasets und erstelle DataLoader ...{RESET}")
    t_loader = time.time()
    train_ds = EthMugsDataset(os.path.join(DATA_DIR, "train_data"), file_list=train_names, augment=True)
    val_ds   = EthMugsDataset(os.path.join(DATA_DIR, "train_data"), file_list=val_names, augment=False)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    t_loader = time.time() - t_loader
    print(f"{GREEN}Loader bereit. (Dauer: {t_loader:.2f}s){RESET}")

    print(f"{BOLD}Initialisiere Modell und Optimizer ...{RESET}")
    net = UNet(base=BASE_CH).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=LR)
    best_iou = 0.0

    print(f"{BOLD}{CYAN}{'-'*48}{RESET}")
    print(f"{BOLD}{CYAN}Starte Training ...{RESET}")
    print(f"{BOLD}{CYAN}{'-'*48}{RESET}")

    t_total = time.time()
    for epoch in range(1, EPOCHS+1):
        t_epoch = time.time()
        print(f"{BOLD}{YELLOW}EPOCH {epoch}/{EPOCHS}{RESET}")
        tl = train_one_epoch(net, train_ld, opt)
        vl, vi = validate_one_epoch(net, val_ld)
        if vi > best_iou:
            best_iou = vi
            torch.save(net.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"    {GREEN}Neues bestes Modell gespeichert! Val IoU: {vi:.4f}{RESET}")
        print(f"    Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | Val IoU: {vi:.4f} | Best IoU: {best_iou:.4f} | {CYAN}Epoch-Zeit: {time.time()-t_epoch:.1f}s{RESET}")
        print(f"{'-'*48}")
    t_total = time.time() - t_total
    print(f"{BOLD}{GREEN}Training abgeschlossen. Gesamtzeit: {t_total:.1f}s{RESET}")

    print(f"{BOLD}{CYAN}Lade bestes Modell für Inferenz ...{RESET}")
    net.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth"), map_location=DEVICE))
    inference(net)
    print(f"{BOLD}{GREEN}Fertig. Viel Erfolg beim Einreichen!{RESET}")
