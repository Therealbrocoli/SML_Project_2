#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

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
BATCH_SIZE     = 2
LR             = 1e-4
EPOCHS         = 10
VAL_RATIO      = 0.2
SEED           = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def mask_to_rle(mask):
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) < 2:
        return ""
    runs[1::2] = runs[1::2] - runs[::2]
    return ' '.join(str(x) for x in runs)

# ----------------- Dataset für Mask R-CNN -----------------
class CupSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, filelist=None, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.imgs = filelist or sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.imgs[idx])
        # Basename ohne _rgb und Extension extrahieren
        if "_rgb" in self.imgs[idx]:
            img_base = self.imgs[idx].replace('_rgb.jpg', '').replace('_rgb.png', '').replace('_rgb.jpeg', '')
        else:
            img_base = os.path.splitext(self.imgs[idx])[0]
        mask_path = os.path.join(self.mask_dir, f"{img_base}_mask.png")
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
        obj_ids = np.unique(mask_np)
        obj_ids = obj_ids[obj_ids != 0]
        masks = (mask_np == obj_ids[:, None, None]).astype(np.uint8) if obj_ids.size > 0 else np.zeros((1, mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)
        num_objs = len(obj_ids) if obj_ids.size > 0 else 1
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            if pos[0].size == 0 or pos[1].size == 0:
                continue
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
        if not boxes:
            boxes = [[0,0,1,1]]
            masks = np.zeros((1, mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)
            num_objs = 1
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        masks = torch.as_tensor(masks[:boxes.shape[0]], dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

def train_val_split(image_dir, val_ratio=VAL_RATIO, seed=SEED):
    all_imgs = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
    random.seed(seed)
    random.shuffle(all_imgs)
    val_count = int(len(all_imgs) * val_ratio)
    return all_imgs[val_count:], all_imgs[:val_count]

def get_instance_segmentation_model(num_classes=2):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)
    return model

# ----------------- Training/Validation Loop -----------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    print(f"    {CYAN}Train-Durchlauf Zeit: {time.time()-t0:.2f}s{RESET}")
    return total_loss / len(loader)

@torch.no_grad()
def validate_one_epoch(model, loader, device):
    total_loss = 0.0
    total_iou = 0.0
    batches = 0
    t0 = time.time()
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # Loss explizit im train-Modus berechnen
        model.train()
        with torch.enable_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
        model.eval()
        outputs = model(images)
        for i in range(len(outputs)):
            if outputs[i]['masks'].shape[0] == 0: continue
            pred = (outputs[i]['masks'] > 0.5).float().cpu().numpy()
            gt = targets[i]['masks'][0].cpu().numpy()
            union = np.logical_or(np.any(pred, axis=0), gt)
            inter = np.logical_and(np.any(pred, axis=0), gt)
            iou = inter.sum() / (union.sum() + 1e-6)
            total_iou += iou
            batches += 1
    print(f"    {CYAN}Validation-Durchlauf Zeit: {time.time()-t0:.2f}s{RESET}")
    return total_loss / len(loader), (total_iou / batches if batches else 0.0)

# ----------------- Inferenz & Submission & Einzelbildspeicherung -----------------
@torch.no_grad()
def inference(model, device):
    print(f"{YELLOW}Starte Inferenz für Testdaten ...{RESET}")
    model.eval()
    test_dir = os.path.join(DATA_DIR, "test_data", "rgb")
    os.makedirs(PREDICT_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))])
    ids, rles = [], []
    t0 = time.time()
    for fname in files:
        img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
        img_t = T.ToTensor()(img).unsqueeze(0).to(device)
        output = model(img_t)[0]
        masks = output["masks"] > 0.5
        if masks.shape[0] == 0:
            mask_img = np.zeros((img.height, img.width), dtype=np.uint8)
        else:
            mask_img = np.any(masks.squeeze(1).cpu().numpy(), axis=0).astype(np.uint8) * 255
        out_name = os.path.splitext(fname.replace("_rgb", ""))[0] + "_pred.png"
        Image.fromarray(mask_img).save(os.path.join(PREDICT_DIR, out_name))
        rle = mask_to_rle(mask_img // 255)
        if rle is None or rle == "" or pd.isna(rle):
            rle = "1 1"
        ids.append(os.path.splitext(fname.replace("_rgb", ""))[0])
        rles.append(rle)
    print(f"{CYAN}Inferenz-Zeit für {len(files)} Bilder: {time.time()-t0:.2f}s{RESET}")
    df = pd.DataFrame({"ImageId": ids, "EncodedPixels": rles})
    if df.isnull().values.any():
        print(f"{RED}Warnung: Submission hätte Nullwerte!{RESET}")
        df = df.fillna("1 1")
    outpath = os.path.join(CHECKPOINT_DIR, "submission.csv")
    df.to_csv(outpath, index=False)
    print(f"{GREEN}Submission gespeichert unter:{RESET} {outpath}")
    print(f"{GREEN}Prediction-Bilder gespeichert im Ordner:{RESET} {PREDICT_DIR}")

# ----------------- Hauptprogramm -----------------
if __name__ == "__main__":
    print(f"{BOLD}{CYAN}{'='*48}{RESET}")
    print(f"{BOLD}{CYAN} ETH MUGS MASK R-CNN TRAINING{RESET}")
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

    img_dir = os.path.join(DATA_DIR, "train_data", "rgb")
    mask_dir = os.path.join(DATA_DIR, "train_data", "masks")
    t_split = time.time()
    train_imgs, val_imgs = train_val_split(img_dir)
    t_split = time.time() - t_split
    print(f"{BOLD}Trainings-/Val-Split:{RESET} {len(train_imgs)} / {len(val_imgs)} (Dauer: {t_split:.2f}s)")
    print(f"{BOLD}Lade Datasets und erstelle DataLoader ...{RESET}")
    t_loader = time.time()
    ttf = T.ToTensor()
    train_ds = CupSegmentationDataset(img_dir, mask_dir, train_imgs, transforms=ttf)
    val_ds   = CupSegmentationDataset(img_dir, mask_dir, val_imgs,   transforms=ttf)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    t_loader = time.time() - t_loader
    print(f"{GREEN}Loader bereit. (Dauer: {t_loader:.2f}s){RESET}")

    print(f"{BOLD}Initialisiere Modell und Optimizer ...{RESET}")
    model = get_instance_segmentation_model(num_classes=2)
    model.to(DEVICE)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    best_iou = 0.0

    print(f"{BOLD}{CYAN}{'-'*48}{RESET}")
    print(f"{BOLD}{CYAN}Starte Training ...{RESET}")
    print(f"{BOLD}{CYAN}{'-'*48}{RESET}")

    t_total = time.time()
    for epoch in range(1, EPOCHS+1):
        t_epoch = time.time()
        print(f"{BOLD}{YELLOW}EPOCH {epoch}/{EPOCHS}{RESET}")
        tl = train_one_epoch(model, train_ld, opt, DEVICE)
        vl, vi = validate_one_epoch(model, val_ld, DEVICE)
        if vi > best_iou:
            best_iou = vi
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"    {GREEN}Neues bestes Modell gespeichert! Val IoU: {vi:.4f}{RESET}")
        print(f"    Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | Val IoU: {vi:.4f} | Best IoU: {best_iou:.4f} | {CYAN}Epoch-Zeit: {time.time()-t_epoch:.1f}s{RESET}")
        print(f"{'-'*48}")
    t_total = time.time() - t_total
    print(f"{BOLD}{GREEN}Training abgeschlossen. Gesamtzeit: {t_total:.1f}s{RESET}")

    print(f"{BOLD}{CYAN}Lade bestes Modell für Inferenz ...{RESET}")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth"), map_location=DEVICE))
    inference(model, DEVICE)
    print(f"{BOLD}{GREEN}Fertig. Viel Erfolg beim Einreichen!{RESET}")