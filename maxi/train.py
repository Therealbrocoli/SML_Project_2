import argparse
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader, random_split
from mugs import ETHMugsDataset
from unet_simplistic import UNet
from utils import compute_iou

def build_model():
    # Tieferes, breiteres UNet (kannst du anpassen, hier wie vorgegeben)
    return UNet(n_channels=3, n_classes=1, bilinear=True)

def train(ckpt_dir: str, data_root: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 8
    num_epochs = 100
    lr = 1e-3
    patience = 10  # Für Early Stopping
    val_ratio = 0.15

    # Lade kompletten Datensatz und splitte (Train/Val)
    full_dataset = ETHMugsDataset(root_dir=data_root, mode="train", use_aug=True)
    val_size = int(val_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Val ohne Augmentation laden!
    val_dataset.dataset.use_aug = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()  # Logits loss, numerisch stabil!

    best_iou = 0.0
    patience_counter = 0

    print("[INFO]: Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)  # [B,1,H,W] -> [B,H,W]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                iou = compute_iou(preds.squeeze(1).cpu().numpy(), masks.squeeze(1).cpu().numpy())
                val_iou += iou
        val_iou /= len(val_loader)

        print(f"[Epoch {epoch+1:03d}] Loss: {avg_loss:.4f} | Val IoU: {val_iou*100:.2f}")

        # Checkpoint only best
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
            print(f"  [INFO] Model improved! Checkpoint saved. IoU: {best_iou*100:.2f}")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("[INFO] Early stopping triggered.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SML Project 2.")
    parser.add_argument("-d", "--data_root", default="./datasets/train_data", help="Pfad zum Trainingsordner (mit rgb/ und masks/)")
    parser.add_argument("--ckpt_dir", default="./checkpoints", help="Checkpoint-Ordner")
    args = parser.parse_args()
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_dir = os.path.join(args.ckpt_dir, now)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)
    train(ckpt_dir, args.data_root)
