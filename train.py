"""Code template for training a model on the ETHMugs dataset."""

import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from eth_mugs_dataset import ETHMugsDataset
from utils import IMAGE_SIZE, compute_iou, save_predictions
from model import FCN  


def build_model():  
    """Build the model."""
    return FCN(in_channels=3, out_channels=1)

def train(
    ckpt_dir: str,
    train_data_root: str,
    val_data_root: str,
):
    """Train function."""
    # Logging and validation settings
    log_frequency = 10
    val_batch_size = 1
    val_frequency = 1

    # Hyperparameters
    num_epochs = 1
    lr = 1e-4
    train_batch_size = 8
    # val_batch_size = 1
    # ...

    print(f"[INFO]: Number of training epochs: {num_epochs}")
    print(f"[INFO]: Learning rate: {lr}")
    print(f"[INFO]: Training batch size: {train_batch_size}")
    # print(f"[INFO]: Image scale: {image_scale}")


    # Choose Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define Dataset and DataLoader
    # ETHMugsDataset 
    # Data loaders

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    train_dataset = ETHMugsDataset(root_dir=train_data_root, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    val_dataset = ETHMugsDataset(root_dir=val_data_root, mode="val", transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    # Define model
    model = build_model()
    model.to(device)

    # Define Loss function
    criterion = torch.nn.BCELoss()  # or any other loss function suitable for your task

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define Learning rate scheduler if needed
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop!
    print("[INFO]: Starting training...")
    for epoch in range(num_epochs):
        model.train()

        for image, gt_mask in train_dataloader:
            image = image.to(device)
            gt_mask = gt_mask.to(device)

            optimizer.zero_grad()

             # Forward pass
            output = model(image)

            # Compute loss
            loss = criterion(output, gt_mask.float())

            # Backward pass
            loss.backward()
            optimizer.step()

            lr_scheduler.step()

        # Save model
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last_epoch.pth"))

        if epoch % val_frequency == 0:
            model.eval()

            val_iou = 0.0
            with torch.no_grad():
                for val_image, val_gt_mask in val_dataloader:
                    val_image = val_image.to(device)
                    val_gt_mask = val_gt_mask.to(device)

                    # Forward pass
                    output = model(val_image)

                    # Compute IoU
                    predicted_mask = (output > 0).float()
                    val_iou += compute_iou(predicted_mask, val_gt_mask)

                val_iou /= len(val_dataloader)

                val_iou *= 100

                print(f"[INFO]: Validation IoU: {val_iou.item():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SML Project 2.")
    parser.add_argument(
        "-d",
        "--data_root",
        default="./datasets",
        help="Path to the datasets folder.",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="./checkpoints",
        help="Path to save the model checkpoints to.",
    )
    args = parser.parse_args()

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_dir = os.path.join(args.ckpt_dir, dt_string)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)
    print("PLEASE ARCHIVE PREDICTIONS AND RENAME THE FILE TO predictions{number}.csv")
    # Set data root
    train_data_root = os.path.join(args.data_root, "training_data")
    print(f"[INFO]: Train data root: {train_data_root}")

    val_data_root = os.path.join(args.data_root, "validation_data")
    print(f"[INFO]: Validation data root: {val_data_root}")

    train(ckpt_dir, train_data_root, val_data_root)
