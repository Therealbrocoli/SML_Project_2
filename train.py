"""Code for training a model on the ETHMugs dataset."""

import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image

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


    # Choose Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    train_dataset = ETHMugsDataset(root_dir=train_data_root, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    test_dataset = ETHMugsDataset(root_dir=val_data_root, mode="val")
    test_dataloader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    out_dir = os.path.join('prediction')
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO]: Saving the predicted segmentation masks to {out_dir}")

    # Define model
    model = build_model()
    model.to(device)

    # Define Loss function
    criterion = torch.nn.BCELoss()  # or any other loss function suitable for your task

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define Learning rate scheduler if needed
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    print("[INFO]: Starting training...")
    for epoch in range(num_epochs):
        model.train()
        print('****************************')
        print(epoch)
        print('****************************')

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

            # Trace output:
            print("         Training Loss: {}".format(loss.data.cpu().numpy()),
                  "- IoU: {}".format(compute_iou(output.data.cpu().numpy() > 0.5, gt_mask.data.cpu().numpy())))

        # Save model
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last_epoch.pth"))

        if epoch % val_frequency == 0:
            model.eval()
            image_ids = []
            pred_masks = []

            with torch.no_grad():
                for i, (image, _) in enumerate(test_dataloader):
                    image = image.to(device)

                    # Forward pass
                    test_output = model(image)
                    test_output = torch.nn.Sigmoid()(test_output)

                    # convert to binary image mask:
                    pred_mask = (test_output > 0.5).squeeze().cpu().numpy()

                    # Save the predicted mask as image (for visualization) - do not submit these files!
                    pred_mask_image = Image.fromarray((pred_mask * 255).astype('uint8'))
                    pred_mask_image.save(os.path.join(out_dir, f"{str(i).zfill(4)}_mask.png"))

                    # Update lists of image ids and masks
                    image_ids.append(str(i).zfill(4))
                    pred_masks.append(pred_mask)

                # Save predictions in Kaggle submission format
                save_predictions(image_ids=image_ids, pred_masks=pred_masks, save_path=os.path.join(out_dir, 'submission.csv'))

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
    train_data_root = os.path.join(args.data_root, "train_data")
    print(f"[INFO]: Train data root: {train_data_root}")

    val_data_root = os.path.join(args.data_root, "test_data")
    print(f"[INFO]: Test data root: {val_data_root}")

    train(ckpt_dir, train_data_root, val_data_root)
