"""Code for training a model on the ETHMugs dataset."""

import argparse #maxi: Importiert das Python-Modul, um Kommandozeilenargumente zu verarbeiten.
import os #maxi: Importiert das Betriebssystem-Modul für Datei- und Verzeichnisoperationen.
from datetime import datetime #maxi: Importiert die Klasse, um Zeitstempel für Ordnernamen oder Logs zu erzeugen.

import torch #maxi: Importiert PyTorch, das Deep-Learning-Framework.
from torch.utils.data import DataLoader #maxi: Importiert den DataLoader von PyTorch, um Daten in Batches zu laden.
from torchvision import transforms #maxi: Importiert die Bild-Transformationen von torchvision.

from PIL import Image #maxi: Importiert die Bibliothek Pillow, um Bilddateien zu speichern oder zu laden.

from eth_mugs_dataset import ETHMugsDataset 
from utils import IMAGE_SIZE, compute_iou, save_predictions
from model import FCN  


def build_model() -> FCN:  
    """Build the model."""
    return FCN(in_channels=3, out_channels=1) #maxi: Gibt ein FCN-Modell zurück, das 3 Eingangskanäle (RGB) und 1 Ausgangskanal (Maske) besitzt.

def train(ckpt_dir: str, train_data_root: str, val_data_root: str):
    """Train function."""
    # Logging and validation settings
    log_frequency = 10 #maxi: Legt fest, wie oft während des Trainings Logs ausgegeben werden.
    val_batch_size = 1 #maxi: Setzt die Batchgröße für die Validierung 
    val_frequency = 1 #maxi: Gibt an, nach wie vielen Epochen eine Validierung durchgeführt wird.

    # Hyperparameters
    num_epochs = 10 #maxi: Anzahl der Trainingsepochen wird auf 10 gesetzt.
    lr = 1e-4 #maxi: Lernrate für den Optimierer.
    train_batch_size = 8 #maxi: Batchgröße fürs Training.
    # val_batch_size = 1b
    # ...

    print(f"[INFO]: Number of training epochs: {num_epochs}")
    print(f"[INFO]: Learning rate: {lr}")
    print(f"[INFO]: Training batch size: {train_batch_size}")


    # Choose Device
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #maxi: Prüft, ob eine GPU verfügbar ist und wählt das richtige Device.
    print(f"[INFO]: Using device: {device}")

    # Define Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]) #maxi: Definiert eine Bildtransformation (Resize und ToTensor) als Pipeline.

    train_dataset: ETHMugsDataset = ETHMugsDataset(root_dir=train_data_root, mode="train") #maxi: Erstellt ein Trainings-Dataset-Objekt mit den Trainingsdaten.
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True) #maxi: Erzeugt einen DataLoader für das Training mit definierter Batchgröße und zufälliger Durchmischung.

    test_dataset: ETHMugsDataset = ETHMugsDataset(root_dir=val_data_root, mode="val") #maxi: Erstellt ein Test-Dataset-Objekt für die Validierung.
    test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False) #maxi: Erstellt einen DataLoader für die Validierung ohne Shuffle.

    out_dir: str = os.path.join('prediction') #maxi: Legt den Pfad zum Ordner für die Vorhersagen fest.
    os.makedirs(out_dir, exist_ok=True) #maxi: Erstellt den Ausgabeordner, falls dieser noch nicht existiert.
    print(f"[INFO]: Saving the predicted segmentation masks to {out_dir}")

    # Define model
    model: FCN = build_model()
    model.to(device) #maxi: Verschiebt das Modell auf das gewählte Device (CPU oder GPU).

    # Define Loss function
    criterion: torch.nn.BCELoss = torch.nn.BCELoss()  # or any other loss function suitable for your task #maxi: Definiert die Loss-Funktion für binäre Klassifikation (Segmentierung).

    # Define Optimizer
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr) #maxi: Initialisiert den Adam-Optimizer mit Lernrate.

    # Define Learning rate scheduler 
    lr_scheduler: torch.optim.lr_scheduler.StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # maxi: erstellt einen Scheduler, der die Lernrate alle 10 Epochen um den Faktor 0.1 reduziert.

    # Training loop
    print(f"[INFO]: Using device: {device}")
    print("[INFO]: Starting training...")
    for epoch in range(num_epochs):
        model.train()
        print('****************************')
        print(epoch)
        print('****************************')

        for image, gt_mask in train_dataloader: #maxi: Schleife über alle Trainings-Batches.
            image = image.to(device) #maxi: Verschiebt die Eingabebilder auf das Device.
            gt_mask = gt_mask.to(device) #maxi: Verschiebt die Ground-Truth-Masken auf das Device.

            optimizer.zero_grad() #maxi: Setzt die Gradienten im Optimierer auf Null zurück.

             # Forward pass
            output = model(image) #maxi: Berechnet die Modellvorhersage für die Bilder.

            # Compute loss
            loss = criterion(output, gt_mask.float()) #maxi: Berechnet den Loss zwischen Vorhersage und Ground Truth.

            # Backward pass
            loss.backward() #maxi: Backpropagation: Gradienten berechnen.
            optimizer.step() #maxi: Aktualisiert die Modellparameter anhand der Gradienten.

            lr_scheduler.step()

            # Trace output:
            print("         Training Loss: {}".format(loss.data.cpu().numpy()),
                  "- IoU: {}".format(compute_iou(output.data.cpu().numpy() > 0.5, gt_mask.data.cpu().numpy())))
            
        # Update learning rate
        lr_scheduler.step()

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
    print("PLEASE ARCHIVE PREDICTIONS FOLDER AND RENAME THE FOLDER TO predictions{number}.csv")
    # Set data root
    train_data_root = os.path.join(args.data_root, "train_data")
    print(f"[INFO]: Train data root: {train_data_root}")

    val_data_root = os.path.join(args.data_root, "test_data")
    print(f"[INFO]: Test data root: {val_data_root}")

    train(ckpt_dir, train_data_root, val_data_root)
