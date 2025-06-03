"""Code for training a model on the ETHMugs dataset."""# which is a mask
import argparse
import os
import random
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from dataset_preprocessing import ETHMugsDataset, ETHMugspred
from utils import IMAGE_SIZE, compute_iou, save_predictions
from kolja.DeepLab import DeepLab

# Definiert eine Funktion, die das Modell erzeugt.
def build_model():
    # Beschreibt in einem Docstring, dass hier das Modell gebaut wird.
    """Build the model."""
    return DeepLab()

# Definiert die Trainingsfunktion mit Speicherorten für Checkpoints und Daten als Argumente.
def train(
    ckpt_dir: str,
    train_data_root: str,
    val_data_root: str,

):  
    #seeds für das Training
    torch.manual_seed(42)
    random.seed(42)
    # Legt fest, wie oft während des Trainings Logs ausgegeben werden.
    log_frequency = 10
    # Setzt die Batchgröße für die Validierung auf 1.
    val_batch_size = 1
    # Gibt an, nach wie vielen Epochen eine Validierung durchgeführt wird.
    val_frequency = 1

### Hyperparamter 
    num_epochs = 10
    # Lernrate für den Optimierer.
    lr = 1e-4
    # Batchgröße fürs Training.
    train_batch_size = 8
    # val_batch_size =1 

    # Gibt die gewählte Anzahl Epochen aus.
    print(f"[INFO]: Number of training epochs: {num_epochs}")
    # Gibt die gewählte Lernrate aus.
    print(f"[INFO]: Learning rate: {lr}")
    # Gibt die Trainings-Batchgröße aus.
    print(f"[INFO]: Training batch size: {train_batch_size}")

    # Prüft, ob eine GPU verfügbar ist und wählt das richtige Device.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[INFO]: Using device: {device}")

    # Definiert eine Bildtransformation (Resize und ToTensor) als Pipeline.
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])


### Alle Datasets laden
    full_dataset = ETHMugsDataset(root_dir="datasets/train_data", mode="train")  # Erstelle das vollständige Dataset-Objekt für den Trainingsmodus
    #train_val split
    total_len = len(full_dataset)  # Bestimme die Gesamtanzahl der Proben im Dataset
    train_len = int(0.8 * total_len)  # Berechne die Anzahl der Trainingsproben (80 % des Gesamtbestands)
    val_len = total_len - train_len  # Berechne die Anzahl der Validierungsproben (restliche 20 %)
    batch_size = 32  # Definiere die Batch-Größe für den DataLoader

    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])  # Teile das Dataset zufällig in Trainings- und Validierungs-Subset auf
    test_dataset= ETHMugspred(oot_dir="datasets/test_data", mode = "test")

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)

    # Legt den Pfad zum Ordner für die Vorhersagen fest.
    out_dir = os.path.join('prediction')
    # Erstellt den Ausgabeordner, falls dieser noch nicht existiert.
    os.makedirs(out_dir, exist_ok=True)
    # Gibt aus, wohin die Masken gespeichert werden.
    print(f"[INFO]: Saving the predicted segmentation masks to {out_dir}")

### Model 
    model = build_model()
    model.to(device) #GPU oder CPU

### defiert die Loss-Funktion für binäre Klassifikation (Segmentierung).
    criterion = torch.nn.BCELoss()
### itialisiert den Adam-Optimizer mit Lernrate.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
### erstellt Scheduler, der die Lernrate alle 10 Epochen um den Faktor 0.1 reduziert.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


### Schleife über alle Trainingsepochen.
    print("[INFO]: Starting training...")
    for epoch in range(num_epochs):
        model.train()# Setzt das Modell in den Trainingsmodus
        
        print('****************************')
        print(epoch)
        print('****************************')

        # Schleife über alle Trainings-Batches.
        for image, gt_mask in train_loader:
            image = image.to(device)  # Verschiebt die Eingabebilder auf das Device.
            gt_mask = gt_mask.to(device)  # Verschiebt die Ground-Truth-Masken auf das Device.
            
            optimizer.zero_grad() # Setzt die Gradienten im Optimierer auf Null zurück.
            #Forward 
            output = model(image)
            loss = criterion(output, gt_mask.float())

            # Backpropagation: Gradienten berechnen.
            loss.backward()

            # Aktualisiert die Modellparameter anhand der Gradienten.
            optimizer.step()

            # Scheduler-Update, passt ggf. die Lernrate an.
            lr_scheduler.step()
            # Gibt Loss und IoU für das aktuelle Batch aus.
            print("         Training Loss: {}".format(loss.data.cpu().numpy()),
                  "- IoU: {}".format(compute_iou(output.data.cpu().numpy() > 0.5, gt_mask.data.cpu().numpy())))

        # Scheduler-Update nach der Epoche.
        lr_scheduler.step()
        # Speichert das Modell nach jeder Epoche als Checkpoint.
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last_epoch.pth"))

### Test Daten  
    model.eval() # Setzt das Modell in den Evaluierungsmodus.
    image_ids = [] #Initialisiert eine Liste für Bild-IDs.
    pred_masks = [] # Initialisiert eine Liste für vorhergesagte Masken.

<<<<<<< HEAD
    with torch.no_grad():
        # Schleife über alle Testbilder im DataLoader.
        for i, (image, _) in enumerate(test_dataloader):
            image = image.to(device)
            test_output = model(image)
            test_output = torch.nn.Sigmoid()(test_output) #binaere Segmentierung in ETH Tasse und Hintergrund 
=======
            # Startet einen Kontext ohne Gradientenberechnung (spart Speicher/Zeit).
            with torch.no_grad():
                # Schleife über alle Testbilder im DataLoader.
                for i, (image, _) in enumerate(test_loader):
                    # Verschiebt das Testbild auf das Device.
                    image = image.to(device)
                    # Berechnet die Vorhersage des Modells für das Testbild.
                    test_output = model(image)
                    # Wendet eine Sigmoid-Funktion auf die Modellvorhersage an (für binäre Segmentierung).
                    test_output = torch.nn.Sigmoid()(test_output)
                    # Schwellenwert auf 0.5: Erzeugt Binärmaske als NumPy-Array.
                    pred_mask = (test_output > 0.5).squeeze().cpu().numpy()
                    # Wandelt die Binärmaske in ein Graustufenbild (PIL Image) um.
                    pred_mask_image = Image.fromarray((pred_mask * 255).astype('uint8'))
                    # Speichert die Maske als PNG im Ausgabeverzeichnis.
                    pred_mask_image.save(os.path.join(out_dir, f"{str(i).zfill(4)}_mask.png"))
                    # Fügt die Bild-ID zur Liste hinzu.
                    image_ids.append(str(i).zfill(4))
                    # Fügt die Vorhersagemaske zur Liste hinzu.
                    pred_masks.append(pred_mask)
>>>>>>> 6bf54dd44695f14aebf7d890bc09ca8eb894f401

            # Schwellenwert auf 0.5: Erzeugt Binärmaske als NumPy-Array.
            pred_mask = (test_output > 0.5).squeeze().cpu().numpy()
            # Wandelt die Binärmaske in ein Graustufenbild (PIL Image) um.
            pred_mask_image = Image.fromarray((pred_mask * 255).astype('uint8'))
            pred_mask_image.save(os.path.join(out_dir, f"{str(i).zfill(4)}_mask.png"))
            
            image_ids.append(str(i).zfill(4))
            pred_masks.append(pred_mask)

        # Speichert alle Vorhersagen im Submission-Format als CSV-Datei.
        save_predictions(image_ids=image_ids, pred_masks=pred_masks, save_path=os.path.join(out_dir, 'submission.csv'))
        print(f"[INFO]: Predictions saved to {os.path.join(out_dir, 'submission.csv')}")


if __name__ == "__main__":
    # Erstellt einen Argumentparser für Kommandozeilenargumente.
    parser = argparse.ArgumentParser(description="SML Project 2.")
    # Fügt Argument für den Datensatz-Pfad hinzu.
    parser.add_argument(
        "-d",
        "--data_root",
        default="./datasets",
        help="Path to the datasets folder.",
    )
    # Fügt Argument für den Checkpoint-Pfad hinzu.
    parser.add_argument(
        "--ckpt_dir",
        default="./checkpoints",
        help="Path to save the model checkpoints to.",
    )
    
    args = parser.parse_args() 
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S") # Formatiert das Datum als String für den Ordnernamen.
    ckpt_dir = os.path.join(args.ckpt_dir, dt_string) # Verbindet Checkpoint-Pfad und Zeitstring zu neuem Verzeichnis.
    os.makedirs(ckpt_dir, exist_ok=True)# Erstellt das Checkpoint-Verzeichnis, falls nicht vorhanden.
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)
    print("PLEASE ARCHIVE PREDICTIONS FOLDER AND RENAME THE FOLDER TO predictions{number}.csv")
    
    # Setzt den Trainingsdaten-Ordner.
    train_data_root = os.path.join(args.data_root, "train_data")
    print(f"[INFO]: Train data root: {train_data_root}")
    
    val_data_root = os.path.join(args.data_root, "test_data")
    print(f"[INFO]: Test data root: {val_data_root}")

    train(ckpt_dir, train_data_root, val_data_root) # Startet das Training mit den definierten Pfaden.
