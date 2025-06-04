import argparse
import os
from datetime import datetime
import torch
from PIL import Image
from eth_mugs_dataset2 import ETHMugsDataset
from utils import compute_iou, save_predictions
from model2 import UNet
from torch.nn import BCEWithLogitsLoss
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage



def build_model():
    """Erstellt und gibt das UNet-Modell zurück."""
    return UNet(3, 1)

def validate(model, dataloader, device):
    model.eval()
    ious = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) != 2:
                continue  # Skip batches without ground truth
            image, gt_mask = batch
            image, gt_mask = image.to(device), gt_mask.to(device)
            output = model(image)
            pred_mask = (torch.sigmoid(output) > 0.5).float()
            iou = compute_iou(pred_mask.cpu().numpy(), gt_mask.cpu().numpy())
            ious.append(iou)
    return np.mean(ious) if ious else 0.0

def train(ckpt_dir: str, train_data_root: str, test_data_root: str):
    
    """Trainiert das UNet-Modell mit dem ETHMugs-Datensatz."""

    # Überprüfen, ob CUDA verfügbar ist, und Gerät festlegen (CUDA erlaubt GPU-Beschleunigung)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]: Verwende Gerät: {device}")

    # Hyperparameter
    best_iou = 0.0       # Beste IoU, initialisiert auf 0
    patience = 3         # Geduld für Early Stopping
    train_batch_size = 8 # Batchgröße für das Training
    test_batch_size = 1  # Batchgröße für die Tests
    num_epochs = 10      # Anzahl der Trainingsepochen
    learning_rate = 1e-3 # Lernrate für den Optimierer
    epochs_without_improvement = 0

    print(f"[INFO]: Anzahl der Trainingsepochen: {num_epochs}")
    
    # Erstellen der Datensätze und DataLoader
    train_dataset = ETHMugsDataset(root_dir=train_data_root, mode="train")                                      # Trainingsdatensatz
    test_dataset = ETHMugsDataset(root_dir=test_data_root, mode="test")                                         # Testdatensatz
    
    # ##########################################################################
    # Data Inspection (Moved AFTER dataset initialization)
    print("\n[INFO]: Inspecting dataset samples...")
    # Visualize a sample from train
    if len(train_dataset) > 0:
        sample_image, sample_gt_mask = train_dataset[0]
        print(f"Train Image shape: {sample_image.shape}, Mask shape: {sample_gt_mask.shape}")
        print(f"Train Mask unique values: {torch.unique(sample_gt_mask)}")
        
        # Display train sample
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        # Assuming image is CxHxW and mask is 1xHxW or HxW. Adjust if necessary.
        # Convert to HxWxC for image if 3 channels, or HxW for grayscale/mask
        if sample_image.ndim == 3 and sample_image.shape[0] in [1, 3]: # C, H, W
            if sample_image.shape[0] == 1: # Grayscale
                plt.imshow(sample_image.squeeze(0).cpu().numpy(), cmap='gray')
            else: # RGB
                plt.imshow(sample_image.permute(1, 2, 0).cpu().numpy()) # H, W, C
        else:
            plt.imshow(sample_image.cpu().numpy(), cmap='gray') # Assume HxW
        plt.title(f'Train Image (shape: {sample_image.shape})')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(sample_gt_mask.squeeze(0).cpu().numpy(), cmap='gray') # Assuming mask is 1xHxW
        plt.title(f'Train Ground Truth Mask (shape: {sample_gt_mask.shape})')
        plt.axis('off')
        plt.show()
    else:
        print("[WARNING]: Training dataset is empty!")

    # Visualize a sample from test
    if len(test_dataset) > 0:
        sample_test_image, sample_test_gt_mask = test_dataset[0] # Corrected typo here
        print(f"Test Image shape: {sample_test_image.shape}, Mask shape: {sample_test_gt_mask.shape}") # Corrected typo here
        print(f"Test Mask unique values: {torch.unique(sample_test_gt_mask)}")

        # Display test sample
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        if sample_test_image.ndim == 3 and sample_test_image.shape[0] in [1, 3]: # C, H, W
            if sample_test_image.shape[0] == 1: # Grayscale
                plt.imshow(sample_test_image.squeeze(0).cpu().numpy(), cmap='gray')
            else: # RGB
                plt.imshow(sample_test_image.permute(1, 2, 0).cpu().numpy()) # H, W, C
        else:
            plt.imshow(sample_test_image.cpu().numpy(), cmap='gray') # Assume HxW
        plt.title(f'Test Image (shape: {sample_test_image.shape})')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(sample_test_gt_mask.squeeze(0).cpu().numpy(), cmap='gray') # Assuming mask is 1xHxW
        plt.title(f'Test Ground Truth Mask (shape: {sample_test_gt_mask.shape})')
        plt.axis('off')
        plt.show()
    else:
        print("[WARNING]: Test dataset is empty! This will result in 0.0 Validation IoU.")

    print("[INFO]: Data inspection complete.")
    # #############################################################################################



    # train_dataloader: 
    #   Seine Aufgabe ist es, die Trainingsdaten effizient in kleinen Paketen, sogenannten Batches, bereitzustellen.
    #   Das Training mit Batches anstelle einzelner Datenpunkte auf einmal ist speichereffizienter und kann zu stabileren Gradienten führen.
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)    # Trainings-DataLoader (Shuffle für zufällige Reihenfolge)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)      # Test-DataLoader (keine Zufälligkeit)

    # Verzeichnis zum Speichern der Vorhersagen
    out_dir = os.path.join('prediction')
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO]: Speichere die vorhergesagten Segmentierungsmasken nach {out_dir}")

    # Initialisierung des Modells, der Verlustfunktion und des Optimierers
    model = build_model().to(device)                                        # Model wird festgelegt 
    criterion = BCEWithLogitsLoss()                                         # Verwende BCEWithLogitsLoss für binäre Segmentierung -> kein Sigmoid erforderlich
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)       # Optimierer (Stochastic Gradient Descent)

    print("[INFO]: Starte das Training...")
    for epoch in range(num_epochs):                                         # Startet die Epochen-Schleife (das gesamte Trainingsdatenset wird durchlaufen)
        # Versetung des Modells in den Trainingsmodus
        # (Dies ist wichtig für bestimmte Schichten wie Dropout oder 
        #   BatchNorm, die sich im Trainingsmodus anders verhalten.)
        model.train()                         
        print(f'Start der Epoche {epoch + 1}/{num_epochs}')
        print('------------------------------')

        for i, (image, gt_mask) in enumerate(train_dataloader):             # Startet die innere Batch-Schleife (Durchlaufen des Trainings-DataLoaders)
            image, gt_mask = image.to(device), gt_mask.to(device)           # Verschiebt die Bilder und Masken auf das gewählte Gerät (CPU oder GPU)

            optimizer.zero_grad()           # Setzt die Gradienten zurück, um sie für die nächste Iteration neu zu berechnen
            output = model(image)           # Berechnet die Vorhersage des Modells für die Eingabebilder
            loss = criterion(output, gt_mask.float())   # Berechnet den Verlust zwischen der Modellvorhersage und der Ground Truth-Maske
            loss.backward() # Führt die Rückwärtspropagation durch, um die Gradienten zu berechnen.
                            # (Gradienten geben an, wie stark sich jeder Parameter ändern müsste, um den Verlust zu reduzieren)

            optimizer.step() # verwendet die im loss.backward()-Schritt berechneten Gradienten, um die Gewichte (Parameter) des Modells zu aktualisieren

            print(f"Trainingsverlust: {loss.data.cpu().numpy()}, IoU: {compute_iou((output.data.cpu().numpy() > 0.5), gt_mask.data.cpu().numpy())}")

        # Speichern des Modell-Checkpoints nach jeder Epoche
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last_epoch.pth"))

        # Evaluierung mit Testdaten
        model.eval() # Vorbereitung des Modells für die Evaluierung (Deaktivierung von Dropout und BatchNorm)
        # Initialisierung von zwei leeren Listen
        image_ids, pred_masks = [], []
            # Validierung
        val_iou = validate(model, test_dataloader, device)
        print(f"Validation IoU: {val_iou}")

        if val_iou > best_iou: # VERBESSERUNG DER VALIDIERUNGS-IoU
            best_iou = val_iou
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
        else:                   # KEINE VERBESSERUNG DER VALIDIERUNGS-IoU
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break
 # (Gradienten werden nur für das Training des Modells benötigt (während der Backpropagation). 
        # Das Deaktivieren mit .no_grad() spart Speicher und Rechenzeit, da keine zusätzlichen Informationen für den Gradientenabstieg gespeichert und berechnet werden müssen.)
    with torch.no_grad():    
        # Schleife über den Test-Datenlader
        for i, test_image in enumerate(test_dataloader):
            # 1. Bild auf das richtige Gerät verschieben (z.B. GPU)
            test_image = test_image.to(device)
            # 2. Vorhersage mit dem Modell machen
            test_output = model(test_image)
            # 3. Sigmoid-Aktivierungsfunktion NICHT MEHR anwenden (da BCEWithLogitsLoss verwendet wird)
            # test_output = torch.nn.Sigmoid()(test_output)
            # 4. Maske erstellen und für die Speicherung vorbereiten
            pred_mask = (test_output > 0.5).squeeze().cpu().numpy()
            # 5. Vorhergesagte Maske als Bild speichern
            Image.fromarray(pred_mask).save(os.path.join(out_dir, f"{str(i).zfill(4)}_mask.png"))
            # 6. Bild-ID und Maske zu den Listen hinzufügen
            image_ids.append(str(i).zfill(4))
            pred_masks.append(pred_mask)

    # 7. Vorhersagen im Kaggle-Einreichungsformat speichern
    save_predictions(image_ids=image_ids, pred_masks=pred_masks, save_path=os.path.join(out_dir, 'submission.csv'))

if __name__ == "__main__":
    # Zerlegt die Argumente für den Skriptaufruf
    parser = argparse.ArgumentParser(description="SML Projekt 2.")
    parser.add_argument("-d", "--data_root", default="./datasets", help="Pfad zum Datensatz-Ordner.")
    parser.add_argument("--ckpt_dir", default="./checkpoints", help="Pfad zum Speichern der Modell-Checkpoints.")
    args = parser.parse_args()

    # Erstellen des Checkpoint-Verzeichnisses mit Zeitstempel
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_dir = os.path.join(args.ckpt_dir, dt_string)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[INFO]: Modell-Checkpoints werden gespeichert unter: {ckpt_dir}")

    # Festlegen der Datenpfade
    train_data_root = os.path.join(args.data_root, "train_data")
    test_data_root = os.path.join(args.data_root, "test_data")
    print(f"[INFO]: Trainingsdaten-Pfad: {train_data_root}")
    print(f"[INFO]: Testdaten-Pfad: {test_data_root}")

    # Starten des Trainingsprozesses
    train(ckpt_dir, train_data_root, test_data_root)