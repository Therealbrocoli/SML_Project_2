import argparse
import os
from datetime import datetime
import torch
from PIL import Image
from eth_mugs_dataset2 import ETHMugsDataset
from utils import compute_iou, save_predictions
from model2 import UNet

def build_model():
    """Erstellt und gibt das UNet-Modell zurück."""
    return UNet(3, 1)

def train(ckpt_dir: str, train_data_root: str, test_data_root: str):
    """Trainiert das UNet-Modell mit dem ETHMugs-Datensatz."""

    # Überprüfen, ob CUDA verfügbar ist, und Gerät festlegen (CUDA erlaubt GPU-Beschleunigung)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]: Verwende Gerät: {device}")

    # Hyperparameter
    train_batch_size = 8 # Batchgröße für das Training
    test_batch_size = 1  # Batchgröße für die Tests
    num_epochs = 10      # Anzahl der Trainingsepochen
    learning_rate = 1e-4 # Lernrate für den Optimierer

    print(f"[INFO]: Anzahl der Trainingsepochen: {num_epochs}")

    # Erstellen der Datensätze und DataLoader
    train_dataset = ETHMugsDataset(root_dir=train_data_root, mode="train")                                      # Trainingsdatensatz
    test_dataset = ETHMugsDataset(root_dir=test_data_root, mode="test")                                         # Testdatensatz
    
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
    criterion = torch.nn.BCELoss()                                          # Verlustfunktion für binäre Klassifikation
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
            output = torch.nn.Sigmoid()(output)         # Wendet die Sigmoid-Aktivierungsfunktion an, um die Ausgaben in den Bereich [0, 1] zu bringen
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

 # (Gradienten werden nur für das Training des Modells benötigt (während der Backpropagation). 
        # Das Deaktivieren mit .no_grad() spart Speicher und Rechenzeit, da keine zusätzlichen Informationen für den Gradientenabstieg gespeichert und berechnet werden müssen.)
    with torch.no_grad():
        # Schleife über den Test-Datenlader
        for i, test_image in enumerate(test_dataloader):
            # 1. Bild auf das richtige Gerät verschieben (z.B. GPU)
            test_image = test_image.to(device)
            # 2. Vorhersage mit dem Modell machen
            test_output = model(test_image)
            # 3. Sigmoid-Aktivierungsfunktion anwenden
            test_output = torch.nn.Sigmoid()(test_output)
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