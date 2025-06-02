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
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
# Definiert eine Funktion, die das Modell erzeugt.
def build_model():
    # Beschreibt in einem Docstring, dass hier das Modell gebaut wird.
    """Build the model."""
    # Gibt ein FCN-Modell zurück, das 3 Eingangskanäle (RGB) und 1 Ausgangskanal (Maske) besitzt.
    return FCN(in_channels=3, out_channels=1)

# Definiert die Trainingsfunktion mit Speicherorten für Checkpoints und Daten als Argumente.
def train(
    ckpt_dir: str,
    train_data_root: str,
    val_data_root: str,

):
    # Legt fest, wie oft während des Trainings Logs ausgegeben werden.
    log_frequency = 10
    # Setzt die Batchgröße für die Validierung auf 1.
    val_batch_size = 1
    # Gibt an, nach wie vielen Epochen eine Validierung durchgeführt wird.
    val_frequency = 1

    # Anzahl der Trainingsepochen wird auf 10 gesetzt.
    num_epochs = 10
    # Lernrate für den Optimierer.
    lr = 1e-4
    # Batchgröße fürs Training.
    train_batch_size = 8

    # Gibt die gewählte Anzahl Epochen aus.
    print(f"[INFO]: Number of training epochs: {num_epochs}")
    # Gibt die gewählte Lernrate aus.
    print(f"[INFO]: Learning rate: {lr}")
    # Gibt die Trainings-Batchgröße aus.
    print(f"[INFO]: Training batch size: {train_batch_size}")

    # Prüft, ob eine GPU verfügbar ist und wählt das richtige Device.
    if torch.cuda.is_available():
         device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    # Gibt das verwendete Device (CPU/GPU) aus.
    print(f"[INFO]: Using device: {device}")

    # Definiert eine Bildtransformation (Resize und ToTensor) als Pipeline.
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # Erstellt ein Trainings-Dataset-Objekt mit den Trainingsdaten.
    train_dataset = ETHMugsDataset(root_dir=train_data_root, mode="train")
    # Erzeugt einen DataLoader für das Training mit definierter Batchgröße und zufälliger Durchmischung.
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # Erstellt ein Test-Dataset-Objekt für die Validierung.
    test_dataset = ETHMugsDataset(root_dir=val_data_root, mode="val")
    # Erstellt einen DataLoader für die Validierung ohne Shuffle.
    test_dataloader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    # Legt den Pfad zum Ordner für die Vorhersagen fest.
    out_dir = os.path.join('prediction')
    # Erstellt den Ausgabeordner, falls dieser noch nicht existiert.
    os.makedirs(out_dir, exist_ok=True)
    # Gibt aus, wohin die Masken gespeichert werden.
    print(f"[INFO]: Saving the predicted segmentation masks to {out_dir}")

    # Baut das Modell mit der zuvor definierten Funktion.
    model = build_model()
    # Verschiebt das Modell auf das gewählte Device (CPU oder GPU).
    model.to(device)

    # Definiert die Loss-Funktion für binäre Klassifikation (Segmentierung).
    criterion = torch.nn.BCELoss()
    # Initialisiert den Adam-Optimizer mit Lernrate.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Erstellt einen Scheduler, der die Lernrate alle 10 Epochen um den Faktor 0.1 reduziert.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Gibt erneut das verwendete Device aus (redundant).
    print(f"[INFO]: Using device: {device}")
    # Gibt aus, dass das Training startet.
    print("[INFO]: Starting training...")

    # Schleife über alle Trainingsepochen.
    for epoch in range(num_epochs):
        # Setzt das Modell in den Trainingsmodus.
        model.train()
        # Druckt Trenner für die Übersichtlichkeit.
        print('****************************')
        # Gibt die aktuelle Epoche aus.
        print(epoch)
        # Druckt noch einen Trenner.
        print('****************************')

        # Schleife über alle Trainings-Batches.
        for image, gt_mask in train_dataloader:
            # Verschiebt die Eingabebilder auf das Device.
            image = image.to(device)
            # Verschiebt die Ground-Truth-Masken auf das Device.
            gt_mask = gt_mask.to(device)
            # Setzt die Gradienten im Optimierer auf Null zurück.
            optimizer.zero_grad()
            # Berechnet die Modellvorhersage für die Bilder.
            output = model(image)
            # Berechnet den Loss zwischen Vorhersage und Ground Truth.
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

        # Führt Validierung aus, wenn die Epoche das Validierungsintervall trifft.
        if epoch % val_frequency == 0:
            # Setzt das Modell in den Evaluierungsmodus.
            model.eval()
            # Initialisiert eine Liste für Bild-IDs.
            image_ids = []
            # Initialisiert eine Liste für vorhergesagte Masken.
            pred_masks = []

            # Startet einen Kontext ohne Gradientenberechnung (spart Speicher/Zeit).
            with torch.no_grad():
                # Schleife über alle Testbilder im DataLoader.
                for i, (image, _) in enumerate(test_dataloader):
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

                # Speichert alle Vorhersagen im Submission-Format als CSV-Datei.
                save_predictions(image_ids=image_ids, pred_masks=pred_masks, save_path=os.path.join(out_dir, 'submission.csv'))
                print(f"[INFO]: Predictions saved to {os.path.join(out_dir, 'submission.csv')}")

# Prüft, ob das Skript direkt ausgeführt wird.
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
    # Parst die übergebenen Argumente.
    args = parser.parse_args()
    # Holt die aktuelle Zeit.
    now = datetime.now()
    # Formatiert das Datum als String für den Ordnernamen.
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # Verbindet Checkpoint-Pfad und Zeitstring zu neuem Verzeichnis.
    ckpt_dir = os.path.join(args.ckpt_dir, dt_string)
    # Erstellt das Checkpoint-Verzeichnis, falls nicht vorhanden.
    os.makedirs(ckpt_dir, exist_ok=True)
    # Gibt aus, wo Checkpoints gespeichert werden.
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)
    # Gibt einen Hinweis zu den Vorhersage-Dateien.
    print("PLEASE ARCHIVE PREDICTIONS FOLDER AND RENAME THE FOLDER TO predictions{number}.csv")
    # Setzt den Trainingsdaten-Ordner.
    train_data_root = os.path.join(args.data_root, "train_data")
    # Gibt den Trainingsdaten-Ordner aus.
    print(f"[INFO]: Train data root: {train_data_root}")
    # Setzt den Validierungsdaten-Ordner.
    val_data_root = os.path.join(args.data_root, "test_data")
    # Gibt den Validierungsdaten-Ordner aus.
    print(f"[INFO]: Test data root: {val_data_root}")
    # Startet das Training mit den definierten Pfaden.
    train(ckpt_dir, train_data_root, val_data_root)