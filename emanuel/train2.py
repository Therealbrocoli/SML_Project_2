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

    # Überprüfen, ob CUDA verfügbar ist, und Gerät festlegen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]: Verwende Gerät: {device}")

    # Hyperparameter
    train_batch_size = 8
    test_batch_size = 1
    num_epochs = 10
    learning_rate = 1e-4

    print(f"[INFO]: Anzahl der Trainingsepochen: {num_epochs}")

    # Erstellen der Datensätze und DataLoader
    train_dataset = ETHMugsDataset(root_dir=train_data_root, mode="train")
    test_dataset = ETHMugsDataset(root_dir=test_data_root, mode="test")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # Verzeichnis zum Speichern der Vorhersagen
    out_dir = os.path.join('prediction')
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO]: Speichere die vorhergesagten Segmentierungsmasken nach {out_dir}")

    # Initialisierung des Modells, der Verlustfunktion und des Optimierers
    model = build_model().to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("[INFO]: Starte das Training...")
    for epoch in range(num_epochs):
        model.train()
        print(f'Start der Epoche {epoch + 1}/{num_epochs}')
        print('------------------------------')

        for i, (image, gt_mask) in enumerate(train_dataloader):
            image, gt_mask = image.to(device), gt_mask.to(device)

            optimizer.zero_grad()
            output = model(image)
            output = torch.nn.Sigmoid()(output)
            loss = criterion(output, gt_mask.float())

            loss.backward()
            optimizer.step()

            print(f"Trainingsverlust: {loss.data.cpu().numpy()}, IoU: {compute_iou((output.data.cpu().numpy() > 0.5), gt_mask.data.cpu().numpy())}")

        # Speichern des Modell-Checkpoints nach jeder Epoche
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last_epoch.pth"))

    # Evaluierung mit Testdaten
    model.eval()
    image_ids, pred_masks = [], []

    with torch.no_grad():
        for i, test_image in enumerate(test_dataloader):
            test_image = test_image.to(device)
            test_output = model(test_image)
            test_output = torch.nn.Sigmoid()(test_output)
            pred_mask = (test_output > 0.5).squeeze().cpu().numpy()

            # Speichern der vorhergesagten Maske als Bild
            Image.fromarray(pred_mask).save(os.path.join(out_dir, f"{str(i).zfill(4)}_mask.png"))

            image_ids.append(str(i).zfill(4))
            pred_masks.append(pred_mask)

    # Speichern der Vorhersagen im Kaggle-Einreichungsformat
    save_predictions(image_ids=image_ids, pred_masks=pred_masks, save_path=os.path.join(out_dir, 'submission.csv'))

if __name__ == "__main__":
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

    train(ckpt_dir, train_data_root, test_data_root)