"""ETH Mugs Dataset."""
import os
from PIL import Image
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from utils import IMAGE_SIZE, load_mask

class ETHMugsDataset(Dataset): #maxi: Definiert eine neue Dataset-Klasse für ETH Mugs, basierend auf PyTorchs Dataset.
    def __init__(self, root_dir, mode="train"): #maxi: Initialisierungsmethode. Wird beim Erstellen des Objekts aufgerufen. Nimmt das Arbeitsverzeichnis und den Modus entgegen.

        self.mode = mode #maxi: string der "train" oder "test",heisst je nach dem mit was wir arbeiten wollen

        self.root_dir = root_dir #maxi: Speichert den PATH von ./dataset/test_data oder von ./dataset/train_data je nach dem was für einen Mode im Terminal gewählt wurde
        self.rgb_dir = os.path.join(self.root_dir, "rgb") #maxi: Setzt den Pfad zum RGB-Ordner (mit Bildern) innerhalb von root_dir.
        self.mask_dir = os.path.join(self.root_dir, "masks") #maxi: Setzt den Pfad zum Masken-Ordner (mit Segmentierungs-Masken).
        
        self.image_paths = [f for f in os.listdir(self.rgb_dir) if f.endswith(".jpg")] #maxi: Liest alle .jpg-Dateien im RGB-Ordner und speichert sie als Liste.
        self.image_paths.sort() #maxi: Sortiert die Bildliste alphabetisch für konsistente Reihenfolge.
        
        self.transform = None #maxi: macht die Transformation
        self.mask_transform = None #maxi: macht die Transformation

        """
        # Define image transformations
        self.transform = transforms.Compose([ # Definiert eine Pipeline für Bild-Transformations- und Vorverarbeitungsschritte (wie Flip, Rotation, Resize, ToTensor, Normalize).
            transforms.RandomHorizontalFlip(), #Transformationsschritt
            transforms.RandomRotation(10), #Transformationsschritt
            transforms.Resize(IMAGE_SIZE), #Transformationsschritt
            transforms.ToTensor(), #Transformationsschritt
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            #Transformationsschritt
            #https://docs.pytorch.org/vision/stable/transforms.html
        ])
        """
        """
        # Define mask transformations
        self.mask_transform = transforms.Compose([
            transforms.Resize((252, 376)),  # Resize to match the model's output dimensions
            transforms.ToTensor(),
        ])
        """

        print("[INFO] Dataset mode:", mode) #maxi: Gibt beim Initialisieren den Modus aus.
        print("[INFO] Number of images in the ETHMugDataset:", len(self.image_paths)) #naxi: Gibt die Bildanzahl im Dataset aus.

    def __len__(self): #maxi: Implementiert die Längenfunktion, die zurückgibt, wie viele Bilder vorhanden sind.
        return len(self.image_paths) #maxi: Gibt die Länge der Bildliste zurück.

    def __getitem__(self, idx: int): #maxi: Ermöglicht das Zugreifen auf das idx-te Bild+Maske mit eckigen Klammern (z.B. dataset[5]).
        img_name = os.path.join(self.rgb_dir, self.image_paths[idx]) #maxi: Baut den Dateipfad zum idx-ten Bild.
        image = Image.open(img_name).convert('RGB') #maxi: Öffnet das Bild und wandelt es zu RGB um.

        if self.mode == "train":
            # Extract the base filename without extension and any suffix like '_rgb'
            img_base_name = os.path.splitext(self.image_paths[idx])[0].split('_rgb')[0] #maxi: Holt sich den Basisnamen ohne .jpg und optionales '_rgb' für die Maskensuche.

            # Construct the mask filename based on the actual naming convention
            mask_name = os.path.join(self.mask_dir, f"{img_base_name}_mask.png") #maxi: Sucht den Maskenpfad, der zu diesem Bild gehört.

            # Debugging information
            # print(f"[DEBUG] Looking for mask at: {mask_name}")

            if not os.path.exists(mask_name):
                raise FileNotFoundError(f"The mask file {mask_name} does not exist.")

            mask = Image.open(mask_name).convert('L')  # Load mask as grayscale
            mask = self.mask_transform(mask)  # Apply mask transformations
        else:
            # In test mode, return a dummy mask or None if masks are not required
            mask = torch.zeros((1, 252, 376), dtype=torch.float32)  # Dummy mask for test mode

        if self.transform:
            image = self.transform(image)

        return image, mask
