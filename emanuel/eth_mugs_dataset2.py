import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ETHMugsDataset(Dataset):
    """Torch-Dataset Version 2."""

    def __init__(self, root_dir, mode="train"):
        """Diese Dataset-Klasse lädt das ETH Mugs Dataset.

        Es gibt das Bild in der entsprechend skalierten Größe und die Masken-Tensoren
        in der ursprünglichen Auflösung zurück.

        Args:
            root_dir (str): Pfad zum Stammverzeichnis des Datensatzes.
            mode (str): Modus des Datensatzes. Es kann "train", "val" oder "test" sein.
        """
        self.mode = mode
        self.root_dir = root_dir

        # Pfade zu Bildern und Masken erhalten
        self.rgb_dir = os.path.join(self.root_dir, "rgb/")
        self.mask_dir = os.path.join(self.root_dir, "masks/")
        self.image_paths = sorted([os.path.join(self.rgb_dir, el) for el in os.listdir(self.rgb_dir) if el.endswith('_rgb.jpg')])

        if self.mode != "test":
            self.mask_paths = sorted([os.path.join(self.mask_dir, el) for el in os.listdir(self.mask_dir) if el.endswith('_mask.png')])
            assert len(self.image_paths) == len(self.mask_paths)

        # Bild-Transformation: PIL → Augmentierung → Tensor
        self.transform = transforms.Compose([
            transforms.Resize((252, 376)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandAugment(num_ops=2, magnitude=5),  # optional: nur eine leichte Augmentierung
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Maske: keine Augmentierung – nur Resize und ToTensor
        self.mask_transform = transforms.Compose([
            transforms.Resize((252, 376)),
            transforms.ToTensor()
        ])

        print("[INFO] Dataset-Modus:", mode)
        print("[INFO] Anzahl der Bilder im ETHMugDataset: {}".format(len(self.image_paths)))

    def __len__(self):
        """Gibt die Länge des Datensatzes zurück."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Holt ein Element aus dem Datensatz."""
        # TODO: Lade Bild und GT-Maske (außer im Testmodus), wende Transformationen an, falls notwendig
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.transform(image)

        if self.mode != "test":
            mask_path = self.mask_paths[idx]
            orig_mask = Image.open(mask_path)
            mask = self.mask_transform(orig_mask)
            mask = (mask.int() == 1)[0, :, :]
            mask = mask.unsqueeze(dim=0)

            return image, mask

        return image
