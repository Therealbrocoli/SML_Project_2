import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
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

        self.rgb_dir = os.path.join(self.root_dir, "rgb/")
        self.mask_dir = os.path.join(self.root_dir, "masks/")
        self.image_paths = sorted([os.path.join(self.rgb_dir, el) for el in os.listdir(self.rgb_dir) if el.endswith('_rgb.jpg')])

        if self.mode != "test":
            self.mask_paths = sorted([os.path.join(self.mask_dir, el) for el in os.listdir(self.mask_dir) if el.endswith('_mask.png')])
            assert len(self.image_paths) == len(self.mask_paths)

        # Common transformations for both image and mask (geometric transformations)
        # Using transforms.Compose with transforms.RandomApply and transforms.Augmentations is one way,
        # or custom transform that applies to both.
        # For simplicity, let's redefine the transforms to apply to both image and target
        # using functional transforms or by creating a custom transform.
        # However, torchvision.transforms.v2 automatically handles this if you pass a dictionary
        # to the transform or use transforms.Compose with transforms.RandomApply(transforms.Augmentations)

        # Image and Mask Transformation
        # Define a joint transform for geometric augmentations that apply to both image and mask
        self.joint_transform = transforms.Compose([
            transforms.Resize((252, 376)),          # Resize both to the target size
            transforms.RandomHorizontalFlip(p=0.5), # Random horizontal flip
            transforms.RandomVerticalFlip(p=0.5),   # Random vertical flip
            transforms.RandomRotation(10),          # Random rotation within 10 degrees
        ])

        # Image-specific transforms (after joint transforms)
        self.image_only_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # Helligkeit, Kontrast, Sättigung, Farbton
            transforms.RandomAdjustSharpness(sharpness_factor=2),                           # Schärfe
            transforms.RandomAutocontrast(p=0.5),                                           # Kontrast
            transforms.RandomEqualize(p=0.5),                                               # Histogram-Ausgleich
            transforms.ToImage(),                                                           # Konvertiert zu PIL Image
            transforms.ToDtype(torch.float32, scale=True),                                  # Konvertiert zu Tensor       
            transforms.Normalize((0.5,), (0.5,))                                            # Normalisiert den Tensor
        ])

        # Mask-specific transforms (after joint transforms)
        self.mask_only_transform = transforms.Compose([
            transforms.ToImage(),                           # Konvertiert zu PIL Image
            transforms.ToDtype(torch.float32, scale=True),  # Konvertiert zu Tensor      
        ])


        print("[INFO] Dataset-Modus:", mode)
        print("[INFO] Anzahl der Bilder im ETHMugDataset: {}".format(len(self.image_paths)))

    def __len__(self):
        """Gibt die Länge des Datensatzes zurück."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Holt ein Element aus dem Datensatz."""
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.mode != "test":
            mask_path = self.mask_paths[idx]
            orig_mask = Image.open(mask_path).convert('L') # Ensure mask is grayscale

            # Apply joint transforms
            # For v2 transforms, you can apply them to a dictionary of { "image": image, "mask": mask }
            # Or define custom functional transforms for this.
            # A simple approach for now, assuming PIL Image for joint transforms
            # and then converting to tensor separately.

            # Create a dictionary for v2 transforms to handle both
            data = {'image': image, 'mask': orig_mask}
            transformed_data = self.joint_transform(data) # Apply transforms to both image and mask

            image = transformed_data['image']
            mask = transformed_data['mask']

            # Apply image-only and mask-only transforms
            image = self.image_only_transform(image)
            mask = self.mask_only_transform(mask)

            # Convert mask to binary (0 or 1) and ensure correct shape
            # Assuming ToTensor converts 0-255 to 0.0-1.0. So, check for values > 0.
            mask = (mask > 0).float() # Keep it float for BCEWithLogitsLoss target

            return image, mask

        else:
            # Apply joint transforms (only to image, as there's no mask)
            # and then image-only transforms
            image = self.joint_transform({'image': image})['image']
            image = self.image_only_transform(image)
            return image
