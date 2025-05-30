"""ETH Mugs Dataset."""


import os
from PIL import Image
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from utils import IMAGE_SIZE, load_mask

class ETHMugsDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        self.mode = mode
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(self.root_dir, "rgb")
        self.mask_dir = os.path.join(self.root_dir, "masks")

        self.image_paths = [f for f in os.listdir(self.rgb_dir) if f.endswith(".jpg")]
        self.image_paths.sort()

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #https://docs.pytorch.org/vision/stable/transforms.html
        ])

        # Define mask transformations
        self.mask_transform = transforms.Compose([
            transforms.Resize((252, 376)),  # Resize to match the model's output dimensions
            transforms.ToTensor(),
        ])

        print("[INFO] Dataset mode:", mode)
        print("[INFO] Number of images in the ETHMugDataset:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_name = os.path.join(self.rgb_dir, self.image_paths[idx])
        image = Image.open(img_name).convert('RGB')

        if self.mode == "train":
            # Extract the base filename without extension and any suffix like '_rgb'
            img_base_name = os.path.splitext(self.image_paths[idx])[0].split('_rgb')[0]

            # Construct the mask filename based on the actual naming convention
            mask_name = os.path.join(self.mask_dir, f"{img_base_name}_mask.png")

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
