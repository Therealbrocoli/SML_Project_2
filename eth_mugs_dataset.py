"""ETH Mugs Dataset."""


import os
from PIL import Image
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from utils import IMAGE_SIZE, load_mask



class ETHMugsDataset(Dataset):

    def __init__(self, root_dir, mode="train"):
        """This dataset class loads the ETH Mugs dataset.

        It will ***return the resized image*** according to the ***scale*** and mask tensors
        in the original resolution.

        Args:
            root_dir (str): Path to the root directory of the dataset.
            mode (str): Mode of the dataset. It can be "train", "val" or "test"
        """
        self.mode = mode
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(self.root_dir, "rgb")
        self.mask_dir = os.path.join(self.root_dir, "masks")

        # Get image and mask paths
        self.image_paths = [f for f in os.listdir(self.rgb_dir) if f.endswith(".jpg")]
        self.image_paths.sort()  # Sort to ensure consistent order

        # Define image transformations - these transforms will be applied to pre-process the data before passing it through the model
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # these are the mean and std values for ImageNet based  on the original paper
        ])

        print("[INFO] Dataset mode:", mode)
        print(
            "[INFO] Number of images in the ETHMugDataset: {}".format(len(self.image_paths))
        )

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Get an item from the dataset."""
        img_name = os.path.join(self.rgb_dir, self.image_paths[idx])
        image = Image.open(img_name).convert('RGB')
        if self.mode != "test":
            mask_name = os.path.join(self.mask_dir, self.image_paths[idx].replace('.jpg', '_mask.png'))
            mask = load_mask(mask_name)
        else:
            mask = torch.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=torch.int64)  # Dummy mask for test mode
        
        if self.transform:
            image = self.transform(image)

        return image, mask
