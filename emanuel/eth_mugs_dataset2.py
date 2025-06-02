"""ETH Mugs Dataset."""

import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class ETHMugsDataset(Dataset):
    """Torch dataset Version 2."""

    def __init__(self, root_dir, mode="train"):
        """This dataset class loads the ETH Mugs dataset.

        It will return the resized image according to the scale and mask tensors
        in the original resolution.

        Args:
            root_dir (str): Path to the root directory of the dataset.
            mode (str): Mode of the dataset. It can be "train", "val" or "test"
        """
        self.mode = mode
        self.root_dir = root_dir

        # get image and mask paths
        self.rgb_dir = os.path.join(self.root_dir, "rgb/")
        self.mask_dir = os.path.join(self.root_dir, "masks/")
        self.image_paths = sorted([os.path.join(self.rgb_dir, el) for el in os.listdir(self.rgb_dir) if el.endswith('_rgb.jpg')])

        if self.mode != "test":
            self.mask_paths = sorted([os.path.join(self.mask_dir, el) for el in os.listdir(self.mask_dir) if el.endswith('_mask.png')])
            assert len(self.image_paths) == len(self.mask_paths)

        # Image-Transform: PIL → Augmentierung → Tensor
        self.transform = transforms.Compose([
            transforms.Resize((252, 376)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandAugment(num_ops=2, magnitude=5),  # optional: just a light augmentation
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Maske: keine Augmentierung – nur Resize und ToTensor
        self.mask_transform = transforms.Compose([
            transforms.Resize((252, 376)),
            transforms.ToTensor()
        ])
        
        print("[INFO] Dataset mode:", mode)
        print("[INFO] Number of images in the ETHMugDataset: {}".format(len(self.image_paths)))

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Get an item from the dataset."""
        # TODO: load image and gt mask (unless when in test mode), apply transforms if necessary
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.transform(image)

        if self.mode != "test":
            mask_path = self.mask_paths[idx]
            orig_mask = Image.open(mask_path)
            mask = self.mask_transform(orig_mask)
            mask = (mask.int()==1)[0, :, :]
            mask = mask.unsqueeze(dim=0)

            return image, mask

        return image