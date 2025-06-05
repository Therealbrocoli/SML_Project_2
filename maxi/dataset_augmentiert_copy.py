import time
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from utils import IMAGE_SIZE  #(W, H)

# =====ANSI Terminal===
BOLD = "\033[1m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

class ETHMugsDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        self.mode = mode
        self.root_dir = root_dir
        self.rgb_dir  = os.path.join(root_dir, "rgb")
        self.mask_dir = os.path.join(root_dir, "masks") if mode=="train" else None
        self.image_list = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith(".jpg")])
        self.N = len(self.image_list)
        self.mean, self.std = [0.427,0.419,0.377], [0.234,0.225,0.236]

    def __len__(self):
        if self.mode=="train":
            return self.N   # return self.N*5 Original + 4 Augmentierungen
        else:
            return self.N

    def __getitem__(self, idx):
        if self.mode=="train":
            base_idx = idx // 2
            aug_idx  = idx % 2
        else:
            base_idx = idx
            aug_idx  = 0

        fname = self.image_list[base_idx]
        img_path = os.path.join(self.rgb_dir, fname)
        image = Image.open(img_path).convert("RGB")

        if self.mode=="train":
            # 1. Maske laden
            base_name = os.path.splitext(fname)[0].split("_rgb")[0]
            mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Maske fehlt: {mask_path}")
            mask = Image.open(mask_path).convert("L")

            # 2. Je nach aug_idx bestimmen, welche Augmentierung
            #    aug_idx==0: Original, keine Veränderung vor Resize
            #    aug_idx==1: Flip+Sharpness
            #    aug_idx==2: Rotation+ColorJitter1
            #    aug_idx==3: ColorJitter2
            #    aug_idx==4: Solarize

            if aug_idx == 0:
                # RandomCrop(threshold=128, p=0.3)
                # 1. Zufälligen Crop-Parameter holen (Scale und Ratio nach Wunsch anpassen)
                i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.3, 1), ratio=(1.0, 1.0))
                # 2. Bild und Maske synchron croppen und auf IMAGE_SIZE skalieren
                image = TF.resized_crop(image, i, j, h, w, IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR)
                mask  = TF.resized_crop(mask,  i, j, h, w, IMAGE_SIZE, interpolation=InterpolationMode.NEAREST)
                if random.random() < 0.75:
                    image = TF.vflip(image)
                    mask  = TF.vflip(mask)

                if random.random() < 0.75:
                    image = TF.hflip(image)
                    mask  = TF.hflip(mask)
                image = TF.resize(image, IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR)
                mask  = TF.resize(mask,  IMAGE_SIZE, interpolation=InterpolationMode.NEAREST)

            """
            elif aug_idx == 1:
                # Keine zufälligen Schritte, nur Resize → Tensor → Norm
                image = TF.resize(image, IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR)
                mask  = TF.resize(mask,  IMAGE_SIZE, interpolation=InterpolationMode.NEAREST)

            elif aug_idx == 2:
                # HorizontalFlip + AdjustSharpness
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
                # RandomAdjustSharpness nur auf Bild:
                image = TF.adjust_sharpness(image, sharpness_factor=2.0)
                image = TF.resize(image, IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR)
                mask  = TF.resize(mask,  IMAGE_SIZE, interpolation=InterpolationMode.NEAREST)

            elif aug_idx == 3:
                # RandomRotation ≤ 30° + ColorJitter(bright=0.2,contrast=0.2,sat=0.2,hue=0.02)
                angle = random.uniform(-30, 30)
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask  = TF.rotate(mask,  angle, interpolation=InterpolationMode.NEAREST)
                color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                      saturation=0.2, hue=0.02)
                image = color_jitter(image)
                image = TF.resize(image, IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR)
                mask  = TF.resize(mask,  IMAGE_SIZE, interpolation=InterpolationMode.NEAREST)

            elif aug_idx == 4:
                # VerticalFlip
                image = TF.vflip(image)
                mask  = TF.vflip(mask)
                # ColorJitter(contrast=0.5, saturation=0.5, hue=0.1)
                color_jitter = transforms.ColorJitter(contrast=0.5, saturation=0.5, hue=0.1)
                image = color_jitter(image)
                image = TF.resize(image, IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR)
                mask  = TF.resize(mask,  IMAGE_SIZE, interpolation=InterpolationMode.NEAREST)
                """

            # 3. In Tensor + Normalize umwandeln
            image = TF.to_tensor(image)
            image = TF.normalize(image, self.mean, self.std)
            mask  = TF.to_tensor(mask)  # ergibt Float in [0,1]; ∵ binäre Maske

            return image, mask

        else:
            # Validierungs-/Testmodus: nur Resize, ToTensor, Normalize; Dummy-Maske falls nötig
            image = TF.resize(image, IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR)
            image = TF.to_tensor(image)
            image = TF.normalize(image, self.mean, self.std)
            if self.mode=="val":
                base_name = os.path.splitext(fname)[0].split("_rgb")[0]
                mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.png")
                mask = Image.open(mask_path).convert("L")
                mask = TF.resize(mask, IMAGE_SIZE, interpolation=InterpolationMode.NEAREST)
                mask = TF.to_tensor(mask)
            else:
                # Für Testmodus: Dummy-Maske (0er-Tensor)
                mask = torch.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=torch.float32)

            return image, mask



if __name__ == "__main__":

    #1. Dataset
    t = time.perf_counter()
    print(f"{GREEN}[INFO]: dataset starts{RESET}")
    dataset = ETHMugsDataset("datasets/train_data", mode="train")
    print(f"{GREEN}[TIME]: dataset done: {time.perf_counter()-t:.3f}{RESET}")

    #2. Dataloader
    t = time.perf_counter()
    print(f"{GREEN}[INFO]: loader starts{RESET}")
    loader  = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    print(f"{GREEN}[TIME]: loader done: {time.perf_counter()-t:.3f}{RESET}")

    print(f"{len(dataset)} for Training")
    print(f"dataset mode: {dataset.mode}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Nur die ersten fünf Bilder/Maske pro Augmentierung abspeichern
    out_root = "datasets/augmented_data"

    # Unterordner 1–5 anlegen (falls nicht vorhanden)
    for i in range(1, 2):
        os.makedirs(os.path.join(out_root, f"rgb{i}"),  exist_ok=True)
        os.makedirs(os.path.join(out_root, f"mask{i}"), exist_ok=True)

    # Counter für jeden Augmentations‐Index (0..4)
    counters = [0]

    # Unnormalisierung vorbereiten (wie gehabt)
    inv_mean = [-m/s for m, s in zip(dataset.mean, dataset.std)]
    inv_std  = [1.0/s   for s   in dataset.std]
    unnormalize = transforms.Normalize(mean=inv_mean, std=inv_std)

    print(f"{GREEN}[INFO]: Speichere maximal 3 Bilder/Masks pro Augmentierung …{RESET}")

    for idx in range(len(dataset)):
        # 1) Basis‐Index und Aug‐Index bestimmen
        base_idx = idx // 1       # welches Originalbild
        aug_idx  = idx % 1        # 0=Original, 1..4=Augmentierungen

        # 2) Prüfen, ob wir für diesen aug_idx bereits 5 abgespeichert haben
        if counters[aug_idx] >= 3:
            # Wenn alle Indizes 0..4 jeweils 5 erreicht haben, können wir abbrechen
            if all(c >= 3 for c in counters):
                break
            else:
                continue

        # 3) Bild‐Tensor und Masken‐Tensor aus dem Dataset holen
        img_tensor, mask_tensor = dataset[idx]

        # 4) Ursprünglicher Dateiname (Basisname) ermitteln
        orig_fname = dataset.image_list[base_idx]
        name_wo_ext = os.path.splitext(orig_fname)[0]

        # 5) Unnormalisieren + PIL‐Image erzeugen
        img_unnorm = unnormalize(img_tensor).clamp(0.0, 1.0)
        img_pil = TF.to_pil_image(img_unnorm)  # RGB

        # 6) Zielpfade zusammenbauen
        rgb_subdir  = os.path.join(out_root, f"rgb{aug_idx+1}")
        mask_subdir = os.path.join(out_root, f"mask{aug_idx+1}")
        save_name = f"{name_wo_ext}.png"

        img_path_to_save  = os.path.join(rgb_subdir,  save_name)
        mask_path_to_save = os.path.join(mask_subdir, save_name)

        # 7) Abspeichern
        img_pil.save(img_path_to_save)

        # Maske: Float‐Tensor in [0,1] → PIL ("L") → abspeichern
        mask_pil = TF.to_pil_image(mask_tensor)
        mask_pil.save(mask_path_to_save)

        # 8) Counter hochzählen
        counters[aug_idx] += 1

    print(f"{GREEN}[INFO]: Fertig – {counters} Bilder/Masks pro Augmentierung abgelegt.{RESET}")
    # ─────────────────────────────────────────────────────────────────────────────


    #4. Dataset_test
    t = time.perf_counter()
    print(f"{GREEN}[INFO]: dataset starts{RESET}")
    dataset_test = ETHMugsDataset("datasets/test_data", mode="test")
    print(f"{GREEN}[TIME]: dataset done: {time.perf_counter()-t:.3f}{RESET}")

    #5. Dataloader_test
    t = time.perf_counter()
    print(f"{GREEN}[INFO]: loader starts{RESET}")
    loader_test  = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    print(f"{GREEN}[TIME]: loader done: {time.perf_counter()-t:.3f}{RESET}")

    print(f"{len(dataset_test)} for Testing")
    print(f"dataset mode: {dataset_test.mode}")