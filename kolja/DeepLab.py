

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import *
#externalisierte Funktion aus train_DeepLab.py 
# === ANSI TERMINAL==
BOLD = "\033[1m"
RED = "\033[91m"
RESET = "\033[0m"

class DeepLab(nn.Module):
    def __init__(self, num_classes=1):#eine Klasse wird segmentiert (Eth Tassen
        print(f"{RED}[INFO]: DeepLab__init__ has been entered{RESET}")
        super(DeepLab, self).__init__()

        # Lade ein ResNet-Modell ohne vortrainierte Gewichte
        resnet = models.resnet50(pretrained=False)

        # Entferne den letzten Fully-Connected-Layer und das durchschnittliche Pooling
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Atrous Spatial Pyramid Pooling (ASPP) Modul
        self.aspp = ASPP(2048, 256)

        # Letzte Schicht zur Erzeugung der Segmentierungskarte
        self.fc = nn.Conv2d(256, num_classes, kernel_size=1)

        print(f"{RED}[INFO]: DeepLab__init__ ist abgeschlossen{RESET}")

    def forward(self, x):

        print(f"{RED}[INFO]: DeepLab_method forward has been entered{RESET}")
        # Extrahiere Merkmale mit dem Backbone
        x = self.backbone(x)

        # Wende ASPP an
        x = self.aspp(x)

        # Wende die letzte Schicht an, um die Segmentierungskarte zu erhalten
        x = self.fc(x)

        # Interpoliere auf die ursprüngliche Bildgröße
        x = F.interpolate(x, size=(252, 376), mode='bilinear', align_corners=False)

        print(f"{RED}[INFO]: DeepLab_method forward ist abgeschlossen{RESET}")

        return x # sigmoid wird in train gemacht ETH Tasse (1) oder Hintergrund (0)
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        print(f"{RED}[INFO]: ASPP__init__ has been entered{RESET}")
        super(ASPP, self).__init__()
        
        # Atrous Convolution mit unterschiedlichen Raten
#Optimierungsmoeglichkeiten
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=6, padding=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=12, padding=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=18, padding=18)

        # Image Pooling
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

        self.out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        print(f"{RED}[INFO]: ASPP__init__ ist abgeschlossen{RESET}")

    def forward(self, x):
        print(f"{RED}[INFO]: DeepLab_method forward has been entered{RESET}")
        # Wende verschiedene atrous convolutions an
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        # Image Pooling
        x5 = self.image_pooling(x)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=False)

        # Verkette die Ergebnisse und wende eine 1x1 convolution an
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.out(x)
        print(f"{RED}[INFO]: DeepLab_method forward ist abgeschlossen{RESET}")

        return x

