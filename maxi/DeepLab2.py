import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import *
#externalisierte Funktion aus train_DeepLab.py 

class DeepLab(nn.Module):
    def __init__(self, num_classes=1):  # eine Klasse wird segmentiert (ETH Tassen)
        super(DeepLab, self).__init__()
        # Lade ein ResNet-Modell ohne vortrainierte Gewichte
        resnet = models.resnet18(weights=None)  ###
        # Entferne den letzten Fully-Connected-Layer und das durchschnittliche Pooling
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Atrous Spatial Pyramid Pooling (ASPP) Modul
        self.aspp = ASPP(512, 256)  ###

        # Dropout nach ASPP zur Regularisierung
        self.dropout = nn.Dropout2d(p=0.5)  ###

        # Letzte Schicht zur Erzeugung der Segmentierungskarte
        self.fc = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Extrahiere Merkmale mit dem Backbone
        x = self.backbone(x)

        # Wende ASPP an
        x = self.aspp(x)
        x = self.dropout(x)

        # Wende die letzte Schicht an, um die Segmentierungskarte (Logits) zu erhalten
        x = self.fc(x)

        # Interpoliere auf die ursprüngliche Bildgröße
        x = F.interpolate(x, size=IMAGE_SIZE, mode='bilinear', align_corners=False)

        return x  # roher Logit statt Sigmoid

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # Atrous Convolution mit unterschiedlichen Raten und nachfolgendem BatchNorm+ReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)  
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=6, padding=6, bias=False),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)  
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=12, padding=12, bias=False),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)  
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=18, padding=18, bias=False),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)  
        )

        # Image Pooling
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)  
        )

        # Ausgabeschicht nach dem Zusammenführen
        self.out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)  
        )

    def forward(self, x):
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

        return x
