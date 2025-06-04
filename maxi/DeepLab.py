import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import IMAGE_SIZE  # Stelle sicher, dass IMAGE_SIZE in utils.py definiert ist

# --------------------
# Helper‐Baustein: ConvBlock
# --------------------
class ConvBlock(nn.Module):
    """
    Ein einfacher „Double‐Conv“-Baustein (wie in U-Net), jeweils Conv2d → BatchNorm → ReLU.
    Dieser Block wird im Decoder von DeepLabUnet verwendet.
    """
    def __init__(self, c_in, c_out):
        super(ConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# --------------------
# ASPP‐Modul
# --------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # Atrous Convs mit BatchNorm + ReLU
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
        # Zusammenführung:
        self.out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.image_pooling(x)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.out(x_cat)


# --------------------
# DeepLabUnet‐Architektur
# --------------------
class DeepLabUnet(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLabUnet, self).__init__()

        # 1) Backbone: ResNet-34 (ohne vortrainierte Gewichte)
        resnet = models.resnet34(weights=None)
        # „layer0“ (Conv1 → BN → ReLU → MaxPool) liefert 64×H/4×W/4
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 64×H/4×W/4
        self.layer2 = resnet.layer2  # 128×H/8×W/8
        self.layer3 = resnet.layer3  # 256×H/16×W/16
        self.layer4 = resnet.layer4  # 512×H/32×W/32

        # 2) ASPP auf den tiefsten Features
        self.aspp = ASPP(512, 256)

        # 3) Decoder: Skip‐Connections
        #    dec3: kombiniert ASPP-Output (256×H/32×W/32 hochskaliert auf H/16×W/16) + layer3 (256×H/16×W/16)
        self.dec3 = ConvBlock(256 + 256, 256)
        #    dec2: kombiniert dec3 (256×H/16×W/16 hochskaliert auf H/8×W/8) + layer2 (128×H/8×W/8)
        self.dec2 = ConvBlock(256 + 128, 128)
        #    dec1: kombiniert dec2 (128×H/8×W/8 hochskaliert auf H/4×W/4) + layer1 (64×H/4×W/4)
        self.dec1 = ConvBlock(128 + 64, 64)

        # 4) Finales 1×1‐Conv auf Originalgröße
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Backbone:
        x0 = self.layer0(x)   # 64×(H/4)×(W/4)
        x1 = self.layer1(x0)  # 64×(H/4)×(W/4)
        x2 = self.layer2(x1)  # 128×(H/8)×(W/8)
        x3 = self.layer3(x2)  # 256×(H/16)×(W/16)
        x4 = self.layer4(x3)  # 512×(H/32)×(W/32)

        # ASPP auf tiefsten Features
        a = self.aspp(x4)     # 256×(H/32)×(W/32)

        # Decoder‐Stufe 1 (H/16 × W/16)
        a_up3 = F.interpolate(a, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([a_up3, x3], dim=1))  # 256×(H/16)×(W/16)

        # Decoder‐Stufe 2 (H/8 × W/8)
        d3_up2 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d3_up2, x2], dim=1))  # 128×(H/8)×(W/8)

        # Decoder‐Stufe 3 (H/4 × W/4)
        d2_up1 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2_up1, x1], dim=1))   # 64×(H/4)×(W/4)

        # Auf Originalgröße hochskalieren
        out = F.interpolate(d1, size=IMAGE_SIZE, mode='bilinear', align_corners=False)  # 64×H×W
        logits = self.final_conv(out)  # num_classes×H×W

        return logits