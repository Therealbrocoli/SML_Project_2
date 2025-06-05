import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_c=64):
        super().__init__()

        # ---------- Encoder ----------
        self.conv00 = DoubleConv(in_channels, base_c)
        self.pool0 = nn.MaxPool2d(2)

        self.conv10 = DoubleConv(base_c, base_c * 2)
        self.pool1 = nn.MaxPool2d(2)

        self.conv20 = DoubleConv(base_c * 2, base_c * 4)
        self.pool2 = nn.MaxPool2d(2)

        self.conv30 = DoubleConv(base_c * 4, base_c * 8)
        self.pool3 = nn.MaxPool2d(2)

        self.conv40 = DoubleConv(base_c * 8, base_c * 16)

        # ---------- Decoder (dense) ----------
        self.conv01 = DoubleConv(base_c + base_c * 2, base_c)
        self.conv11 = DoubleConv(base_c * 2 + base_c * 4, base_c * 2)
        self.conv21 = DoubleConv(base_c * 4 + base_c * 8, base_c * 4)
        self.conv31 = DoubleConv(base_c * 8 + base_c * 16, base_c * 8)

        self.conv02 = DoubleConv(base_c * 2 + base_c, base_c)
        self.conv12 = DoubleConv(base_c * 4 + base_c * 2, base_c * 2)
        self.conv22 = DoubleConv(base_c * 8 + base_c * 4, base_c * 4)

        self.conv03 = DoubleConv(base_c * 2 + base_c, base_c)
        self.conv13 = DoubleConv(base_c * 4 + base_c * 2, base_c * 2)

        self.conv04 = DoubleConv(base_c * 2, base_c)

        self.final_conv = nn.Conv2d(base_c, out_channels, 1)

    # ---------- Helper ----------
    @staticmethod
    def _up(x, ref):
        """Upsample tensor x to the spatial size of ref (H, W)."""
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=True)

    # ---------- Forward ----------
    def forward(self, x):
        # Encoder
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))
        x30 = self.conv30(self.pool2(x20))
        x40 = self.conv40(self.pool3(x30))

        # Decoder – dense skip connections
        x01 = self.conv01(torch.cat([x00, self._up(x10, x00)], 1))
        x11 = self.conv11(torch.cat([x10, self._up(x20, x10)], 1))
        x21 = self.conv21(torch.cat([x20, self._up(x30, x20)], 1))
        x31 = self.conv31(torch.cat([x30, self._up(x40, x30)], 1))

        x02 = self.conv02(torch.cat([x01, self._up(x11, x01)], 1))
        x12 = self.conv12(torch.cat([x11, self._up(x21, x11)], 1))
        x22 = self.conv22(torch.cat([x21, self._up(x31, x21)], 1))

        x03 = self.conv03(torch.cat([x02, self._up(x12, x02)], 1))
        x13 = self.conv13(torch.cat([x12, self._up(x22, x12)], 1))

        x04 = self.conv04(torch.cat([x03, self._up(x13, x03)], 1))

        return self.final_conv(x04)


# ---------- Quick sanity check ----------
if __name__ == "__main__":
    model = UNetPlusPlus()
    dummy = torch.randn(1, 3, 256, 256)  # 256%16==0 ⇒ smooth ride
    out = model(dummy)
    print("Output shape:", out.shape)