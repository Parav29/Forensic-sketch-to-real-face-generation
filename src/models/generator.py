import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    """
    UNet-256: 8 downsampling stages, skip connections from encoder to decoder.
    Input: 3-channel sketch  ->  Output: 3-channel photo
    """

    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, ngf, 4, 2, 1)                 # 128
        self.enc2 = self._down(ngf,      ngf * 2)                        # 64
        self.enc3 = self._down(ngf * 2,  ngf * 4)                        # 32
        self.enc4 = self._down(ngf * 4,  ngf * 8)                        # 16
        self.enc5 = self._down(ngf * 8,  ngf * 8)                        # 8
        self.enc6 = self._down(ngf * 8,  ngf * 8)                        # 4
        self.enc7 = self._down(ngf * 8,  ngf * 8)                        # 2
        self.enc8 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1))  # 1 (bottleneck)

        # Decoder with skip connections
        self.dec8 = self._up(ngf * 8,   ngf * 8, dropout=True)
        self.dec7 = self._up(ngf * 16,  ngf * 8, dropout=True)
        self.dec6 = self._up(ngf * 16,  ngf * 8, dropout=True)
        self.dec5 = self._up(ngf * 16,  ngf * 8)
        self.dec4 = self._up(ngf * 16,  ngf * 4)
        self.dec3 = self._up(ngf * 8,   ngf * 2)
        self.dec2 = self._up(ngf * 4,   ngf)
        self.dec1 = nn.Sequential(nn.ReLU(True),
                                  nn.ConvTranspose2d(ngf * 2, out_channels, 4, 2, 1),
                                  nn.Tanh())

    def _down(self, in_ch, out_ch):
        return nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
        )

    def _up(self, in_ch, out_ch, dropout=False):
        layers = [
            nn.ReLU(True),
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        d8 = self.dec8(e8)
        d7 = self.dec7(torch.cat([d8, e7], 1))
        d6 = self.dec6(torch.cat([d7, e6], 1))
        d5 = self.dec5(torch.cat([d6, e5], 1))
        d4 = self.dec4(torch.cat([d5, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        return self.dec1(torch.cat([d2, e1], 1))
