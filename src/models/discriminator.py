import torch
import torch.nn as nn


class PatchGAN(nn.Module):
    """
    70x70 PatchGAN discriminator.
    Input: [sketch, photo] concatenated -> real/fake classification per patch.
    """

    def __init__(self, in_channels=6, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            # No norm on the first layer
            nn.Conv2d(in_channels, ndf,     4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf,         ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),                nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2,     ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),                nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4,     ndf * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),                nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8,     1,       4, 1, 1),   # output: 30x30 patch map
        )

    def forward(self, sketch, photo):
        x = torch.cat([sketch, photo], dim=1)
        return self.model(x)
