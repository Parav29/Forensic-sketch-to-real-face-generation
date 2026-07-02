import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as sn


def _conv(in_ch, out_ch, stride, spectral=True):
    layer = nn.Conv2d(in_ch, out_ch, 4, stride, 1)
    return sn(layer) if spectral else layer


class PatchGAN(nn.Module):
    """
    70x70 PatchGAN discriminator.

    Input: [sketch, photo] concatenated -> real/fake classification per patch.

    Upgrades:
      * Spectral Normalization on every conv (replaces InstanceNorm) to enforce
        a Lipschitz constraint and stabilise adversarial training.
      * ``forward(..., return_features=True)`` also returns the intermediate
        activations, which the training loop consumes for feature-matching loss.
    """

    def __init__(self, in_channels=6, ndf=64, spectral=True, use_instancenorm=False):
        super().__init__()
        # Each entry is (conv, activation); InstanceNorm is kept optional and
        # off by default since spectral norm already regularises the conv.
        def norm(ch):
            return nn.InstanceNorm2d(ch) if use_instancenorm else nn.Identity()

        self.block1 = nn.Sequential(
            _conv(in_channels, ndf, 2, spectral), nn.LeakyReLU(0.2, True))
        self.block2 = nn.Sequential(
            _conv(ndf, ndf * 2, 2, spectral), norm(ndf * 2), nn.LeakyReLU(0.2, True))
        self.block3 = nn.Sequential(
            _conv(ndf * 2, ndf * 4, 2, spectral), norm(ndf * 4), nn.LeakyReLU(0.2, True))
        self.block4 = nn.Sequential(
            _conv(ndf * 4, ndf * 8, 1, spectral), norm(ndf * 8), nn.LeakyReLU(0.2, True))
        self.head = _conv(ndf * 8, 1, 1, spectral)   # output: patch map

    def forward(self, sketch, photo, return_features=False):
        x = torch.cat([sketch, photo], dim=1)
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        out = self.head(f4)
        if return_features:
            return out, [f1, f2, f3, f4]
        return out


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale PatchGAN (pix2pixHD style).

    Runs ``num_scales`` independent PatchGAN discriminators, each on a
    progressively downsampled version of the input. The coarser scales give the
    generator a larger effective receptive field / global coherence signal,
    while the full-resolution scale preserves fine detail.

    ``forward`` returns a list (one entry per scale) of ``(prediction,
    features)`` tuples, keeping the API uniform for both adversarial and
    feature-matching losses. Loss aggregation (average) is handled by the
    training loop / helper below.
    """

    def __init__(self, in_channels=6, ndf=64, num_scales=2, spectral=True):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            PatchGAN(in_channels=in_channels, ndf=ndf, spectral=spectral)
            for _ in range(num_scales)
        ])

    def downsample(self, x):
        return F.avg_pool2d(x, kernel_size=3, stride=2, padding=1,
                            count_include_pad=False)

    def forward(self, sketch, photo, return_features=True):
        results = []
        s, p = sketch, photo
        for i, disc in enumerate(self.discriminators):
            results.append(disc(s, p, return_features=return_features))
            if i < self.num_scales - 1:
                s = self.downsample(s)
                p = self.downsample(p)
        return results
