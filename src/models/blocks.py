"""
Reusable network building blocks shared by the generator and discriminator.

Kept intentionally small and dependency-free so the modules can be unit-tested
in isolation and composed into the pix2pix-style architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    SAGAN-style self-attention block.

    Computes attention over spatial positions so the generator can relate
    distant facial regions (e.g. both eyes) at the bottleneck. The output is a
    residual: ``out = x + gamma * attention(x)`` with ``gamma`` initialised to
    zero, so a freshly added block starts as an identity mapping — this is what
    lets us bolt attention onto an existing checkpoint without destroying it.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        reduced = max(in_channels // 8, 1)
        self.query = nn.Conv2d(in_channels, reduced, 1)
        self.key = nn.Conv2d(in_channels, reduced, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        q = self.query(x).view(b, -1, n).permute(0, 2, 1)   # (B, N, C')
        k = self.key(x).view(b, -1, n)                        # (B, C', N)
        attn = self.softmax(torch.bmm(q, k))                  # (B, N, N)
        v = self.value(x).view(b, -1, n)                      # (B, C, N)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return x + self.gamma * out


class ResidualBlock(nn.Module):
    """
    Lightweight residual refinement block used in the decoder.

    Two 3x3 convolutions with InstanceNorm; the final conv weight is zero-init
    so the block also starts as an identity mapping (safe to add to an existing
    generator checkpoint).
    """

    def __init__(self, channels: int, norm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels) if norm else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels) if norm else nn.Identity(),
        ]
        self.block = nn.Sequential(*layers)
        # Zero-init the last conv so the residual starts at identity.
        nn.init.zeros_(self.block[3].weight)
        if self.block[3].bias is not None:
            nn.init.zeros_(self.block[3].bias)

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=True)


class SkipGate(nn.Module):
    """
    Squeeze-and-excitation gate applied to an encoder skip feature before it is
    concatenated into the decoder.

    Improves skip-connection aggregation by letting the network re-weight skip
    channels instead of naively concatenating them. Initialised so it starts
    close to a pass-through (gate ≈ 1).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        # Bias the final layer positive so sigmoid(gate) starts near 1.
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 3.0)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))              # global average pool -> (B, C)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s
