import torch
import torch.nn as nn

from .blocks import SelfAttention, ResidualBlock, SkipGate


class UNetGenerator(nn.Module):
    """
    UNet-256: 8 downsampling stages, skip connections from encoder to decoder.
    Input: 3-channel sketch  ->  Output: 3-channel photo

    Research upgrades (all optional and additive so that a checkpoint trained
    with them disabled still loads with ``strict=True``, and the enhanced
    modules are identity-initialised so enabling them on an old checkpoint does
    not destroy it):

      * ``use_attention``   – SAGAN self-attention on the 16x16 feature map.
      * ``use_residual``    – residual refinement blocks in the decoder.
      * ``use_skip_fusion`` – squeeze-excitation gating of skip connections.

    Disable all three to recover the original pix2pix UNet exactly.
    """

    def __init__(self, in_channels=3, out_channels=3, ngf=64,
                 use_attention=True, use_residual=True, use_skip_fusion=True):
        super().__init__()
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_skip_fusion = use_skip_fusion

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

        # --- Optional research upgrades (identity-initialised) ---
        if use_attention:
            # e4 is the 16x16 feature map (ngf*8 channels).
            self.attn = SelfAttention(ngf * 8)

        if use_skip_fusion:
            # Gate each encoder skip feature before concatenation.
            self.skip_gate1 = SkipGate(ngf)
            self.skip_gate2 = SkipGate(ngf * 2)
            self.skip_gate3 = SkipGate(ngf * 4)
            self.skip_gate4 = SkipGate(ngf * 8)
            self.skip_gate5 = SkipGate(ngf * 8)
            self.skip_gate6 = SkipGate(ngf * 8)
            self.skip_gate7 = SkipGate(ngf * 8)

        if use_residual:
            # Refinement in the higher-resolution decoder stages.
            self.refine4 = ResidualBlock(ngf * 4)
            self.refine3 = ResidualBlock(ngf * 2)
            self.refine2 = ResidualBlock(ngf)

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

    def init_enhancements(self):
        """
        Re-apply the identity initialisation of the optional upgrade modules.

        Call this AFTER a blanket weight init (e.g. normal(0, 0.02)) so the
        added attention/residual/skip-gate modules start as identity mappings
        and do not disturb early training or a loaded base checkpoint.
        """
        if self.use_attention:
            nn.init.zeros_(self.attn.gamma)
        if self.use_residual:
            for blk in (self.refine4, self.refine3, self.refine2):
                nn.init.zeros_(blk.block[3].weight)
                if blk.block[3].bias is not None:
                    nn.init.zeros_(blk.block[3].bias)
        if self.use_skip_fusion:
            for i in range(1, 8):
                gate = getattr(self, f"skip_gate{i}")
                nn.init.zeros_(gate.fc2.weight)
                nn.init.constant_(gate.fc2.bias, 3.0)

    def _gate(self, name, feat):
        """Apply the named skip gate if skip fusion is enabled."""
        if self.use_skip_fusion:
            return getattr(self, name)(feat)
        return feat

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        if self.use_attention:
            e4 = self.attn(e4)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        d8 = self.dec8(e8)
        d7 = self.dec7(torch.cat([d8, self._gate("skip_gate7", e7)], 1))
        d6 = self.dec6(torch.cat([d7, self._gate("skip_gate6", e6)], 1))
        d5 = self.dec5(torch.cat([d6, self._gate("skip_gate5", e5)], 1))
        d4 = self.dec4(torch.cat([d5, self._gate("skip_gate4", e4)], 1))
        if self.use_residual:
            d4 = self.refine4(d4)
        d3 = self.dec3(torch.cat([d4, self._gate("skip_gate3", e3)], 1))
        if self.use_residual:
            d3 = self.refine3(d3)
        d2 = self.dec2(torch.cat([d3, self._gate("skip_gate2", e2)], 1))
        if self.use_residual:
            d2 = self.refine2(d2)
        return self.dec1(torch.cat([d2, self._gate("skip_gate1", e1)], 1))
