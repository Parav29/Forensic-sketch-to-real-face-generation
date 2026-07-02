import torch
from models.generator import UNetGenerator
from models.discriminator import PatchGAN, MultiScaleDiscriminator
from models.losses import GANLoss, FeatureMatchingLoss


def test_generator_output_shape():
    G = UNetGenerator(ngf=8)
    x = torch.randn(2, 3, 256, 256)
    assert G(x).shape == (2, 3, 256, 256)


def test_generator_base_is_backward_compatible():
    """Flags off -> identical module set, so a base checkpoint loads strictly."""
    base = UNetGenerator(ngf=8, use_attention=False, use_residual=False,
                         use_skip_fusion=False)
    enhanced = UNetGenerator(ngf=8)
    # Base state loads into enhanced with only enhancement keys missing.
    missing, unexpected = enhanced.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == []
    assert all(k.startswith(("attn", "refine", "skip_gate")) for k in missing)


def test_enhancements_are_identity_initialised():
    G = UNetGenerator(ngf=8)
    G.init_enhancements()
    assert torch.allclose(G.attn.gamma, torch.zeros_like(G.attn.gamma))
    # Skip gate biased so sigmoid(gate) ~ 1 (pass-through).
    assert torch.all(G.skip_gate1.fc2.bias > 0)


def test_patchgan_features():
    D = PatchGAN(ndf=8)
    x = torch.randn(2, 3, 256, 256)
    out, feats = D(x, x, return_features=True)
    assert out.dim() == 4 and len(feats) == 4


def test_multiscale_discriminator():
    D = MultiScaleDiscriminator(ndf=8, num_scales=2)
    x = torch.randn(2, 3, 256, 256)
    res = D(x, x)
    assert len(res) == 2
    for pred, feats in res:
        assert pred.shape[1] == 1 and len(feats) == 4


def test_gan_loss_handles_multiscale():
    D = MultiScaleDiscriminator(ndf=8, num_scales=2)
    x = torch.randn(1, 3, 256, 256)
    loss = GANLoss()(D(x, x), True)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_feature_matching_loss():
    D = MultiScaleDiscriminator(ndf=8, num_scales=2)
    x = torch.randn(1, 3, 256, 256)
    fm = FeatureMatchingLoss()(D(x, x), D(x, x))
    assert torch.isfinite(fm) and fm.item() >= 0
