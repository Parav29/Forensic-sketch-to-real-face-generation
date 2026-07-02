import numpy as np
from PIL import Image

import niqe
from metrics import ssim_psnr, compute_niqe


def test_niqe_features_dim():
    img = (np.random.rand(128, 128) * 255).astype(np.float64)
    feats = niqe.extract_features(img)
    assert feats.shape == (36,)


def test_ssim_psnr_identical_is_perfect():
    img = np.random.rand(64, 64, 3)
    s, p = ssim_psnr(img, img)
    assert s > 0.999
    assert p > 60  # near-infinite PSNR for identical images


def test_compute_niqe_runs(tmp_path):
    real = tmp_path / "real"; gen = tmp_path / "gen"
    real.mkdir(); gen.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        a = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        Image.fromarray(a).save(real / f"i{i}.png")
        Image.fromarray(a).save(gen / f"i{i}.png")
    score = compute_niqe(str(gen), str(real))
    assert score >= 0
