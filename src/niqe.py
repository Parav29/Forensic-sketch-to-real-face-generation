"""
Self-contained NIQE (Natural Image Quality Evaluator).

NIQE scores an image by the Mahalanobis distance between its natural-scene-
statistics (NSS) features and a multivariate-Gaussian (MVG) model fitted on
*pristine* images. The canonical implementation ships a pristine model fitted on
a natural-image corpus; to stay fully offline we fit the pristine MVG on the
evaluation set's own ground-truth photos (genuine natural faces). This is a
reference-informed NIQE variant — lower is better, and it is internally
consistent for comparing models on the same test set.

Features per patch: 18 GGD/AGGD parameters, extracted at 2 scales -> 36-dim.
"""
import numpy as np
from scipy import ndimage
from scipy.special import gamma as gamma_fn


def _gaussian_kernel(size=7, sigma=7 / 6):
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


_KERNEL = _gaussian_kernel()


def _mscn(img):
    """Mean-subtracted contrast-normalised coefficients."""
    img = img.astype(np.float64)
    mu = ndimage.convolve(img, _KERNEL, mode="nearest")
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(ndimage.convolve(img * img, _KERNEL, mode="nearest") - mu_sq))
    return (img - mu) / (sigma + 1.0)


_GAMMA_RANGE = np.arange(0.2, 10.0, 0.001)
_R_GAMMA = (gamma_fn(1.0 / _GAMMA_RANGE) * gamma_fn(3.0 / _GAMMA_RANGE)
            / (gamma_fn(2.0 / _GAMMA_RANGE) ** 2))


def _ggd_fit(x):
    sigma_sq = np.mean(x ** 2)
    if sigma_sq == 0:
        return 0.0, 0.0
    E = np.mean(np.abs(x))
    rho = sigma_sq / (E ** 2 + 1e-12)
    idx = np.argmin(np.abs(rho - _R_GAMMA))
    alpha = _GAMMA_RANGE[idx]
    return alpha, np.sqrt(sigma_sq)


def _aggd_fit(x):
    x = x.flatten()
    left = x[x < 0]
    right = x[x >= 0]
    sig_l = np.sqrt(np.mean(left ** 2)) if left.size else 0.0
    sig_r = np.sqrt(np.mean(right ** 2)) if right.size else 0.0
    if sig_r == 0:
        return 0.0, sig_l, sig_r
    gamma_hat = sig_l / (sig_r + 1e-12)
    rhat = (np.mean(np.abs(x)) ** 2) / (np.mean(x ** 2) + 1e-12)
    rhat_norm = (rhat * (gamma_hat ** 3 + 1)) * (gamma_hat + 1) / ((gamma_hat ** 2 + 1) ** 2)
    idx = np.argmin(np.abs(rhat_norm - _R_GAMMA))
    alpha = _GAMMA_RANGE[idx]
    return alpha, sig_l, sig_r


def _feature_vector(mscn):
    feats = []
    alpha, sigma = _ggd_fit(mscn)
    feats += [alpha, sigma ** 2]
    # Paired products along 4 orientations -> AGGD params.
    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dy, dx in shifts:
        shifted = np.roll(np.roll(mscn, dy, axis=0), dx, axis=1)
        pair = mscn * shifted
        alpha, sl, sr = _aggd_fit(pair)
        mean = (sr - sl) * (gamma_fn(2.0 / (alpha + 1e-12)) /
                            gamma_fn(1.0 / (alpha + 1e-12))) if alpha > 0 else 0.0
        feats += [alpha, mean, sl ** 2, sr ** 2]
    return np.array(feats, dtype=np.float64)  # 2 + 4*4 = 18


def extract_features(img_gray):
    """36-dim NIQE features across 2 scales for a grayscale [0,255] image."""
    f1 = _feature_vector(_mscn(img_gray))
    half = ndimage.zoom(img_gray, 0.5, order=1)
    f2 = _feature_vector(_mscn(half))
    return np.concatenate([f1, f2])


def fit_mvg(feature_rows):
    feats = np.asarray(feature_rows)
    mu = np.mean(feats, axis=0)
    cov = np.cov(feats, rowvar=False)
    return mu, cov


def niqe_distance(feat, mu, cov):
    diff = feat - mu
    inv = np.linalg.pinv(cov)
    return float(np.sqrt(np.maximum(diff @ inv @ diff.T, 0.0)))
