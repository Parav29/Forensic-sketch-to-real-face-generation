"""
Pluggable synthetic-sketch generators.

Each style takes a grayscale uint8 image and returns a grayscale uint8 sketch
(white background, dark lines — matching the CUFS convention). New styles can be
registered with the ``@register_style`` decorator and become immediately usable
from ``synth_data.py`` via ``--styles``.
"""
import cv2
import numpy as np


STYLE_REGISTRY = {}


def register_style(name):
    def deco(fn):
        STYLE_REGISTRY[name] = fn
        return fn
    return deco


def get_style(name):
    if name not in STYLE_REGISTRY:
        raise KeyError(f"Unknown sketch style '{name}'. "
                       f"Available: {sorted(STYLE_REGISTRY)}")
    return STYLE_REGISTRY[name]


def available_styles():
    return sorted(STYLE_REGISTRY)


@register_style("xdog")
def xdog(img_gray, sigma=0.8, k=1.6, p=19, epsilon=-0.1, phi=10.0):
    """Extended Difference of Gaussians — pencil-sketch-like edges (CUFS-style)."""
    g = img_gray.astype(np.float32)
    g1 = cv2.GaussianBlur(g, (0, 0), sigma)
    g2 = cv2.GaussianBlur(g, (0, 0), sigma * k)
    dog = g1 - p * g2
    result = np.where(dog >= epsilon, 1.0, 1.0 + np.tanh(phi * dog))
    sketch = np.clip(result * 255, 0, 255).astype(np.uint8)
    return 255 - sketch  # dark lines on white


@register_style("canny")
def canny(img_gray, low=50, high=150):
    """Clean Canny edge map, dilated slightly to look hand-drawn."""
    edges = cv2.Canny(img_gray, low, high)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return 255 - edges


@register_style("pencil")
def pencil(img_gray, blur_ksize=21):
    """Classic dodge-blend pencil sketch (grayscale)."""
    inv = 255 - img_gray
    blur = cv2.GaussianBlur(inv, (blur_ksize, blur_ksize), 0)
    inv_blur = 255 - blur
    sketch = cv2.divide(img_gray, inv_blur, scale=256.0)
    return np.clip(sketch, 0, 255).astype(np.uint8)


@register_style("adaptive")
def adaptive_threshold(img_gray, block=9, c=5):
    """Adaptive-threshold line drawing (crisp, high-contrast strokes)."""
    blurred = cv2.medianBlur(img_gray, 5)
    edges = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, block, c)
    return edges


def apply_style(name, img_rgb):
    """Convenience: RGB uint8 -> grayscale sketch via the named style."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return get_style(name)(gray)
