"""
Apply xDOG (extended Difference of Gaussians) edge filter to CelebA images
to create synthetic sketch-photo pairs for pretraining.

xDOG paper: https://users.cs.northwestern.edu/~sco590/winnemoeller-cag2012.pdf

These pseudo sketch/photo pairs are used to pretrain the GAN (Phase A) before
fine-tuning on the much smaller set of real forensic CUFS pairs (Phase B).
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def xdog_filter(img_gray: np.ndarray,
                sigma: float = 0.8,
                k: float = 1.6,
                p: float = 19,
                epsilon: float = -0.1,
                phi: float = 10.0) -> np.ndarray:
    """
    Extended Difference of Gaussians — produces pencil-sketch-like edges.
    Tuned defaults mimic the artist sketch style found in CUFS.
    """
    g1 = cv2.GaussianBlur(img_gray.astype(np.float32), (0, 0), sigma)
    g2 = cv2.GaussianBlur(img_gray.astype(np.float32), (0, 0), sigma * k)
    dog = g1 - p * g2
    # Thresholded soft step
    result = np.where(dog >= epsilon,
                      np.ones_like(dog),
                      1.0 + np.tanh(phi * dog))
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def process_celeba(celeba_dir: str, out_dir: str, max_images: int = 10000):
    celeba_path = Path(celeba_dir)
    out_path = Path(out_dir)
    (out_path / "sketch").mkdir(parents=True, exist_ok=True)
    (out_path / "photo").mkdir(parents=True, exist_ok=True)

    images = sorted(celeba_path.glob("*.jpg"))[:max_images]
    if not images:
        raise FileNotFoundError(
            f"No .jpg images found under {celeba_dir}. "
            "Download CelebA first (see README Phase 2c)."
        )
    print(f"Processing {len(images)} CelebA images with xDOG...")

    for img_path in tqdm(images):
        img = np.array(Image.open(img_path).convert("RGB").resize((256, 256)))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sketch = xdog_filter(gray)
        # Invert: white background, dark lines (matches CUFS convention)
        sketch = 255 - sketch

        stem = img_path.stem
        cv2.imwrite(str(out_path / "sketch" / f"{stem}.png"), sketch)
        Image.fromarray(img).save(out_path / "photo" / f"{stem}.jpg")

    print(f"Done. {len(images)} pairs saved to {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--celeba_dir", default="data/raw/celeba/img_align_celeba")
    p.add_argument("--out_dir", default="data/synth")
    p.add_argument("--max_images", type=int, default=10000)
    args = p.parse_args()
    process_celeba(args.celeba_dir, args.out_dir, args.max_images)
