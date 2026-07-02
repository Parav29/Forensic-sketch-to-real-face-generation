"""
Generate synthetic sketch-photo pairs from CelebA for GAN pretraining (Phase A).

Multiple sketch styles are supported through a pluggable registry
(``sketch_styles.py``): xDOG, Canny, dodge-blend pencil, adaptive-threshold.
Pass one or more ``--styles`` to mix styles across the generated set, which
improves robustness to the varied stroke styles of real forensic sketches.

These pseudo pairs pretrain the GAN before fine-tuning on the much smaller set
of real forensic CUFS pairs (Phase B).
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sketch_styles import apply_style, available_styles  # noqa: E402


def process_celeba(celeba_dir, out_dir, max_images=10000,
                   styles=("xdog",), size=256, seed=42):
    celeba_path = Path(celeba_dir)
    out_path = Path(out_dir)
    (out_path / "sketch").mkdir(parents=True, exist_ok=True)
    (out_path / "photo").mkdir(parents=True, exist_ok=True)

    images = sorted(celeba_path.glob("*.jpg"))[:max_images]
    if not images:
        raise FileNotFoundError(
            f"No .jpg images found under {celeba_dir}. Download CelebA first.")

    rng = np.random.default_rng(seed)
    print(f"Processing {len(images)} CelebA images with styles={list(styles)}...")

    for img_path in tqdm(images):
        img = np.array(Image.open(img_path).convert("RGB").resize((size, size)))
        # Round-robin / random style selection when several are provided.
        style = styles[rng.integers(len(styles))] if len(styles) > 1 else styles[0]
        sketch = apply_style(style, img)

        stem = img_path.stem
        Image.fromarray(sketch).save(out_path / "sketch" / f"{stem}.png")
        Image.fromarray(img).save(out_path / "photo" / f"{stem}.jpg")

    print(f"Done. {len(images)} pairs saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--celeba_dir", default="data/raw/celeba/img_align_celeba")
    p.add_argument("--out_dir", default="data/synth")
    p.add_argument("--max_images", type=int, default=10000)
    p.add_argument("--styles", nargs="+", default=["xdog"],
                   help=f"One or more of: {available_styles()}")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    process_celeba(args.celeba_dir, args.out_dir, args.max_images,
                   styles=tuple(args.styles), seed=args.seed)
