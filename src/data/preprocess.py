"""
Preprocessing pipeline:
- Face-crop and align both sketch and photo to 256x256
- Gamma inversion on sketches (improves SSIM ~0.70 -> 0.80 per literature)
- Side-by-side pair format for pix2pix: [sketch | photo] -> 256x512 image
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def align_face(img: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Resize and center-crop to target_size x target_size."""
    h, w = img.shape[:2]
    scale = target_size / min(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    h, w = img.shape[:2]
    top = (h - target_size) // 2
    left = (w - target_size) // 2
    return img[top:top + target_size, left:left + target_size]


def gamma_invert(sketch: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """
    Gamma inversion preprocessing from the SketchGAN paper.
    Improves contrast and makes dark lines more prominent.
    """
    inv = 255 - sketch
    inv_norm = inv / 255.0
    adjusted = np.power(inv_norm, gamma)
    return (adjusted * 255).astype(np.uint8)


def make_pair(sketch_path: Path, photo_path: Path,
              out_path: Path, target_size: int = 256,
              use_gamma_inv: bool = True):
    sketch = np.array(Image.open(sketch_path).convert("RGB"))
    photo = np.array(Image.open(photo_path).convert("RGB"))

    sketch = align_face(sketch, target_size)
    photo = align_face(photo, target_size)

    if use_gamma_inv:
        sketch_gray = cv2.cvtColor(sketch, cv2.COLOR_RGB2GRAY)
        sketch_inv = gamma_invert(sketch_gray)
        sketch = cv2.cvtColor(sketch_inv, cv2.COLOR_GRAY2RGB)

    # Side-by-side: [sketch | photo]
    pair = np.concatenate([sketch, photo], axis=1)
    Image.fromarray(pair).save(out_path)


def build_pairs(sketch_dir: str, photo_dir: str, out_dir: str,
                split: tuple = (0.8, 0.1, 0.1)):
    sketch_dir = Path(sketch_dir)
    photo_dir = Path(photo_dir)
    out_dir = Path(out_dir)

    for split_name in ["train", "val", "test"]:
        (out_dir / split_name).mkdir(parents=True, exist_ok=True)

    sketches = sorted(sketch_dir.glob("*.png")) + sorted(sketch_dir.glob("*.jpg"))
    n = len(sketches)
    if n == 0:
        raise FileNotFoundError(
            f"No sketches found in {sketch_dir}. Download/sort data first."
        )
    n_train = int(n * split[0])
    n_val = int(n * split[1])

    splits = (["train"] * n_train +
              ["val"] * n_val +
              ["test"] * (n - n_train - n_val))

    print(f"Building {n} pairs: {n_train} train / {n_val} val / "
          f"{n - n_train - n_val} test")
    matched = 0
    for sketch_path, split_name in tqdm(zip(sketches, splits), total=n):
        stem = sketch_path.stem
        photo_path = photo_dir / f"{stem}.jpg"
        if not photo_path.exists():
            photo_path = photo_dir / f"{stem}.png"
        if not photo_path.exists():
            continue
        out_path = out_dir / split_name / f"{stem}.png"
        make_pair(sketch_path, photo_path, out_path)
        matched += 1

    print(f"Done. {matched}/{n} sketches had a matching photo.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sketch_dir", required=True)
    p.add_argument("--photo_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    build_pairs(args.sketch_dir, args.photo_dir, args.out_dir)
