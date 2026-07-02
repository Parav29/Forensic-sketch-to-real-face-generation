"""
Preprocessing pipeline:
- Landmark-based face alignment of both sketch and photo to 256x256
  (dlib / InsightFace / OpenCV, with center-crop fallback — see alignment.py)
- Gamma inversion on sketches (improves SSIM ~0.70 -> 0.80 per literature)
- Side-by-side pair format for pix2pix: [sketch | photo] -> 256x512 image
- Subject-aware, leakage-free train/val/test split
"""
import os
import re
import sys
import random
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.alignment import align_face  # noqa: E402


def gamma_invert(sketch: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """
    Gamma inversion preprocessing from the SketchGAN paper.
    Improves contrast and makes dark lines more prominent.
    """
    inv = 255 - sketch
    inv_norm = inv / 255.0
    adjusted = np.power(inv_norm, gamma)
    return (adjusted * 255).astype(np.uint8)


def make_pair(sketch_path: Path, photo_path: Path, out_path: Path,
              target_size: int = 256, use_gamma_inv: bool = True,
              backend: str = "auto"):
    sketch = np.array(Image.open(sketch_path).convert("RGB"))
    photo = np.array(Image.open(photo_path).convert("RGB"))

    sketch = align_face(sketch, target_size, backend=backend)
    photo = align_face(photo, target_size, backend=backend)

    if use_gamma_inv:
        sketch_gray = cv2.cvtColor(sketch, cv2.COLOR_RGB2GRAY)
        sketch_inv = gamma_invert(sketch_gray)
        sketch = cv2.cvtColor(sketch_inv, cv2.COLOR_GRAY2RGB)

    # Side-by-side: [sketch | photo]
    pair = np.concatenate([sketch, photo], axis=1)
    Image.fromarray(pair).save(out_path)


def subject_id(stem: str) -> str:
    """
    Heuristic subject key: strip common per-image suffixes so multiple images
    of the same person land in the same split.

    e.g. ``m-001-01_sketch`` -> ``m-001``; ``00042_a`` -> ``00042``.
    Falls back to the full stem when no pattern matches (treated as its own id).
    """
    s = stem.lower()
    for suffix in ("_sketch", "_photo", "-sketch", "-photo"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    # Drop a trailing "_NN" / "-NN" / single-letter variant marker.
    s = re.sub(r"[-_](\d{1,2}|[a-z])$", "", s)
    return s


def split_indices(stems, split=(0.8, 0.1, 0.1), subject_aware=True, seed=42):
    """
    Return a dict mapping stem -> split name. When ``subject_aware`` is set, all
    images of a subject go to the same split (prevents identity leakage);
    otherwise items are shuffled and split directly.
    """
    rng = random.Random(seed)
    if subject_aware:
        groups = {}
        for st in stems:
            groups.setdefault(subject_id(st), []).append(st)
        keys = list(groups.keys())
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(n * split[0])
        n_val = int(n * split[1])
        assign = {}
        for i, k in enumerate(keys):
            name = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            for st in groups[k]:
                assign[st] = name
        return assign
    else:
        items = list(stems)
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * split[0])
        n_val = int(n * split[1])
        assign = {}
        for i, st in enumerate(items):
            assign[st] = ("train" if i < n_train else
                          "val" if i < n_train + n_val else "test")
        return assign


def _find_photo(photo_dir: Path, stem: str):
    for ext in (".jpg", ".png", ".jpeg"):
        p = photo_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def build_pairs(sketch_dir, photo_dir, out_dir, split=(0.8, 0.1, 0.1),
                subject_aware=True, backend="auto", use_gamma_inv=True, seed=42):
    sketch_dir, photo_dir, out_dir = Path(sketch_dir), Path(photo_dir), Path(out_dir)
    for split_name in ("train", "val", "test"):
        (out_dir / split_name).mkdir(parents=True, exist_ok=True)

    sketches = sorted(sketch_dir.glob("*.png")) + sorted(sketch_dir.glob("*.jpg"))
    if not sketches:
        raise FileNotFoundError(
            f"No sketches found in {sketch_dir}. Download/sort data first.")

    stems = [s.stem for s in sketches]
    assign = split_indices(stems, split, subject_aware, seed)
    counts = {"train": 0, "val": 0, "test": 0}
    matched = 0

    for sketch_path in tqdm(sketches, desc="Building pairs"):
        stem = sketch_path.stem
        photo_path = _find_photo(photo_dir, stem)
        if photo_path is None:
            continue
        split_name = assign[stem]
        out_path = out_dir / split_name / f"{stem}.png"
        make_pair(sketch_path, photo_path, out_path,
                  use_gamma_inv=use_gamma_inv, backend=backend)
        counts[split_name] += 1
        matched += 1

    print(f"Done. {matched}/{len(sketches)} sketches matched a photo.")
    print(f"Split ({'subject-aware' if subject_aware else 'random'}): "
          f"{counts['train']} train / {counts['val']} val / {counts['test']} test")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sketch_dir", required=True)
    p.add_argument("--photo_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--backend", default="auto",
                   choices=["auto", "dlib", "insightface", "opencv", "center"])
    p.add_argument("--random_split", action="store_true",
                   help="Disable subject-aware split (use plain shuffle).")
    p.add_argument("--no_gamma", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_pairs(args.sketch_dir, args.photo_dir, args.out_dir,
                subject_aware=not args.random_split, backend=args.backend,
                use_gamma_inv=not args.no_gamma, seed=args.seed)
