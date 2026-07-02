import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class SketchPhotoDataset(Dataset):
    """
    Loads side-by-side [sketch | photo] 256x512 pairs.
    Left half = sketch (input), right half = photo (target).

    Augmentation notes:
      * Spatial transforms (flip / rotate / affine) are applied identically to
        the sketch and photo via ``ReplayCompose`` so the pair stays aligned.
      * Photometric / degradation transforms (blur, coarse dropout, pencil-line
        jitter) are applied to the *sketch only*, since they model imperfect
        forensic input and must not corrupt the target photo.
    """

    def __init__(self, pairs_dir: str, split: str = "train",
                 image_size: int = 256, augment: bool = True):
        self.pairs_dir = Path(pairs_dir) / split
        self.files = sorted(self.pairs_dir.glob("*.png"))
        self.image_size = image_size
        self.augment = augment and split == "train"

        # Shared spatial augmentation (replayed on both images to stay aligned).
        self.spatial = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-15, 15), scale=(0.92, 1.08),
                     translate_percent=(0.0, 0.04), p=0.5, fill=255),
        ]) if self.augment else None

        # Sketch-only degradations: model imperfect / robust pencil strokes.
        self.sketch_only = A.Compose([
            A.GaussianBlur(blur_limit=(3, 5), p=0.25),
            A.CoarseDropout(num_holes_range=(1, 8),
                            hole_height_range=(8, 16),
                            hole_width_range=(8, 16),
                            fill=255, p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.1,
                                       contrast_limit=0.15, p=0.3),
            A.ImageCompression(quality_range=(60, 100), p=0.2),
        ]) if self.augment else None

        self.to_tensor = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pair = np.array(Image.open(self.files[idx]).convert("RGB"))
        w = pair.shape[1] // 2
        sketch = pair[:, :w, :]
        photo = pair[:, w:, :]

        if self.spatial is not None:
            out = self.spatial(image=sketch)
            sketch = out["image"]
            photo = A.ReplayCompose.replay(out["replay"], image=photo)["image"]

        if self.sketch_only is not None:
            sketch = self.sketch_only(image=sketch)["image"]

        sketch = self.to_tensor(image=sketch)["image"]
        photo = self.to_tensor(image=photo)["image"]

        return {"sketch": sketch, "photo": photo,
                "filename": self.files[idx].stem}
