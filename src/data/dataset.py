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
    """

    def __init__(self, pairs_dir: str, split: str = "train",
                 image_size: int = 256, augment: bool = True):
        self.pairs_dir = Path(pairs_dir) / split
        self.files = sorted(self.pairs_dir.glob("*.png"))
        self.image_size = image_size
        self.augment = augment and split == "train"

        # ReplayCompose so the SAME spatial augmentation can be replayed on
        # both the sketch and the photo (they must stay aligned).
        self.transform = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, p=0.3),
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

        # Apply the SAME spatial augmentation to both via replay.
        if self.transform:
            out = self.transform(image=sketch)
            sketch = out["image"]
            photo = A.ReplayCompose.replay(out["replay"], image=photo)["image"]

        sketch = self.to_tensor(image=sketch)["image"]
        photo = self.to_tensor(image=photo)["image"]

        return {"sketch": sketch, "photo": photo,
                "filename": self.files[idx].stem}
