"""
Checkpoint management: rolling retention of the last-N epoch checkpoints, a
separate "best" checkpoint tracked by a validation metric (FID), and a small
resume helper.
"""
from pathlib import Path
import torch


class CheckpointManager:
    def __init__(self, ckpt_dir, keep_last: int = 3, best_mode: str = "min"):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.best_mode = best_mode
        self.best_metric = float("inf") if best_mode == "min" else float("-inf")

    def _is_better(self, metric: float) -> bool:
        if self.best_mode == "min":
            return metric < self.best_metric
        return metric > self.best_metric

    def save(self, state: dict, epoch: int, tag: str = "epoch"):
        """Save a rolling epoch checkpoint and prune older ones."""
        path = self.dir / f"{tag}_{epoch:04d}.pt"
        torch.save(state, path)
        self._prune(tag)
        return path

    def save_latest(self, state: dict):
        """Overwrite a single ``latest.pt`` used for cheap resume."""
        path = self.dir / "latest.pt"
        torch.save(state, path)
        return path

    def maybe_save_best(self, state: dict, metric: float):
        """Save ``best.pt`` if ``metric`` improves the tracked best value."""
        if self._is_better(metric):
            self.best_metric = metric
            state = {**state, "best_metric": metric}
            torch.save(state, self.dir / "best.pt")
            return True
        return False

    def _prune(self, tag: str):
        ckpts = sorted(self.dir.glob(f"{tag}_*.pt"))
        for old in ckpts[:-self.keep_last] if self.keep_last > 0 else []:
            old.unlink(missing_ok=True)

    @staticmethod
    def load(path, map_location="cpu"):
        return torch.load(path, map_location=map_location)
