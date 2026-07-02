import copy
import pytest
import torch
import torch.nn as nn

from utils.config import DEFAULTS, validate_config, _deep_merge, _legacy_migrate
from utils.ema import ModelEMA
from utils.checkpoint import CheckpointManager
from utils.seed import set_seed


def test_defaults_are_valid():
    assert validate_config(copy.deepcopy(DEFAULTS))


def test_validate_rejects_bad_values():
    bad = _deep_merge(DEFAULTS, {"train": {"lr_g": -1}})
    with pytest.raises(ValueError):
        validate_config(bad)
    bad2 = _deep_merge(DEFAULTS, {"loss": {"identity_backbone": "bogus"}})
    with pytest.raises(ValueError):
        validate_config(bad2)


def test_legacy_lr_migration():
    migrated = _legacy_migrate({"train": {"lr": 0.0003}})
    assert migrated["train"]["lr_g"] == 0.0003
    assert migrated["train"]["lr_d"] == 0.0003
    assert "lr" not in migrated["train"]


def test_ema_tracks_weights():
    model = nn.Linear(4, 4)
    ema = ModelEMA(model, decay=0.5)
    before = ema.ema_model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    after = ema.ema_model.weight.detach()
    # Moved halfway toward the new weights (decay=0.5).
    assert torch.allclose(after, before + 0.5, atol=1e-5)


def test_checkpoint_best_and_prune(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last=2, best_mode="min")
    for e in range(1, 4):
        mgr.save({"epoch": e}, e)
    remaining = sorted(p.name for p in tmp_path.glob("epoch_*.pt"))
    assert remaining == ["epoch_0002.pt", "epoch_0003.pt"]  # pruned to last 2

    assert mgr.maybe_save_best({"epoch": 1}, metric=10.0) is True
    assert mgr.maybe_save_best({"epoch": 2}, metric=20.0) is False  # worse
    assert mgr.maybe_save_best({"epoch": 3}, metric=5.0) is True    # better
    assert (tmp_path / "best.pt").exists()


def test_set_seed_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.allclose(a, b)
