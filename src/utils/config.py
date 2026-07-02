"""
Configuration loading, default-merging and validation.

A single ``DEFAULTS`` tree documents every knob and its default. User configs
are deep-merged onto it, so older / partial configs keep working (backward
compatibility) and new keys always have a sane fallback. ``validate_config``
then type-checks the merged result and fails fast with a clear message.
"""
import copy
import yaml


DEFAULTS = {
    "seed": 42,
    "deterministic": False,
    "device": "cuda",
    "amp": True,                       # mixed precision (auto-off on CPU)
    "wandb": False,

    "data": {
        "pairs_dir": "data/cufs_pairs",
        "image_size": 256,
        "batch_size": 4,
        "num_workers": 4,
    },

    "model": {
        "ngf": 64,
        "ndf": 64,
        "use_attention": True,
        "use_residual": True,
        "use_skip_fusion": True,
        "num_scales": 2,               # multi-scale discriminator
        "spectral_norm": True,
    },

    "loss": {
        "lambda_l1": 100.0,
        "lambda_perceptual": 10.0,
        "lambda_identity": 5.0,
        "lambda_lpips": 1.0,
        "lambda_fm": 10.0,
        "identity_backbone": "facenet",  # or "arcface"
        "lpips_net": "alex",
    },

    "train": {
        "epochs": 200,
        "lr_g": 2.0e-4,                # TTUR: generator LR
        "lr_d": 8.0e-4,                # TTUR: discriminator LR
        "beta1": 0.5,
        "beta2": 0.999,
        "decay_start": 100,
        "grad_clip": 0.0,             # 0 disables gradient clipping
        "ema_decay": 0.999,
        "save_every": 10,
        "sample_every": 5,
        "keep_last": 3,               # rolling checkpoint retention
        "eval_fid_every": 0,          # 0 disables periodic validation FID
    },

    "paths": {
        "checkpoints": "outputs/checkpoints",
        "samples": "outputs/samples",
        "metrics": "outputs/metrics",
        "logs": "outputs/logs",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _legacy_migrate(user: dict) -> dict:
    """Map a few old keys onto the new schema so pre-refactor configs load."""
    user = copy.deepcopy(user)
    train = user.get("train", {})
    # Old single `lr` -> TTUR generator/discriminator LRs.
    if "lr" in train:
        train.setdefault("lr_g", train["lr"])
        train.setdefault("lr_d", train["lr"])
        train.pop("lr")
    return user


def load_config(path: str) -> dict:
    """Load a YAML config, migrate legacy keys, merge defaults and validate."""
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULTS, _legacy_migrate(user))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict):
    """Type/range-check the merged config; raise ValueError on any problem."""
    errors = []

    def check(cond, msg):
        if not cond:
            errors.append(msg)

    check(isinstance(cfg["seed"], int), "seed must be an int")
    check(cfg["device"] in ("cuda", "cpu"), "device must be 'cuda' or 'cpu'")

    d = cfg["data"]
    check(d["image_size"] > 0 and d["image_size"] % 256 == 0 or d["image_size"] == 256,
          "data.image_size should be 256 for the UNet-256 architecture")
    check(d["batch_size"] >= 1, "data.batch_size must be >= 1")
    check(d["num_workers"] >= 0, "data.num_workers must be >= 0")

    m = cfg["model"]
    check(m["ngf"] >= 1 and m["ndf"] >= 1, "model.ngf/ndf must be >= 1")
    check(1 <= m["num_scales"] <= 4, "model.num_scales must be in [1, 4]")

    lo = cfg["loss"]
    for k in ("lambda_l1", "lambda_perceptual", "lambda_identity",
              "lambda_lpips", "lambda_fm"):
        check(lo[k] >= 0, f"loss.{k} must be >= 0")
    check(lo["identity_backbone"] in ("facenet", "arcface"),
          "loss.identity_backbone must be 'facenet' or 'arcface'")

    t = cfg["train"]
    check(t["epochs"] >= 1, "train.epochs must be >= 1")
    check(t["lr_g"] > 0 and t["lr_d"] > 0, "train.lr_g/lr_d must be > 0")
    check(0 <= t["decay_start"] <= t["epochs"],
          "train.decay_start must be within [0, epochs]")
    check(0.0 <= t["ema_decay"] < 1.0, "train.ema_decay must be in [0, 1)")
    check(t["grad_clip"] >= 0, "train.grad_clip must be >= 0")

    if errors:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))
    return True
