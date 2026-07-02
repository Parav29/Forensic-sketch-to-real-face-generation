"""
Checkpoint -> generator loading shared by eval.py and demo.py.

Reads the architecture flags embedded in the checkpoint's ``cfg`` (falling back
to the enhanced defaults), prefers the EMA weights for inference, and loads with
``strict=False`` so both old (base UNet) and new checkpoints work.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.generator import UNetGenerator


def load_generator(ckpt_path, device, prefer_ema=True):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = (ck.get("cfg") or {}).get("model", {}) if isinstance(ck, dict) else {}

    G = UNetGenerator(
        ngf=mc.get("ngf", 64),
        use_attention=mc.get("use_attention", True),
        use_residual=mc.get("use_residual", True),
        use_skip_fusion=mc.get("use_skip_fusion", True),
    )

    # Prefer EMA weights for inference; fall back to raw generator weights.
    if isinstance(ck, dict):
        state = ck.get("ema") if (prefer_ema and "ema" in ck) else ck.get("G", ck)
    else:
        state = ck
    missing, unexpected = G.load_state_dict(state, strict=False)
    G.eval().to(device)
    return G, {"missing": list(missing), "unexpected": list(unexpected),
               "used_ema": prefer_ema and isinstance(ck, dict) and "ema" in ck}
