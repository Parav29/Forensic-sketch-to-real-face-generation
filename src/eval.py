"""
Evaluation: generate test outputs and report a full metric suite as JSON.

Metrics:
  FID, LPIPS, SSIM, PSNR, Identity cosine similarity, Rank-1/5/10, NIQE.

Also writes ``Sketch | Generated | Ground Truth`` comparison grids and a
``report.json`` summarising everything.
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import SketchPhotoDataset
from model_io import load_generator
from utils import get_logger
import metrics as M


def _to_pil(t):
    img = (t.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) / 2
    return Image.fromarray(np.clip(img * 255, 0, 255).astype("uint8"))


def evaluate(model_path, pairs_dir, output_dir, device="cuda",
             fid_dims=2048, make_grids=True, lpips_net="alex"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger = get_logger("eval")

    G, info = load_generator(model_path, device, prefer_ema=True)
    logger.info(f"Loaded {model_path} (used_ema={info['used_ema']})")

    test_ds = SketchPhotoDataset(pairs_dir, "test", augment=False)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
    if len(test_ds) == 0:
        raise RuntimeError(f"No test pairs found under {pairs_dir}/test")

    out = Path(output_dir)
    gen_dir, real_dir, grid_dir = out / "generated", out / "real", out / "grids"
    for d in (gen_dir, real_dir, grid_dir):
        d.mkdir(parents=True, exist_ok=True)

    lpips_metric = None
    try:
        lpips_metric = M.LPIPSMetric(net=lpips_net, device=device)
    except Exception as e:
        logger.warning(f"LPIPS unavailable: {e}")

    ssim_scores, psnr_scores, lpips_scores = [], [], []

    logger.info("Generating test outputs...")
    with torch.no_grad():
        for batch in tqdm(test_dl):
            sketch = batch["sketch"].to(device)
            real = batch["photo"].to(device)
            fname = batch["filename"][0]
            fake = G(sketch)

            _to_pil(fake).save(gen_dir / f"{fname}.png")
            _to_pil(real).save(real_dir / f"{fname}.png")

            if make_grids:
                grid = vutils.make_grid(
                    torch.cat([sketch, fake, real], 0), nrow=3,
                    normalize=True, value_range=(-1, 1))
                vutils.save_image(grid, grid_dir / f"{fname}.png")

            fake_np = (fake.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) / 2
            real_np = (real.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) / 2
            s, p = M.ssim_psnr(fake_np.astype(np.float64), real_np.astype(np.float64))
            ssim_scores.append(s); psnr_scores.append(p)
            if lpips_metric is not None:
                lpips_scores.append(lpips_metric(fake, real))

    results = {
        "n_samples": len(ssim_scores),
        "SSIM": round(float(np.mean(ssim_scores)), 4),
        "PSNR": round(float(np.mean(psnr_scores)), 4),
    }
    if lpips_scores:
        results["LPIPS"] = round(float(np.mean(lpips_scores)), 4)

    # FID (direct pytorch-fid API)
    try:
        results["FID"] = round(M.compute_fid(real_dir, gen_dir, device, dims=fid_dims), 4)
    except Exception as e:
        logger.warning(f"FID failed: {e}")
        results["FID"] = None

    # NIQE (no-reference; pristine model from real photos)
    try:
        results["NIQE"] = M.compute_niqe(gen_dir, real_dir)
    except Exception as e:
        logger.warning(f"NIQE failed: {e}")
        results["NIQE"] = None

    # Identity: cosine + Rank-1/5/10
    try:
        id_metric = M.IdentityMetric(device=device)
        results.update(id_metric.evaluate(gen_dir, real_dir, ranks=(1, 5, 10)))
    except Exception as e:
        logger.warning(f"Identity metrics failed: {e}")

    logger.info("=== EVALUATION RESULTS ===")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")

    with open(out / "report.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Report written to {out / 'report.json'}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="outputs/checkpoints/best.pt")
    p.add_argument("--pairs_dir", default="data/cufs_pairs")
    p.add_argument("--out_dir", default="outputs/eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fid_dims", type=int, default=2048,
                   choices=[64, 192, 768, 2048])
    p.add_argument("--no_grids", action="store_true")
    args = p.parse_args()
    evaluate(args.model, args.pairs_dir, args.out_dir, args.device,
             fid_dims=args.fid_dims, make_grids=not args.no_grids)
