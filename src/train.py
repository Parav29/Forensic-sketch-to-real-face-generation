"""
Main training loop.

Research-grade training features:
  * TTUR — separate generator / discriminator learning rates + schedulers.
  * Mixed precision (AMP) — autocast + GradScaler, auto-disabled on CPU.
  * EMA — exponential moving average of generator weights (used for sampling,
    validation and the exported checkpoints).
  * Combined loss — GAN + L1 + Perceptual + Identity + LPIPS + FeatureMatching.
  * Multi-scale spectral-norm PatchGAN discriminator.
  * Best-checkpoint tracking by validation FID, rolling retention, gradient
    clipping, reproducible seeding and full resume support.

Two-phase strategy:
  Phase A (pretraining): synthetic sketch/photo pairs.
  Phase B (fine-tuning): real CUFS pairs (pass ``--resume``).
"""
import os
import sys
import json
import argparse
import tempfile
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import SketchPhotoDataset
from models.generator import UNetGenerator
from models.discriminator import MultiScaleDiscriminator
from models.losses import (GANLoss, PerceptualLoss, IdentityLoss,
                           LPIPSLoss, FeatureMatchingLoss)
from utils import (get_logger, ModelEMA, set_seed, load_config, CheckpointManager)


def build_lr_lambda(decay_start, total_epochs):
    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / max(1, total_epochs - decay_start))
    return lr_lambda


@torch.no_grad()
def _save_val_images(generator, val_dl, device, out_dir):
    """Generate images for the whole val set (used for FID) into out_dir."""
    real_dir = Path(out_dir) / "real"
    gen_dir = Path(out_dir) / "gen"
    real_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    import numpy as np

    def to_pil(t):
        img = (t.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) / 2
        return Image.fromarray(np.clip(img * 255, 0, 255).astype("uint8"))

    for batch in val_dl:
        s = batch["sketch"].to(device)
        f = generator(s)
        name = batch["filename"][0]
        to_pil(f).save(gen_dir / f"{name}.png")
        to_pil(batch["photo"]).save(real_dir / f"{name}.png")
    return real_dir, gen_dir


def train(cfg: dict, resume: str = None):
    set_seed(cfg["seed"], cfg["deterministic"])
    logger = get_logger("train", logfile=str(Path(cfg["paths"]["logs"]) / "train.log"))

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["amp"]) and device.type == "cuda"   # AMP only on CUDA
    logger.info(f"Device={device} | AMP={amp}")

    # --- Data ---
    dc = cfg["data"]
    train_ds = SketchPhotoDataset(dc["pairs_dir"], "train", dc["image_size"], augment=True)
    val_ds = SketchPhotoDataset(dc["pairs_dir"], "val", dc["image_size"], augment=False)
    train_dl = DataLoader(train_ds, batch_size=dc["batch_size"], shuffle=True,
                          num_workers=dc["num_workers"],
                          pin_memory=(device.type == "cuda"), drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    logger.info(f"Train pairs={len(train_ds)} | Val pairs={len(val_ds)}")

    # --- Models ---
    mc = cfg["model"]
    G = UNetGenerator(ngf=mc["ngf"], use_attention=mc["use_attention"],
                      use_residual=mc["use_residual"],
                      use_skip_fusion=mc["use_skip_fusion"]).to(device)
    D = MultiScaleDiscriminator(ndf=mc["ndf"], num_scales=mc["num_scales"],
                                spectral=mc["spectral_norm"]).to(device)

    def init_weights(m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            if getattr(m, "weight", None) is not None and m.weight.requires_grad:
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
    G.apply(init_weights)
    G.init_enhancements()   # keep upgrade modules identity-initialised

    # --- Losses ---
    lc = cfg["loss"]
    crit_gan = GANLoss(smooth_real=0.9)
    crit_l1 = torch.nn.L1Loss()
    crit_perc = PerceptualLoss().to(device) if lc["lambda_perceptual"] > 0 else None
    crit_id = (IdentityLoss(device=str(device), backbone=lc["identity_backbone"]).to(device)
               if lc["lambda_identity"] > 0 else None)
    crit_lpips = (LPIPSLoss(net=lc["lpips_net"], device=str(device))
                  if lc["lambda_lpips"] > 0 else None)
    crit_fm = FeatureMatchingLoss() if lc["lambda_fm"] > 0 else None

    # --- Optimizers (TTUR) + separate schedulers ---
    tc = cfg["train"]
    betas = (tc["beta1"], tc["beta2"])
    opt_G = optim.Adam(G.parameters(), lr=tc["lr_g"], betas=betas)
    opt_D = optim.Adam(D.parameters(), lr=tc["lr_d"], betas=betas)
    lr_lambda = build_lr_lambda(tc["decay_start"], tc["epochs"])
    sched_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    # --- AMP + EMA + checkpoints ---
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ema = ModelEMA(G, decay=tc["ema_decay"])
    ckpt_mgr = CheckpointManager(cfg["paths"]["checkpoints"],
                                 keep_last=tc["keep_last"], best_mode="min")

    use_wandb = cfg.get("wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project="forensic-sketch2photo", config=cfg)

    # --- Resume ---
    start_epoch = 0
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device)
        G.load_state_dict(ck["G"], strict=False)
        if "D" in ck:
            D.load_state_dict(ck["D"], strict=False)
        for key, obj in (("opt_G", opt_G), ("opt_D", opt_D),
                         ("sched_G", sched_G), ("sched_D", sched_D)):
            if key in ck:
                obj.load_state_dict(ck[key])
        if "scaler" in ck and amp:
            scaler.load_state_dict(ck["scaler"])
        if "ema" in ck:
            ema.load_state_dict(ck["ema"], strict=False)
        if "best_metric" in ck:
            ckpt_mgr.best_metric = ck["best_metric"]
        start_epoch = ck.get("epoch", -1) + 1
        logger.info(f"Resumed from {resume} at epoch {start_epoch}")

    for p in ("samples", "metrics"):
        Path(cfg["paths"][p]).mkdir(parents=True, exist_ok=True)

    l1_w, perc_w = lc["lambda_l1"], lc["lambda_perceptual"]
    id_w, lpips_w, fm_w = lc["lambda_identity"], lc["lambda_lpips"], lc["lambda_fm"]
    grad_clip = tc["grad_clip"]
    metrics_log = []

    # ===================== TRAINING LOOP =====================
    for epoch in range(start_epoch, tc["epochs"]):
        G.train(); D.train()
        ep_d = ep_g = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{tc['epochs']}")
        for batch in pbar:
            sketch = batch["sketch"].to(device, non_blocking=True)
            real = batch["photo"].to(device, non_blocking=True)

            # -------- Discriminator --------
            opt_D.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp):
                fake = G(sketch)
                pred_real = D(sketch, real)
                pred_fake = D(sketch, fake.detach())
                loss_D = 0.5 * (crit_gan(pred_real, True) + crit_gan(pred_fake, False))
            scaler.scale(loss_D).backward()
            if grad_clip > 0:
                scaler.unscale_(opt_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), grad_clip)
            scaler.step(opt_D)

            # -------- Generator --------
            opt_G.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp):
                fake = G(sketch)
                pred_fake = D(sketch, fake)
                loss_adv = crit_gan(pred_fake, True)
                loss_G = loss_adv + crit_l1(fake, real) * l1_w
                if crit_perc is not None:
                    loss_G = loss_G + crit_perc(fake, real) * perc_w
                if crit_id is not None:
                    loss_G = loss_G + crit_id(fake, real) * id_w
                if crit_lpips is not None:
                    loss_G = loss_G + crit_lpips(fake, real) * lpips_w
                if crit_fm is not None:
                    with torch.no_grad():
                        pred_real_fm = D(sketch, real)
                    loss_G = loss_G + crit_fm(pred_fake, pred_real_fm) * fm_w
            scaler.scale(loss_G).backward()
            if grad_clip > 0:
                scaler.unscale_(opt_G)
                torch.nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            scaler.step(opt_G)
            scaler.update()

            ema.update(G)   # EMA after every optimization step

            ld, lg = float(loss_D.detach()), float(loss_G.detach())
            ep_d += ld; ep_g += lg
            pbar.set_postfix(D=f"{ld:.3f}", G=f"{lg:.3f}")

        sched_G.step(); sched_D.step()
        avg_d, avg_g = ep_d / len(train_dl), ep_g / len(train_dl)
        rec = {"epoch": epoch + 1, "D": round(avg_d, 4), "G": round(avg_g, 4),
               "lr_g": sched_G.get_last_lr()[0], "lr_d": sched_D.get_last_lr()[0]}
        logger.info(f"Epoch {epoch + 1}: D={avg_d:.4f} G={avg_g:.4f}")
        if use_wandb:
            wandb.log({"loss_D": avg_d, "loss_G": avg_g, **rec})

        # Use EMA generator for all evaluation/sampling.
        ema_G = ema.ema_model.to(device).eval()

        # ---- Samples ----
        if (epoch + 1) % tc["sample_every"] == 0 and len(val_ds) > 0:
            with torch.no_grad():
                sb = next(iter(val_dl))
                s = sb["sketch"].to(device); p = sb["photo"].to(device)
                f = ema_G(s)
                grid = vutils.make_grid(torch.cat([s, f, p], 0), nrow=3,
                                        normalize=True, value_range=(-1, 1))
                vutils.save_image(grid, Path(cfg["paths"]["samples"]) / f"epoch_{epoch + 1:04d}.png")

        # ---- Validation FID (best checkpoint tracking) ----
        fid_every = tc["eval_fid_every"]
        if fid_every and (epoch + 1) % fid_every == 0 and len(val_ds) >= 2:
            try:
                from metrics import compute_fid
                with tempfile.TemporaryDirectory() as tmp:
                    real_dir, gen_dir = _save_val_images(ema_G, val_dl, device, tmp)
                    fid = compute_fid(real_dir, gen_dir, device=device, dims=192)
                rec["val_FID"] = round(fid, 4)
                logger.info(f"  val FID={fid:.4f}")
                if ckpt_mgr.maybe_save_best(_state(epoch, G, D, opt_G, opt_D,
                                                    sched_G, sched_D, scaler, ema, cfg), fid):
                    logger.info(f"  new best FID -> saved best.pt")
            except Exception as e:
                logger.warning(f"  FID computation skipped: {e}")

        metrics_log.append(rec)

        # ---- Checkpoints ----
        state = _state(epoch, G, D, opt_G, opt_D, sched_G, sched_D, scaler, ema, cfg)
        ckpt_mgr.save_latest(state)                         # resume point
        if (epoch + 1) % tc["save_every"] == 0:
            ckpt_mgr.save(state, epoch + 1)

    # Final EMA-based checkpoint for inference.
    torch.save(_state(tc["epochs"] - 1, G, D, opt_G, opt_D,
                      sched_G, sched_D, scaler, ema, cfg),
               Path(cfg["paths"]["checkpoints"]) / "final.pt")
    with open(Path(cfg["paths"]["metrics"]) / "train_log.json", "w") as fh:
        json.dump(metrics_log, fh, indent=2)
    logger.info("Training complete.")


def _state(epoch, G, D, opt_G, opt_D, sched_G, sched_D, scaler, ema, cfg):
    return {
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "ema": ema.state_dict(),      # EMA weights (used for inference)
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "sched_G": sched_G.state_dict(),
        "sched_D": sched_D.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": cfg,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, resume=args.resume)
