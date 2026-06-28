"""
Main training loop.

Two-phase strategy:
  Phase A (pretraining): train on synthetic CelebA-xDOG pairs (10k images, ~50 epochs)
  Phase B (fine-tuning): fine-tune on real CUFS pairs (311 pairs, 200 epochs)

Run phase A first, then pass --resume to phase B.
"""
import argparse
import json
import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import torchvision.utils as vutils

# Allow `python src/train.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import SketchPhotoDataset
from models.generator import UNetGenerator
from models.discriminator import PatchGAN
from models.losses import GANLoss, PerceptualLoss, IdentityLoss


def train(cfg: dict, resume: str = None):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # --- Data ---
    train_ds = SketchPhotoDataset(cfg["data"]["pairs_dir"], "train",
                                  cfg["data"]["image_size"], augment=True)
    val_ds = SketchPhotoDataset(cfg["data"]["pairs_dir"], "val",
                                cfg["data"]["image_size"], augment=False)
    train_dl = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"],
                          shuffle=True, num_workers=cfg["data"]["num_workers"],
                          pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # --- Models ---
    G = UNetGenerator(ngf=cfg["model"]["ngf"]).to(device)
    D = PatchGAN(ndf=cfg["model"]["ndf"]).to(device)

    # Weight init: normal(0, 0.02) as per the pix2pix paper
    def init_weights(m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
    G.apply(init_weights)
    D.apply(init_weights)

    # --- Losses ---
    crit_gan = GANLoss(smooth_real=0.9)
    crit_l1 = torch.nn.L1Loss()
    crit_perc = PerceptualLoss().to(device)
    crit_id = IdentityLoss(device=str(device)).to(device)

    # --- Optimizers ---
    lr = cfg["train"]["lr"]
    opt_G = optim.Adam(G.parameters(), lr=lr,
                       betas=(cfg["train"]["beta1"], cfg["train"]["beta2"]))
    opt_D = optim.Adam(D.parameters(), lr=lr,
                       betas=(cfg["train"]["beta1"], cfg["train"]["beta2"]))

    # --- LR schedulers: linear decay after decay_start epochs ---
    decay_start = cfg["train"]["decay_start"]
    total_epochs = cfg["train"]["epochs"]

    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / (total_epochs - decay_start))

    sched_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    # --- Optional W&B ---
    use_wandb = cfg.get("wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project="forensic-sketch2photo", config=cfg)

    # --- Resume ---
    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        if "opt_G" in ckpt:
            opt_G.load_state_dict(ckpt["opt_G"])
            opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Resumed from epoch {start_epoch}")

    out_ckpt = Path(cfg["paths"]["checkpoints"])
    out_samples = Path(cfg["paths"]["samples"])
    out_metrics = Path(cfg["paths"]["metrics"])
    out_ckpt.mkdir(parents=True, exist_ok=True)
    out_samples.mkdir(parents=True, exist_ok=True)
    out_metrics.mkdir(parents=True, exist_ok=True)

    l1_w = cfg["loss"]["lambda_l1"]
    perc_w = cfg["loss"]["lambda_perceptual"]
    id_w = cfg["loss"]["lambda_identity"]

    metrics_log = []

    # ===================== TRAINING LOOP =====================
    for epoch in range(start_epoch, total_epochs):
        G.train()
        D.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{total_epochs}")
        for batch in pbar:
            sketch = batch["sketch"].to(device)
            real_photo = batch["photo"].to(device)

            # ---- Train Discriminator ----
            fake_photo = G(sketch).detach()

            opt_D.zero_grad()
            pred_real = D(sketch, real_photo)
            pred_fake = D(sketch, fake_photo)
            loss_D = 0.5 * (crit_gan(pred_real, True) + crit_gan(pred_fake, False))
            loss_D.backward()
            opt_D.step()

            # ---- Train Generator ----
            fake_photo = G(sketch)

            opt_G.zero_grad()
            pred_fake = D(sketch, fake_photo)

            loss_adv = crit_gan(pred_fake, True)
            loss_l1 = crit_l1(fake_photo, real_photo) * l1_w
            loss_perc = crit_perc(fake_photo, real_photo) * perc_w
            loss_id = crit_id(fake_photo, real_photo) * id_w
            loss_G = loss_adv + loss_l1 + loss_perc + loss_id

            loss_G.backward()
            opt_G.step()

            epoch_d_loss += loss_D.item()
            epoch_g_loss += loss_G.item()
            pbar.set_postfix(D=f"{loss_D.item():.3f}", G=f"{loss_G.item():.3f}")

        sched_G.step()
        sched_D.step()

        avg_d = epoch_d_loss / len(train_dl)
        avg_g = epoch_g_loss / len(train_dl)
        metrics_log.append({"epoch": epoch + 1, "D": avg_d, "G": avg_g})
        print(f"Epoch {epoch + 1}: D={avg_d:.4f} G={avg_g:.4f}")
        if use_wandb:
            wandb.log({"epoch": epoch + 1, "loss_D": avg_d, "loss_G": avg_g})

        # ---- Save samples ----
        if (epoch + 1) % cfg["train"]["sample_every"] == 0 and len(val_ds) > 0:
            G.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_dl))
                s = sample_batch["sketch"].to(device)
                p = sample_batch["photo"].to(device)
                f = G(s)
                grid = vutils.make_grid(
                    torch.cat([s, f, p], dim=0), nrow=3,
                    normalize=True, value_range=(-1, 1)
                )
                vutils.save_image(grid, out_samples / f"epoch_{epoch + 1:04d}.png")
            G.train()

        # ---- Save checkpoint ----
        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            torch.save({
                "epoch": epoch,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            }, out_ckpt / f"epoch_{epoch + 1:04d}.pt")

    # Final checkpoint
    torch.save({"epoch": total_epochs, "G": G.state_dict(),
                "D": D.state_dict()}, out_ckpt / "final.pt")

    # Save metrics log
    with open(out_metrics / "train_log.json", "w") as fh:
        json.dump(metrics_log, fh, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume=args.resume)
