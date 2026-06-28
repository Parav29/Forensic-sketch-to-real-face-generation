# Forensic Sketch → Photo GAN

A forensic sketch-to-realistic-face GAN that turns a hand-drawn / semi-forensic
face sketch into a photorealistic face. Built as a full end-to-end pipeline:
**data → preprocessing → training → evaluation → Gradio demo.**

The model is a **UNet-256 generator + 70×70 PatchGAN discriminator** (pix2pix
style), trained with a combined objective:

```
L_total = L_cGAN + λ1·L_L1 + λ2·L_perceptual(VGG16) + λ3·L_identity(ArcFace)
          (λ1=100, λ2=10, λ3=5)
```

The ArcFace identity loss is the forensic ingredient — it pushes generated
faces toward the *correct identity* in face-embedding space, not just a
plausible-looking face.

---

## Repository layout

```
.
├── data/
│   ├── raw/              # downloaded zips
│   ├── cufs/             # 311 pairs: sketch/ photo/
│   ├── cufsf/            # 1194 pairs: sketch/ photo/
│   └── synth/            # CelebA-derived synthetic pairs (xDOG)
├── src/
│   ├── data/
│   │   ├── dataset.py    # SketchPhotoDataset (paired loader)
│   │   └── preprocess.py # align, gamma-invert, pair side-by-side
│   ├── models/
│   │   ├── generator.py     # UNet-256 generator
│   │   ├── discriminator.py # PatchGAN discriminator
│   │   └── losses.py        # cGAN + L1 + perceptual + identity
│   ├── train.py          # training loop (two-phase)
│   ├── eval.py           # FID + SSIM + PSNR + rank-1 accuracy
│   └── demo.py           # Gradio interface
├── scripts/
│   ├── download_data.sh  # CUFS / CUFSF download + manual fallbacks
│   └── synth_data.py     # xDOG filter on CelebA → synthetic pairs
├── configs/default.yaml  # all hyperparameters
├── outputs/              # checkpoints / samples / metrics
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

> Training requires a CUDA GPU. The pipeline falls back to CPU automatically
> (via `torch.cuda.is_available()`) but real training is impractical on CPU.

---

## 1. Data acquisition

### Real paired data (CUFS / CUFSF)

```bash
bash scripts/download_data.sh
```

CUHK mmlab servers are frequently offline, so the script prints **manual
fallback** instructions (Kaggle mirror + reference repos). After downloading,
sort files so that each sketch and its matching photo share the same filename
stem:

```
data/cufs/sketch/<id>.png   data/cufs/photo/<id>.jpg
```

### Synthetic pretraining data (CelebA + xDOG)

```bash
# Get CelebA aligned (~1.3 GB) via Kaggle
kaggle datasets download -d jessicali9530/celeba-dataset -p data/raw/
unzip data/raw/celeba-dataset.zip -d data/raw/celeba/

# Generate pseudo sketch/photo pairs with the xDOG edge filter
python scripts/synth_data.py --max_images 10000
```

---

## 2. Preprocessing

Build side-by-side `[sketch | photo]` 256×512 pairs (with face-align + gamma
inversion on the sketch), split 80/10/10 into train/val/test:

```bash
# Real CUFS
python src/data/preprocess.py \
  --sketch_dir data/cufs/sketch \
  --photo_dir  data/cufs/photo \
  --out_dir    data/cufs_pairs

# Synthetic (for pretraining)
python src/data/preprocess.py \
  --sketch_dir data/synth/sketch \
  --photo_dir  data/synth/photo \
  --out_dir    data/synth_pairs
```

---

## 3. Training (two-phase)

```bash
# Phase A — pretrain on synthetic data
#   (edit configs/default.yaml: pairs_dir: data/synth_pairs, epochs: 50)
python src/train.py --config configs/default.yaml

# Phase B — fine-tune on real CUFS
#   (edit configs/default.yaml: pairs_dir: data/cufs_pairs, epochs: 200)
python src/train.py --config configs/default.yaml \
  --resume outputs/checkpoints/epoch_0050.pt
```

Samples are written to `outputs/samples/` every `sample_every` epochs and
checkpoints to `outputs/checkpoints/`.

---

## 4. Evaluation

```bash
python src/eval.py \
  --model     outputs/checkpoints/final.pt \
  --pairs_dir data/cufs_pairs \
  --out_dir   outputs/eval
```

Reports **FID**, **SSIM**, **PSNR**, and **rank-1 ArcFace recognition
accuracy** to `outputs/eval/results.json`.

---

## 5. Demo

```bash
python src/demo.py --ckpt outputs/checkpoints/final.pt --share
```

Upload a sketch and get a generated photo.

---

## Targets & baselines

| Method                  | FID ↓     | SSIM ↑    | PSNR ↑   | Rank-1 ↑ |
|-------------------------|-----------|-----------|----------|----------|
| Pix2Pix (baseline)      | 72.18     | 0.67      | 18.40    | ~30%     |
| CycleGAN                | 45.39     | 0.83      | —        | —        |
| HFs2P                   | 60.21     | 0.79      | 23.01    | —        |
| **This project (target)** | **< 55** | **> 0.75** | **> 21** | **> 50%** |

*Baseline source: "Quality Guided Sketch-to-Photo Image Synthesis" (2020),
CUFS benchmark.*

### Troubleshooting

- **FID not improving after 50 epochs** → confirm `InstanceNorm2d` (not
  `BatchNorm`); bump `lambda_perceptual` to 20; verify augmentation is
  train-split only.
- **Mode collapse** (D→0, G constant) → drop D learning rate 10×; switch to
  WGAN-GP.
- **Good SSIM but low rank-1** (right look, wrong identity) → raise
  `lambda_identity` to 10–15; confirm ArcFace VGGFace2 weights loaded.

---

## Stretch goals

1. Train on **CUFSF** (1,194 pairs, illumination variation).
2. **CLIP-conditioned attributes** ("add beard", "make older").
3. **CycleGAN** variant for unpaired training.
4. **Difficulty ablation** table (viewed / semi-forensic / forensic).
5. **HuggingFace Spaces** deploy for a public demo URL.
