# Forensic Sketch → Photo GAN

A research-grade forensic **sketch-to-photorealistic-face** GAN. It turns a
hand-drawn / semi-forensic face sketch into a photorealistic face through a full
pipeline: **data → preprocessing → training → evaluation → demo.**

The model is a **UNet-256 generator** (with self-attention, residual refinement
and gated skip connections) plus a **multi-scale, spectrally-normalised
PatchGAN** discriminator, trained with a six-term objective and a suite of
modern training tricks (TTUR, AMP, EMA, feature matching).

---

## Highlights

| Area          | What it does |
|---------------|--------------|
| **Generator** | UNet-256 + SAGAN self-attention (16×16), decoder residual refinement, squeeze-excitation skip gating. Upgrades are identity-initialised and config-gated → **old checkpoints still load**. |
| **Discriminator** | Multi-scale PatchGAN (full-res + downsampled), **spectral norm** on every conv, exposes intermediate features for feature matching. |
| **Losses**    | GAN + L1 + Perceptual (VGG16) + Identity (FaceNet / ArcFace) + **LPIPS** + **Feature Matching**, all weights configurable. |
| **Training**  | **TTUR** (separate G/D LR), **AMP** mixed precision (CPU-safe), **EMA** generator, gradient clipping, seeding, resume, best-FID checkpointing, rolling retention, separate schedulers. |
| **Data**      | Landmark **face alignment** (dlib / InsightFace / OpenCV + center-crop fallback), **subject-aware leakage-free split**, stronger augmentations. |
| **Synthetic** | Pluggable sketch styles (xDOG / Canny / pencil / adaptive) via a registry. |
| **Eval**      | Direct `pytorch-fid` API + **LPIPS, SSIM, PSNR, identity cosine, Rank-1/5/10, NIQE**, comparison grids, JSON report. |
| **Demo**      | Gradio Blocks: examples, face-detection validation, identity score, download. |
| **Eng**       | Logging, config validation, `Dockerfile`, `environment.yml`, tests, checkpoint management, MIT license. |

---

## Architecture

```
                          ┌──────────────────────────────────────────┐
        sketch (3×256²)   │                GENERATOR                  │   photo (3×256²)
        ───────────────►  │  Encoder  e1..e8  (downsample ×8)         │  ───────────────►
                          │     e4 (16×16) ── SelfAttention           │
                          │  Decoder  d8..d1  (upsample ×8)           │
                          │     skip: SE-gate(e_i) ⊕ d_i              │
                          │     residual refine on d4,d3,d2           │
                          └──────────────────────────────────────────┘
                                              │  fake / real
                                              ▼
                          ┌──────────────────────────────────────────┐
                          │        MULTI-SCALE PatchGAN (SN)          │
                          │   scale 0: 256²  ─┐                       │
                          │   scale 1: 128²  ─┴─► per-patch real/fake │
                          │   + intermediate features (feat-match)    │
                          └──────────────────────────────────────────┘
```

**Loss:**

```
L_G = L_GAN
    + λ_L1        · L1
    + λ_perc      · Perceptual(VGG16)
    + λ_identity  · Identity(FaceNet | ArcFace)
    + λ_lpips     · LPIPS
    + λ_fm        · FeatureMatching(discriminator)

defaults: λ_L1=100, λ_perc=10, λ_identity=5, λ_lpips=1, λ_fm=10
```

> **Identity loss naming:** earlier versions mislabelled this term "ArcFace"
> while actually running **FaceNet** (InceptionResnetV1 / VGGFace2). It is now
> named correctly and defaults to FaceNet. Set
> `loss.identity_backbone: arcface` to use a real ArcFace backbone via
> InsightFace (auto-falls back to FaceNet if InsightFace is unavailable).

---

## Repository layout

```
src/
  models/  generator.py  discriminator.py  losses.py  blocks.py
  data/    dataset.py     alignment.py       preprocess.py
  utils/   config.py ema.py seed.py checkpoint.py logging_utils.py
  train.py  eval.py  metrics.py  niqe.py  model_io.py  demo.py
scripts/   download_data.sh  synth_data.py  sketch_styles.py
configs/   default.yaml         # every hyperparameter, validated at load
tests/     unit tests (run with `pytest -q`)
Dockerfile  environment.yml  requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt          # or: conda env create -f environment.yml
# Optional landmark/ArcFace backends:
pip install dlib insightface onnxruntime
```

The pipeline auto-detects CUDA and **falls back to CPU** (AMP disabled) when no
GPU is present.

---

## 1. Data preparation

### Real paired data (CUFS / CUFSF)
```bash
bash scripts/download_data.sh   # prints manual/Kaggle fallbacks if mirrors are down
```
Sort files so each sketch and its photo share a filename stem:
```
data/cufs/sketch/<id>.png   data/cufs/photo/<id>.jpg
```

### Synthetic pretraining data (CelebA + multiple sketch styles)
```bash
kaggle datasets download -d jessicali9530/celeba-dataset -p data/raw/
unzip data/raw/celeba-dataset.zip -d data/raw/celeba/
# Mix several sketch styles for robustness:
python scripts/synth_data.py --styles xdog pencil canny --max_images 10000
```

### Preprocess → aligned side-by-side pairs (leakage-free split)
```bash
python src/data/preprocess.py \
  --sketch_dir data/cufs/sketch --photo_dir data/cufs/photo \
  --out_dir data/cufs_pairs --backend auto          # subject-aware split by default
```
`--backend` selects the alignment method (`auto|dlib|insightface|opencv|center`);
`--random_split` disables subject-aware grouping.

---

## 2. Training (two-phase)

```bash
# Phase A — pretrain on synthetic pairs (edit pairs_dir/epochs in the config)
python src/train.py --config configs/default.yaml

# Phase B — fine-tune on real CUFS, resuming Phase A weights
python src/train.py --config configs/default.yaml \
  --resume outputs/checkpoints/latest.pt
```

Key config knobs (`configs/default.yaml`, all validated):

```yaml
train: { lr_g: 2e-4, lr_d: 8e-4, ema_decay: 0.999, grad_clip: 0.0,
         eval_fid_every: 0, keep_last: 3 }
model: { use_attention: true, use_residual: true, use_skip_fusion: true,
         num_scales: 2, spectral_norm: true }
amp: true        # mixed precision (auto-off on CPU)
```

Outputs: EMA samples → `outputs/samples/`, checkpoints → `outputs/checkpoints/`
(`latest.pt` for resume, `best.pt` for best validation FID, rolling `epoch_*.pt`),
logs → `outputs/logs/train.log`.

---

## 3. Evaluation

```bash
python src/eval.py --model outputs/checkpoints/best.pt \
  --pairs_dir data/cufs_pairs --out_dir outputs/eval
```

Reports **FID, LPIPS, SSIM, PSNR, Identity cosine, Rank-1/5/10, NIQE** to
`outputs/eval/report.json`, and writes `Sketch | Generated | Ground Truth`
grids to `outputs/eval/grids/`.

| Metric | Meaning | Direction |
|--------|---------|-----------|
| FID | Distribution distance (InceptionV3) | ↓ |
| LPIPS | Learned perceptual similarity | ↓ |
| SSIM / PSNR | Structural / pixel fidelity | ↑ |
| Identity cosine | FaceNet embedding match to GT | ↑ |
| Rank-1/5/10 | Correct-identity retrieval @ k (the forensic metric) | ↑ |
| NIQE | No-reference naturalness (pristine model = real photos) | ↓ |

---

## 4. Demo

```bash
python src/demo.py --ckpt outputs/checkpoints/best.pt --share
```
Upload a sketch (+ optional reference photo for an identity-similarity score),
validate face detection, view and download the generated photo.

---

## GPU requirements & training time

| Setting | VRAM | Notes |
|---------|------|-------|
| `batch_size=4`, `ngf=ndf=64`, `num_scales=2`, AMP on | ~8–10 GB | fits a single 12 GB GPU |
| AMP off | ~12–14 GB | use if you see AMP instability |
| CPU | — | supported for smoke tests only |

Rough wall-clock (single RTX 3090-class GPU):

| Phase | Data | Epochs | Time |
|-------|------|--------|------|
| A (pretrain) | ~10k synthetic | 50 | ~4–6 h |
| B (fine-tune) | ~311 CUFS pairs | 200 | ~1–2 h |

Reduce cost by lowering `num_scales` to 1, disabling `use_residual`/
`use_attention`, or dropping the LPIPS/identity terms (`lambda_*: 0`).

---

## Targets & baselines (CUFS)

| Method | FID ↓ | SSIM ↑ | PSNR ↑ | Rank-1 ↑ |
|--------|-------|--------|--------|----------|
| Pix2Pix (baseline) | 72.18 | 0.67 | 18.40 | ~30% |
| CycleGAN | 45.39 | 0.83 | — | — |
| HFs2P | 60.21 | 0.79 | 23.01 | — |
| **This project (target)** | **< 55** | **> 0.75** | **> 21** | **> 50%** |

---

## Testing

```bash
pytest -q      # 21 tests; those needing downloadable weights self-skip offline
```

## License

MIT (code only) — see [LICENSE](LICENSE). Datasets and pretrained backbones keep
their own terms; generated faces are for **research / forensic-assistance only**
and must not be used to misrepresent real individuals.
