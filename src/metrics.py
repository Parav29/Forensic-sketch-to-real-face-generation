"""
Evaluation metrics, computed directly via library APIs (no subprocess):

  * FID  – pytorch-fid ``calculate_fid_given_paths`` (direct API)
  * SSIM / PSNR – skimage
  * LPIPS – lpips package
  * Identity cosine similarity + Rank-1/5/10 – FaceNet embeddings
  * NIQE – self-contained (see niqe.py)
"""
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import niqe as _niqe  # noqa: E402


# --------------------------------------------------------------------------- #
# FID (direct pytorch-fid API)
# --------------------------------------------------------------------------- #
def compute_fid(real_dir, gen_dir, device="cpu", batch_size=50, dims=2048):
    from pytorch_fid.fid_score import calculate_fid_given_paths
    n = len(list(Path(gen_dir).glob("*.png")))
    batch_size = max(1, min(batch_size, n))
    return float(calculate_fid_given_paths(
        [str(real_dir), str(gen_dir)], batch_size, str(device), dims,
        num_workers=0))


# --------------------------------------------------------------------------- #
# Full-reference pixel/structure metrics
# --------------------------------------------------------------------------- #
def ssim_psnr(fake_np, real_np):
    """Both inputs HxWx3 float in [0,1]."""
    s = ssim_fn(real_np, fake_np, channel_axis=2, data_range=1.0)
    p = psnr_fn(real_np, fake_np, data_range=1.0)
    return float(s), float(p)


# --------------------------------------------------------------------------- #
# LPIPS
# --------------------------------------------------------------------------- #
class LPIPSMetric:
    def __init__(self, net="alex", device="cpu"):
        import lpips
        self.model = lpips.LPIPS(net=net).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, fake, real):
        """Tensors in [-1,1], shape (B,3,H,W)."""
        return float(self.model(fake.to(self.device), real.to(self.device)).mean())


# --------------------------------------------------------------------------- #
# Identity: cosine similarity + Rank-k retrieval (FaceNet embeddings)
# --------------------------------------------------------------------------- #
class IdentityMetric:
    def __init__(self, device="cpu"):
        from facenet_pytorch import InceptionResnetV1
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self.device = device

    @torch.no_grad()
    def embed_path(self, path):
        img = Image.open(path).convert("RGB").resize((160, 160))
        t = torch.tensor(np.array(img) / 255.0).permute(2, 0, 1).float()
        t = ((t - 0.5) / 0.5).unsqueeze(0).to(self.device)
        return self.model(t).cpu()

    def evaluate(self, gen_dir, real_dir, ranks=(1, 5, 10)):
        gen_files = sorted(Path(gen_dir).glob("*.png"))
        real_files = sorted(Path(real_dir).glob("*.png"))
        if not gen_files:
            return {"identity_cosine": -1.0,
                    **{f"rank{k}": -1.0 for k in ranks}}

        gen_emb = torch.cat([self.embed_path(f) for f in gen_files])
        real_emb = torch.cat([self.embed_path(f) for f in real_files])
        gen_names = [f.stem for f in gen_files]
        real_names = [f.stem for f in real_files]
        real_index = {name: i for i, name in enumerate(real_names)}

        gen_n = F.normalize(gen_emb, dim=1)
        real_n = F.normalize(real_emb, dim=1)
        sims = gen_n @ real_n.t()                     # (n_gen, n_real)

        # Paired cosine similarity (same identity generated vs. real).
        cos_scores = []
        for i, name in enumerate(gen_names):
            if name in real_index:
                cos_scores.append(float(sims[i, real_index[name]]))
        cos_mean = float(np.mean(cos_scores)) if cos_scores else -1.0

        # Rank-k retrieval accuracy.
        order = sims.argsort(dim=1, descending=True)
        rank_hits = {k: 0 for k in ranks}
        max_k = max(ranks)
        for i, name in enumerate(gen_names):
            topk = [real_names[j] for j in order[i, :max_k].tolist()]
            for k in ranks:
                if name in topk[:k]:
                    rank_hits[k] += 1
        n = len(gen_names)
        out = {"identity_cosine": round(cos_mean, 4)}
        out.update({f"rank{k}": round(rank_hits[k] / n, 4) for k in ranks})
        return out


# --------------------------------------------------------------------------- #
# NIQE (no-reference; pristine model fitted on the real photos)
# --------------------------------------------------------------------------- #
def compute_niqe(gen_dir, real_dir):
    def gray(path):
        return np.array(Image.open(path).convert("L"), dtype=np.float64)

    real_files = sorted(Path(real_dir).glob("*.png"))
    gen_files = sorted(Path(gen_dir).glob("*.png"))
    if len(real_files) < 2 or not gen_files:
        return -1.0
    pristine = [_niqe.extract_features(gray(f)) for f in real_files]
    mu, cov = _niqe.fit_mvg(pristine)
    scores = [_niqe.niqe_distance(_niqe.extract_features(gray(f)), mu, cov)
              for f in gen_files]
    return round(float(np.mean(scores)), 4)
