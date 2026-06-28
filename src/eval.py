"""
Evaluation metrics:
- FID (Fréchet Inception Distance) — overall generation quality
- SSIM — structural similarity
- PSNR — peak signal-to-noise ratio
- Rank-1 face recognition accuracy — the forensic metric that matters
"""
import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Allow `python src/eval.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def evaluate(model_path: str, pairs_dir: str, output_dir: str, device="cuda"):
    from models.generator import UNetGenerator
    from data.dataset import SketchPhotoDataset
    from torch.utils.data import DataLoader

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    G = UNetGenerator().to(device)
    ckpt = torch.load(model_path, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    test_ds = SketchPhotoDataset(pairs_dir, "test", augment=False)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    out_path = Path(output_dir)
    (out_path / "generated").mkdir(parents=True, exist_ok=True)
    (out_path / "real").mkdir(parents=True, exist_ok=True)

    ssim_scores, psnr_scores = [], []

    print("Generating test outputs...")
    with torch.no_grad():
        for batch in tqdm(test_dl):
            sketch = batch["sketch"].to(device)
            real = batch["photo"].to(device)
            fname = batch["filename"][0]

            fake = G(sketch)

            # Save for FID computation
            def to_pil(t):
                img = (t.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                return Image.fromarray(img)

            to_pil(fake).save(out_path / "generated" / f"{fname}.png")
            to_pil(real).save(out_path / "real" / f"{fname}.png")

            # SSIM and PSNR
            fake_np = (fake.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2
            real_np = (real.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2
            ssim_scores.append(ssim(real_np, fake_np, channel_axis=2, data_range=1.0))
            psnr_scores.append(psnr(real_np, fake_np, data_range=1.0))

    # FID
    result = subprocess.run([
        sys.executable, "-m", "pytorch_fid",
        str(out_path / "real"), str(out_path / "generated"),
        "--device", str(device)
    ], capture_output=True, text=True)
    fid_line = [l for l in result.stdout.splitlines() if "FID" in l]
    fid = float(fid_line[0].split()[-1]) if fid_line else -1

    # Rank-1 face recognition accuracy
    rank1 = compute_rank1_accuracy(
        gen_dir=str(out_path / "generated"),
        real_dir=str(out_path / "real"),
        device=str(device)
    )

    results = {
        "FID": round(fid, 4),
        "SSIM": round(float(np.mean(ssim_scores)), 4),
        "PSNR": round(float(np.mean(psnr_scores)), 4),
        "Rank1_Accuracy": round(rank1, 4),
        "n_samples": len(ssim_scores),
    }

    print("\n=== EVALUATION RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    with open(out_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def compute_rank1_accuracy(gen_dir: str, real_dir: str, device: str = "cuda") -> float:
    """
    For each generated photo, retrieve the most similar real photo by ArcFace
    embedding. Rank-1 accuracy = fraction of queries where the top-1 retrieval
    is the correct identity.
    """
    from facenet_pytorch import InceptionResnetV1
    import torch.nn.functional as F

    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def get_embedding(img_path):
        img = Image.open(img_path).convert("RGB").resize((160, 160))
        t = torch.tensor(np.array(img) / 255.0).permute(2, 0, 1).float().unsqueeze(0)
        t = (t - 0.5) / 0.5
        with torch.no_grad():
            return model(t.to(device)).cpu()

    gen_files = sorted(Path(gen_dir).glob("*.png"))
    real_files = sorted(Path(real_dir).glob("*.png"))
    if not gen_files:
        return -1.0

    print("Computing ArcFace embeddings...")
    gen_embs = torch.cat([get_embedding(f) for f in tqdm(gen_files)])
    real_embs = torch.cat([get_embedding(f) for f in tqdm(real_files)])

    gen_names = [f.stem for f in gen_files]
    real_names = [f.stem for f in real_files]

    correct = 0
    for emb, name in zip(gen_embs, gen_names):
        sims = F.cosine_similarity(emb.unsqueeze(0), real_embs)
        top1_idx = sims.argmax().item()
        if real_names[top1_idx] == name:
            correct += 1

    return correct / len(gen_files)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="outputs/checkpoints/final.pt")
    p.add_argument("--pairs_dir", default="data/cufs_pairs")
    p.add_argument("--out_dir", default="outputs/eval")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    evaluate(args.model, args.pairs_dir, args.out_dir, args.device)
