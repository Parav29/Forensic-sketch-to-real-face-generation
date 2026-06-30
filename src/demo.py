"""
Gradio web demo.
Upload a sketch -> get a generated photo.
"""
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import gradio as gr

# Allow `python src/demo.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.generator import UNetGenerator

CHECKPOINT = os.environ.get("CKPT", "outputs/checkpoints/final.pt")
G = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_model(checkpoint_path: str = CHECKPOINT):
    global G, DEVICE
    G = UNetGenerator()
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        G.load_state_dict(ckpt["G"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"WARNING: checkpoint '{checkpoint_path}' not found — "
              "using randomly initialised weights (output will be noise).")
    G.eval().to(DEVICE)
    return G


def generate(sketch_img: Image.Image) -> Image.Image:
    """Take a PIL sketch, return a PIL generated photo."""
    if sketch_img is None:
        return None
    from PIL import ImageOps
    
    sketch = sketch_img.convert("RGB").resize((256, 256))
    
    # The model was trained on inverted sketches (white lines on black background).
    # If the user uploads a typical sketch (black lines on a white background),
    # we automatically invert it.
    gray = sketch.convert("L")
    if np.mean(np.array(gray)) > 127:
        sketch = ImageOps.invert(sketch)
        
    t = torch.tensor(np.array(sketch) / 255.0).permute(2, 0, 1).float()
    t = (t - 0.5) / 0.5
    t = t.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = G(t).squeeze().cpu()

    out_np = (out.numpy().transpose(1, 2, 0) + 1) / 2
    out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out_np)


def build_demo():
    return gr.Interface(
        fn=generate,
        inputs=gr.Image(type="pil", label="Upload face sketch"),
        outputs=gr.Image(type="pil", label="Generated photo"),
        title="Forensic Sketch → Photo (pix2pix GAN)",
        description=(
            "Upload a forensic or hand-drawn face sketch. "
            "The model generates a photorealistic face using a UNet + PatchGAN "
            "architecture trained on the CUHK Face Sketch (CUFS) dataset."
        ),
        examples=[],  # add example sketches from data/cufs/sketch/ here
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CHECKPOINT)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    load_model(args.ckpt)
    build_demo().launch(share=args.share)
