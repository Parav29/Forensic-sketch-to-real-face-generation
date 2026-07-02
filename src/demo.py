"""
Gradio web demo (Blocks UI).

Features:
  * Example sketches auto-loaded from data/cufs/sketch/ (if present).
  * Face-detection validation on the uploaded sketch.
  * Optional reference photo -> identity cosine-similarity score.
  * Downloadable generated image and a tidy two-column layout.
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_io import load_generator
from data.alignment import FaceAligner
from utils import get_logger

logger = get_logger("demo")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE = {"G": None, "aligner": None, "id_model": None}


def load_model(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        STATE["G"], info = load_generator(checkpoint_path, DEVICE, prefer_ema=True)
        logger.info(f"Loaded checkpoint: {checkpoint_path} (ema={info['used_ema']})")
    else:
        from models.generator import UNetGenerator
        STATE["G"] = UNetGenerator().eval().to(DEVICE)
        logger.warning(f"Checkpoint '{checkpoint_path}' not found — using random "
                       "weights (output will be noise).")
    STATE["aligner"] = FaceAligner(backend="auto")
    try:
        from facenet_pytorch import InceptionResnetV1
        STATE["id_model"] = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    except Exception as e:
        logger.warning(f"Identity model unavailable: {e}")
    return STATE["G"]


def _has_face(img_rgb: np.ndarray) -> bool:
    aligner = STATE["aligner"]
    if aligner is None or aligner._impl is None:
        return True  # no detector available -> don't block the user
    try:
        return aligner._impl(img_rgb) is not None
    except Exception:
        return True


def _to_tensor(pil_img):
    arr = np.array(pil_img.convert("RGB").resize((256, 256))) / 255.0
    t = torch.tensor(arr).permute(2, 0, 1).float()
    return ((t - 0.5) / 0.5).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def _embed(pil_img):
    if STATE["id_model"] is None:
        return None
    arr = np.array(pil_img.convert("RGB").resize((160, 160))) / 255.0
    t = torch.tensor(arr).permute(2, 0, 1).float()
    t = ((t - 0.5) / 0.5).unsqueeze(0).to(DEVICE)
    return STATE["id_model"](t)


def generate(sketch_img, reference_img):
    if sketch_img is None:
        return None, "Please upload a sketch.", None

    sketch_rgb = np.array(sketch_img.convert("RGB"))
    face_msg = ("✅ Face detected in sketch." if _has_face(sketch_rgb)
                else "⚠️ No face detected — result may be unreliable.")

    with torch.no_grad():
        out = STATE["G"](_to_tensor(sketch_img)).squeeze(0).cpu()
    out_np = np.clip(((out.numpy().transpose(1, 2, 0) + 1) / 2) * 255, 0, 255).astype("uint8")
    generated = Image.fromarray(out_np)

    id_msg = "Upload a reference photo to compute identity similarity."
    if reference_img is not None and STATE["id_model"] is not None:
        try:
            eg = _embed(generated)
            er = _embed(reference_img)
            cos = torch.nn.functional.cosine_similarity(eg, er).item()
            id_msg = f"Identity cosine similarity vs. reference: {cos:.3f}"
        except Exception as e:
            id_msg = f"Identity similarity unavailable: {e}"

    # Save for download.
    out_path = Path("outputs") / "demo_generated.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generated.save(out_path)
    return generated, f"{face_msg}\n{id_msg}", str(out_path)


def _example_sketches():
    ex_dir = Path("data/cufs/sketch")
    if not ex_dir.exists():
        return []
    files = sorted(ex_dir.glob("*.png")) + sorted(ex_dir.glob("*.jpg"))
    return [[str(f), None] for f in files[:6]]


def build_demo():
    with gr.Blocks(title="Forensic Sketch → Photo") as demo:
        gr.Markdown(
            "# Forensic Sketch → Photo (pix2pix GAN)\n"
            "Upload a forensic / hand-drawn face sketch to synthesise a "
            "photorealistic face. Optionally add a reference photo to score "
            "identity preservation.")
        with gr.Row():
            with gr.Column():
                sketch_in = gr.Image(type="pil", label="Face sketch (input)")
                ref_in = gr.Image(type="pil", label="Reference photo (optional)")
                run_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out_img = gr.Image(type="pil", label="Generated photo")
                status = gr.Textbox(label="Status", lines=2, interactive=False)
                download = gr.File(label="Download generated image")
        examples = _example_sketches()
        if examples:
            gr.Examples(examples=examples, inputs=[sketch_in, ref_in],
                        label="Example sketches")
        run_btn.click(generate, inputs=[sketch_in, ref_in],
                      outputs=[out_img, status, download])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.environ.get(
        "CKPT", "outputs/checkpoints/best.pt"))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    load_model(args.ckpt)
    build_demo().launch(share=args.share)
