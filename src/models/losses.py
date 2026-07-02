"""
Combined generator objective:

    L_total = L_cGAN
            + lambda_l1        * L1
            + lambda_perceptual* Perceptual(VGG16)
            + lambda_identity  * Identity(FaceNet / ArcFace)
            + lambda_lpips     * LPIPS
            + lambda_fm        * FeatureMatching(discriminator)

The identity term used to be mislabelled "ArcFace" while actually running
FaceNet (InceptionResnetV1 pretrained on VGGFace2). It is now named correctly,
and a real ArcFace backbone (InsightFace) is available as an opt-in backend.
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """VGG16 feature-space loss on relu2_2 and relu3_3 layers."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.slice1 = nn.Sequential(*list(vgg.features)[:9])    # relu2_2
        self.slice2 = nn.Sequential(*list(vgg.features)[9:16])  # relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.l1 = nn.L1Loss()

    def forward(self, generated, target):
        # Denormalise from [-1,1] to [0,1] for VGG
        g = (generated + 1) / 2
        t = (target + 1) / 2
        g1, t1 = self.slice1(g), self.slice1(t)
        g2, t2 = self.slice2(g1), self.slice2(t1)
        return self.l1(g1, t1) + self.l1(g2, t2)


class _FaceNetBackbone(nn.Module):
    """InceptionResnetV1 (VGGFace2) — the historical, dependency-light backbone."""

    input_size = 160

    def __init__(self, device):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.model(x)


class _ArcFaceBackbone(nn.Module):
    """
    Real ArcFace backbone via InsightFace's ``iresnet`` (r50, glint360k).

    This is the "Option A" backbone. It requires ``insightface`` + the model
    weights to be available locally; if anything is missing we raise so the
    caller can fall back to FaceNet with a clear message.
    """

    input_size = 112

    def __init__(self, device):
        super().__init__()
        try:
            from insightface.model_zoo import get_model
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ArcFace backend requires the 'insightface' package. "
                "Install it or use identity_backbone='facenet'."
            ) from e
        # Recognition model returns L2-normalised 512-d embeddings.
        self.model = get_model("arcface_r50_v1")
        self.model.prepare(ctx_id=0 if str(device).startswith("cuda") else -1)
        self.device = device

    def forward(self, x):
        # InsightFace expects BGR-ish [0,255]; we approximate by rescaling the
        # [-1,1] tensor. Kept differentiable for the generator gradient path.
        x = (x + 1) * 127.5
        return self.model.forward(x)


class IdentityLoss(nn.Module):
    """
    Cosine identity loss — penalises generated faces whose embedding does not
    match the ground-truth photo. This is the forensic ingredient: it pushes
    the generator toward the *correct identity*, not merely a plausible face.

    Backbones:
      * ``backbone="facenet"`` (default) – FaceNet / InceptionResnetV1 VGGFace2.
      * ``backbone="arcface"``          – real ArcFace via InsightFace (Option A);
        falls back to FaceNet if InsightFace is unavailable.

    Gradients flow through the generated image; the target embedding is
    detached. Backbone weights are always frozen.
    """

    def __init__(self, device="cuda", backbone="facenet"):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "arcface":
            try:
                self.backbone = _ArcFaceBackbone(device)
            except RuntimeError as e:
                warnings.warn(f"{e} Falling back to FaceNet.")
                self.backbone_name = "facenet"
                self.backbone = _FaceNetBackbone(device)
        else:
            self.backbone = _FaceNetBackbone(device)
        self.size = self.backbone.input_size
        self.cos = nn.CosineSimilarity(dim=1)

    def _resize(self, x):
        return F.interpolate(x, size=(self.size, self.size),
                             mode="bilinear", align_corners=False)

    def forward(self, generated, target):
        emb_g = self.backbone(self._resize(generated))   # grad flows to G
        with torch.no_grad():
            emb_t = self.backbone(self._resize(target))   # fixed reference
        return (1 - self.cos(emb_g, emb_t)).mean()


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (Zhang et al. 2018).

    Wraps the ``lpips`` package (AlexNet backbone by default). Expects images in
    [-1, 1], which is exactly the range the generator produces, so no rescaling
    is needed.
    """

    def __init__(self, net="alex", device="cuda"):
        super().__init__()
        import lpips
        self.model = lpips.LPIPS(net=net).to(device)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, generated, target):
        return self.model(generated, target).mean()


class FeatureMatchingLoss(nn.Module):
    """
    Feature-matching loss (pix2pixHD): L1 distance between the discriminator's
    intermediate activations for real vs. generated images. Stabilises training
    and improves detail by asking the generator to match discriminator
    statistics, not just fool the final logit.

    Accepts the per-scale ``(pred, features)`` output of
    ``MultiScaleDiscriminator`` for both real and fake, and averages across
    scales and layers.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, fake_out, real_out):
        loss = 0.0
        n = 0
        for (_, feats_fake), (_, feats_real) in zip(fake_out, real_out):
            for ff, fr in zip(feats_fake, feats_real):
                loss = loss + self.l1(ff, fr.detach())
                n += 1
        return loss / max(n, 1)


class GANLoss(nn.Module):
    """Label-smoothed BCE for the GAN adversarial loss (supports multi-scale)."""

    def __init__(self, smooth_real=0.9):
        super().__init__()
        self.real_label = smooth_real
        self.fake_label = 0.0
        self.loss = nn.BCEWithLogitsLoss()

    def _single(self, pred, is_real: bool):
        label = torch.full_like(pred,
                                self.real_label if is_real else self.fake_label)
        return self.loss(pred, label)

    def __call__(self, pred, is_real: bool):
        # ``pred`` may be a single tensor (PatchGAN) or a list of per-scale
        # ``(logit, features)`` tuples (MultiScaleDiscriminator).
        if isinstance(pred, (list, tuple)) and len(pred) and isinstance(pred[0], (list, tuple)):
            losses = [self._single(p[0], is_real) for p in pred]
            return sum(losses) / len(losses)
        return self._single(pred, is_real)
