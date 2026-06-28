"""
Combined loss: cGAN + L1 + Perceptual (VGG) + Identity (ArcFace)

L_total = L_cGAN + λ1*L_L1 + λ2*L_perceptual + λ3*L_identity
Typical: λ1=100, λ2=10, λ3=5
"""
import torch
import torch.nn as nn
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


class IdentityLoss(nn.Module):
    """
    ArcFace-style cosine identity loss — penalises generated faces that don't
    match the ground-truth photo in embedding space.
    Uses facenet-pytorch (InceptionResnetV1 / VGGFace2) as the backbone.

    NOTE: gradients must flow through the *generated* image so the generator
    learns to preserve identity. Only the backbone weights are frozen; the
    target embedding is detached.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for p in self.parameters():
            p.requires_grad = False
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, generated, target):
        # Resize to 160x160 (InceptionResnet input size)
        g = nn.functional.interpolate(generated, size=(160, 160),
                                      mode="bilinear", align_corners=False)
        t = nn.functional.interpolate(target, size=(160, 160),
                                      mode="bilinear", align_corners=False)
        emb_g = self.model(g)                 # grad flows to the generator
        with torch.no_grad():
            emb_t = self.model(t)             # target is a fixed reference
        # 1 - cosine similarity as loss
        return (1 - self.cos(emb_g, emb_t)).mean()


class GANLoss(nn.Module):
    """Label-smoothed BCE for the GAN adversarial loss."""

    def __init__(self, smooth_real=0.9):
        super().__init__()
        self.real_label = smooth_real
        self.fake_label = 0.0
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, pred, is_real: bool):
        label = torch.full_like(pred,
                                self.real_label if is_real else self.fake_label)
        return self.loss(pred, label)
