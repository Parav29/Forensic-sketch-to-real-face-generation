"""Exponential Moving Average of model weights."""
import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Maintain an exponential moving average of a model's parameters and buffers.

    EMA weights typically yield noticeably smoother, higher-quality samples for
    GANs. Call :meth:`update` after every optimizer step; use :meth:`state_dict`
    to checkpoint and :meth:`copy_to` / :meth:`apply_shadow` to run inference
    with the averaged weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # A frozen deep copy that holds the shadow (averaged) weights.
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        ema_params = dict(self.ema_model.named_parameters())
        for name, p in model.named_parameters():
            if p.requires_grad:
                ema_params[name].mul_(d).add_(p.detach(), alpha=1.0 - d)
        # Buffers (e.g. norm running stats) are copied verbatim.
        ema_buffers = dict(self.ema_model.named_buffers())
        for name, b in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(b)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        return self.ema_model.load_state_dict(state_dict, strict=strict)

    def copy_to(self, model: nn.Module):
        """Copy EMA weights into ``model`` (in place)."""
        model.load_state_dict(self.ema_model.state_dict(), strict=False)
