"""Grad-CAM for the visual branch.

Turns "this page looks 0.94 like PayPal" into a heatmap over the screenshot that
localises *why* -- typically the logo region and the login form. For a metric
model there is no class logit to differentiate, so we differentiate the cosine
similarity to the matched brand anchor, which is the quantity the detection
decision actually rests on.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gradcam(encoder, image: torch.Tensor, anchor_vec: torch.Tensor,
            device) -> np.ndarray:
    """Return a HxW heatmap in [0,1] for one image vs one brand anchor."""
    encoder.eval()
    handle = encoder._hook()
    try:
        x = image.unsqueeze(0).to(device).requires_grad_(True)
        z = encoder(x)                                   # normalised embedding
        anchor = F.normalize(anchor_vec.reshape(1, -1).to(device), dim=1)
        similarity = (z * anchor).sum()                  # cosine, the decision axis
        encoder.zero_grad()
        similarity.backward()

        acts = encoder._activations            # (1, C, h, w)
        grads = encoder._gradients             # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)   # GAP over spatial dims
        cam = F.relu((weights * acts).sum(dim=1)).squeeze(0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=(image.shape[1], image.shape[2]),
                            mode="bilinear", align_corners=False)[0, 0]
        return cam.detach().cpu().numpy()
    finally:
        handle.remove()


def overlay(cam: np.ndarray, image_chw: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    """Blend the heatmap over the de-normalised screenshot for display."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = (image_chw.cpu().numpy() * std + mean).clip(0, 1).transpose(1, 2, 0)
    # Simple red-hot colourisation without a matplotlib dependency at serve time.
    heat = np.zeros_like(img)
    heat[..., 0] = cam
    heat[..., 1] = np.clip(cam - 0.4, 0, 1)
    return (img * (1 - alpha) + heat * alpha).clip(0, 1)
