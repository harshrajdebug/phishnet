"""Cost-priced visual attacks, and the Phase-2 encoder crucible.

The lexical pilot priced *features*, because there attacks and features coincided.
In pixel space they diverge: a patch has no economic value, but the semantic
operation that alters it does. So we price ATTACKS and sample them during
training with P(a) proportional to exp(-cost(a)/tau).

The conversion axis dominates here in a way it never did lexically. Recolouring a
cloned login page is free and barely dents the victim's belief. Deleting the
brand wordmark also defeats a logo-matching detector for free -- but it removes
the very cue the victim uses, so the attack is close to self-defeating. A
defender should therefore spend its capacity on invariance to recolouring, and
comparatively little on invariance to logo removal.

The encoder objective has two sides that pull against each other:
  invariance   a cheaply degraded brand page must embed near its pristine anchor
  separation   distinct brands must stay far apart (the analogue of clean accuracy)
A defence that achieves the first by collapsing the space fails.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# name -> (c_money, c_effort, c_conversion)
ATTACK_COSTS: dict[str, tuple[float, float, float]] = {
    "identity":       (0.00, 0.00, 0.00),
    "colour_shift":   (0.00, 0.05, 0.15),   # free, victim barely notices
    "blur":           (0.00, 0.05, 0.25),
    "jpeg":           (0.00, 0.05, 0.10),
    "brightness":     (0.00, 0.05, 0.15),
    "layout_shift":   (0.00, 0.10, 0.20),
    "grayscale":      (0.00, 0.05, 0.35),   # visibly off-brand
    "logo_occlude":   (0.00, 0.05, 0.85),   # victim loses the brand cue
    "logo_delete":    (0.00, 0.05, 0.95),   # near self-defeating
}
W = np.array([0.45, 0.20, 0.35])


def attack_cost(name: str) -> float:
    return float(np.array(ATTACK_COSTS[name]) @ W)


def sampling_probs(tau: float = 0.12, names: list[str] | None = None) -> np.ndarray:
    """P(a) ∝ exp(-cost/tau): cheap attacks dominate the training distribution."""
    names = names or list(ATTACK_COSTS)
    c = np.array([attack_cost(n) for n in names])
    p = np.exp(-c / tau)
    return p / p.sum()


def apply_attack(x: torch.Tensor, name: str, rng: np.random.Generator) -> torch.Tensor:
    """x: (C,H,W) in [0,1]. Returns the manipulated image."""
    if name == "identity":
        return x
    if name == "colour_shift":
        g = torch.tensor(rng.uniform(0.6, 1.4, size=(3, 1, 1)), dtype=x.dtype)
        return (x * g).clamp(0, 1)
    if name == "brightness":
        return (x * float(rng.uniform(0.55, 1.45))).clamp(0, 1)
    if name == "grayscale":
        g = x.mean(0, keepdim=True)
        return g.repeat(3, 1, 1)
    if name == "blur":
        k = torch.ones(3, 1, 5, 5) / 25.0
        return F.conv2d(x[None], k, padding=2, groups=3)[0].clamp(0, 1)
    if name == "jpeg":            # quantisation proxy for compression artefacts
        q = float(rng.choice([8, 12, 16]))
        return (torch.round(x * q) / q).clamp(0, 1)
    if name == "layout_shift":
        dx, dy = int(rng.integers(-18, 19)), int(rng.integers(-18, 19))
        return torch.roll(x, shifts=(dy, dx), dims=(1, 2))
    H = x.shape[1]
    if name == "logo_occlude":    # cover the wordmark band (top of the page)
        y = x.clone(); y[:, : H // 6, :] = 1.0
        return y
    if name == "logo_delete":     # remove the entire header region
        y = x.clone(); y[:, : H // 4, :] = 1.0
        return y
    raise ValueError(name)


def attack_batch(X: torch.Tensor, names: list[str], rng) -> torch.Tensor:
    return torch.stack([apply_attack(X[i], names[i], rng) for i in range(len(X))])
