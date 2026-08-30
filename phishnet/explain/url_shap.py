"""SHAP explanations for the URL branch.

Two levels, because two audiences need different things:

  feature level   SHAP over the hand-engineered lexical features (num_hyphens,
                  brand_outside_domain, ...). This is the human-facing
                  explanation: "flagged because a brand name appears outside the
                  registered domain and the TLD is high-risk". Computed against
                  the RandomForest, whose features are exactly these names.

  character level occlusion saliency over the raw string for the CNN, so the UI
                  can underline the specific substring ("paypa1-") that drove the
                  verdict. Gradient attribution on an embedding lookup is noisy;
                  occlusion is slower but faithful and needs no backward pass
                  through a non-differentiable table.
"""
from __future__ import annotations

import numpy as np
import torch

from phishnet.features.lexical import FEATURE_NAMES, extract
from phishnet.features.url_encoding import encode_url, PAD, UNK, STOI


def feature_shap(rf_model, scaler, url: str, background=None, top_k: int = 6):
    """Per-feature SHAP values for one URL, using the trained RandomForest."""
    import shap

    x = extract(url).reshape(1, -1)
    explainer = shap.TreeExplainer(rf_model)
    sv = explainer.shap_values(x)
    # Binary RF returns a list [neg, pos]; take the positive-class attribution.
    vals = sv[1][0] if isinstance(sv, list) else (sv[0, :, 1] if sv.ndim == 3 else sv[0])
    order = np.argsort(-np.abs(vals))[:top_k]
    return [{"feature": FEATURE_NAMES[i], "value": float(x[0, i]),
             "shap": float(vals[i]),
             "direction": "phishing" if vals[i] > 0 else "benign"}
            for i in order]


@torch.no_grad()
def char_saliency(model, url: str, device, max_len: int = 200) -> list[dict]:
    """Occlusion saliency: how much the phishing score drops when each character
    is masked. Positive == that character pushed the verdict toward phishing."""
    model.eval()
    base_ids = encode_url(url, max_len)
    n = min(len(url), max_len)
    x = torch.from_numpy(base_ids).unsqueeze(0).to(device)
    base = torch.sigmoid(model(x)).item()

    # Batch the occlusions: one row per position, that position set to <UNK>.
    variants = np.tile(base_ids, (n, 1))
    for i in range(n):
        variants[i, i] = UNK
    xv = torch.from_numpy(variants).to(device)
    scores = torch.sigmoid(model(xv)).cpu().numpy()
    drops = base - scores            # positive => char was evidence for phishing

    u = url[-max_len:] if len(url) > max_len else url
    return [{"pos": i, "char": u[i] if i < len(u) else "",
             "saliency": float(drops[i])} for i in range(n)]


def top_spans(saliency: list[dict], k: int = 3, window: int = 6) -> list[dict]:
    """Collapse per-character saliency into the k most suspicious substrings,
    which is what a user can actually read."""
    if not saliency:
        return []
    vals = np.array([s["saliency"] for s in saliency])
    chars = "".join(s["char"] for s in saliency)
    scores = np.convolve(vals, np.ones(window), mode="same")
    spans, used = [], set()
    for idx in np.argsort(-scores):
        if scores[idx] <= 0 or idx in used:
            continue
        a, b = max(0, idx - window // 2), min(len(chars), idx + window // 2 + 1)
        if any(p in used for p in range(a, b)):
            continue
        used.update(range(a, b))
        spans.append({"span": chars[a:b], "start": int(a),
                      "weight": float(scores[idx])})
        if len(spans) >= k:
            break
    return spans
