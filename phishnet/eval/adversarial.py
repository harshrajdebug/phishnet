"""Adversarial robustness of the URL branch.

Phishing is adversarial: a real attacker will not hand the detector a clean
malicious URL, they will perturb it to slip past. A phishing detector that is
never tested under perturbation is reporting its best case, not its operating
case. We take URLs the model already flags as phishing and apply evasion
transforms that a kit can apply for free, then measure how much detection drops.
Reporting where the model holds and where it breaks is the honest thing to do,
and it points at exactly what a v2 must harden.

Transforms (each a real, documented evasion technique):
  https_upgrade       serve over TLS so "is_https" flips (certs are free now)
  benign_padding      pad the path with legitimate-looking brand tokens
  hyphen_brand        move the brand into a hyphenated subdomain of a random host
  homoglyph           swap ASCII letters for confusable look-alikes (IDN attack)
  shortener_wrap      hide the URL behind a bit.ly-style short host
  tld_swap            move to a cheaper but less flagged TLD
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from phishnet.config import Config, resolve_device
from phishnet.features.url_encoding import encode_url
from phishnet.models.url_cnn import URLClassifier, URLEncoder

HOMOGLYPH = str.maketrans({"a": "а", "e": "е", "o": "о",
                           "p": "р", "c": "с", "i": "і"})
BENIGN_TOKENS = ["login", "account", "secure", "signin", "auth", "verify",
                 "www", "com", "portal", "home"]
CHEAP_TLDS = ["xyz", "top", "cfd", "sbs", "icu", "click", "rest"]


def https_upgrade(u): return re.sub(r"^http://", "https://", u) if u.startswith("http://") else u


def benign_padding(u, rng):
    m = re.match(r"^(https?://[^/]+)(/.*)?$", u)
    if not m:
        return u
    pad = "/".join(rng.sample(BENIGN_TOKENS, 3))
    return f"{m.group(1)}/{pad}{m.group(2) or ''}"


def hyphen_brand(u, rng):
    m = re.match(r"^(https?://)([^/]+)(/.*)?$", u)
    if not m:
        return u
    host = m.group(2)
    brand = rng.choice(["paypal", "microsoft", "apple", "amazon"])
    return f"{m.group(1)}{brand}-{rng.choice(BENIGN_TOKENS)}.{host}{m.group(3) or ''}"


def homoglyph(u):
    m = re.match(r"^(https?://)([^/]+)(/.*)?$", u)
    if not m:
        return u
    return f"{m.group(1)}{m.group(2).translate(HOMOGLYPH)}{m.group(3) or ''}"


def shortener_wrap(u, rng):
    return f"https://{rng.choice(['bit', 'tinyurl', 'cutt'])}.ly/{rng.randint(100000, 999999)}"


def tld_swap(u, rng):
    m = re.match(r"^(https?://[^/]+?\.)([a-z]+)(/.*|$)", u)
    if not m:
        return u
    return f"{m.group(1)}{rng.choice(CHEAP_TLDS)}{m.group(3)}"


TRANSFORMS = {
    "https_upgrade": lambda u, r: https_upgrade(u),
    "benign_padding": benign_padding,
    "hyphen_brand": hyphen_brand,
    "homoglyph": lambda u, r: homoglyph(u),
    "shortener_wrap": shortener_wrap,
    "tld_swap": tld_swap,
}


@torch.no_grad()
def score(model, urls, device, cfg, thr):
    if not urls:
        return np.array([]), 0.0
    x = torch.tensor(np.stack([encode_url(u, cfg.url.max_len) for u in urls])).to(device)
    p = torch.sigmoid(model(x)).cpu().numpy()
    return p, float((p >= thr).mean())


def main() -> int:
    cfg = Config()
    device = resolve_device(cfg.train.device)
    out = Path("results")

    enc = URLEncoder(embed_dim=cfg.url.embed_dim, num_filters=cfg.url.num_filters,
                     kernel_sizes=cfg.url.kernel_sizes, dropout=cfg.url.dropout,
                     out_dim=cfg.url.out_dim)
    model = URLClassifier(enc).to(device)
    model.load_state_dict(torch.load(out / "url_cnn_best.pt", map_location=device,
                                     weights_only=False)["state_dict"])
    model.eval()
    thr = json.loads((out / "url_results.json").read_text())["url_cnn"]["threshold"]

    # Take phishing URLs the model already catches, so any drop is the attack's doing.
    raw = Path(cfg.data.raw_dir)
    rng = random.Random(1337)
    phish = [u for u in raw.joinpath("phish_urls.txt").read_text().splitlines() if u]
    rng.shuffle(phish)
    phish = phish[:4000]
    base_p, base_rate = score(model, phish, device, cfg, thr)
    caught = [u for u, p in zip(phish, base_p) if p >= thr]
    print(f"baseline detection on {len(phish)} phishing URLs: {base_rate:.3f} "
          f"({len(caught)} caught)")

    results = {"_baseline_detection": round(base_rate, 4), "_n": len(phish),
               "_threshold": round(thr, 4)}
    for name, fn in TRANSFORMS.items():
        adv = [fn(u, rng) for u in caught]
        _, rate = score(model, adv, device, cfg, thr)
        drop = base_rate * 0 + (1.0 - rate)  # fraction of previously-caught now evading
        results[name] = {"detection_after": round(rate, 4),
                         "evasion_rate": round(1 - rate, 4)}
        print(f"  {name:16s} detection {rate:.3f}  (evasion {1-rate:.1%})")

    (out / "adversarial_results.json").write_text(json.dumps(results, indent=2))
    print(f"saved -> {out/'adversarial_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
