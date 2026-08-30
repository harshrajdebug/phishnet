"""Phase 2: does the redundancy mechanism survive the jump to pixels?

Tests cost-weighted ATTACK augmentation inside the Siamese metric space, where
there is no phishing/benign decision boundary to regularise. The encoder must
satisfy two objectives that pull against each other:

  invariance   a cheaply degraded brand page embeds near its pristine anchor
  separation   distinct brands stay far apart  <- the analogue of clean accuracy

A defence that buys invariance by collapsing the embedding space has failed, so
both are measured. Brands are held out entirely for the zero-shot condition.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from phishnet.config import Config, set_seed
from phishnet.models.visual_siamese import VisualEncoder
from phishnet.train.train_visual import load_image
from phishnet.eval.visual_attacks import ATTACK_COSTS, sampling_probs, apply_attack

DEV = "cpu"
ATTACKS = list(ATTACK_COSTS)


def build(cfg):
    sd = Path(cfg.data.screenshot_dir)
    man = json.loads((sd / "manifest.json").read_text())
    anchors = {b: a for b, a in man["anchors"].items() if (sd / a).exists()}
    by_brand: dict[str, list] = {}
    for s in man["samples"]:
        if s["label"] == 1 and (sd / s["path"]).exists() and s["brand"] in anchors:
            by_brand.setdefault(s["brand"], []).append(s["path"])
    brands = sorted([b for b in anchors if by_brand.get(b)])
    return sd, anchors, by_brand, brands


def load_all(sd, paths, size):
    return torch.stack([load_image(sd / p, size) for p in paths])


def train_encoder(cfg, sd, anchors, by_brand, train_brands, defended: bool,
                  seed=0, epochs=12, tau=0.12, margin=0.3):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    enc = VisualEncoder(embed_dim=cfg.visual.embed_dim, out_dim=cfg.fusion.d_model).to(DEV)
    opt = torch.optim.AdamW(enc.parameters(), lr=2e-4, weight_decay=1e-4)
    probs = sampling_probs(tau)
    size = cfg.visual.image_size
    cache = {}

    def img(p):
        if p not in cache:
            cache[p] = load_image(sd / p, size)
        return cache[p]

    for _ in range(epochs):
        random.shuffle(train_brands)
        for b in train_brands:
            negs = [x for x in train_brands if x != b]
            if not negs or not by_brand.get(b):
                continue
            a = img(anchors[b])
            p = img(random.choice(by_brand[b]))
            nb = random.choice(negs)
            n = img(anchors[nb]) if random.random() < .5 else img(random.choice(by_brand[nb]))
            if defended:
                # cost-weighted attack augmentation: the positive is degraded by
                # an attack drawn in proportion to how cheap it is
                a_name = str(rng.choice(ATTACKS, p=probs))
                p = apply_attack(p, a_name, rng)
            batch = torch.stack([a, p, n])
            opt.zero_grad(set_to_none=True)
            z = F.normalize(enc(batch), dim=1)
            d_pos = (1 - (z[0] * z[1]).sum())
            d_neg = (1 - (z[0] * z[2]).sum())
            loss = F.relu(d_pos - d_neg + margin)
            if loss.requires_grad:
                loss.backward()
                opt.step()
    return enc.eval()


@torch.no_grad()
def embed(enc, X):
    return F.normalize(enc(X), dim=1)


@torch.no_grad()
def evaluate(enc, sd, anchors, by_brand, brands, size, rng):
    """Top-1 brand retrieval, clean and under each attack, plus inter-brand margin."""
    A = embed(enc, load_all(sd, [anchors[b] for b in brands], size))
    inter = (1 - A @ A.T)
    iu = inter[~torch.eye(len(brands), dtype=bool)]
    margin = float(iu.mean())

    out = {"_inter_brand_margin": margin}
    for atk in ATTACKS:
        hits = tot = 0
        for bi, b in enumerate(brands):
            for p in by_brand[b][:4]:
                x = load_image(sd / p, size)
                x = apply_attack(x, atk, rng)
                z = embed(enc, x[None])
                pred = int(torch.argmax(z @ A.T))
                hits += int(pred == bi); tot += 1
        out[atk] = hits / max(tot, 1)
    return out


def main() -> int:
    cfg = Config()
    sd, anchors, by_brand, brands = build(cfg)
    rng = np.random.default_rng(0)
    size = cfg.visual.image_size
    holdout = sorted(brands)[::3][:8]                 # brand-disjoint zero-shot
    seen = [b for b in brands if b not in holdout]
    print(f"brands={len(brands)}  seen={len(seen)}  zero-shot holdout={holdout}\n")

    res = {}
    for defended in (False, True):
        tag = "cost_weighted_aug" if defended else "baseline"
        accs_seen, accs_zs, margins = [], [], []
        for s in (0, 1, 2):
            enc = train_encoder(cfg, sd, anchors, by_brand, list(seen), defended, seed=s)
            r_seen = evaluate(enc, sd, anchors, by_brand, seen, size, rng)
            r_zs = evaluate(enc, sd, anchors, by_brand, holdout, size, rng)
            accs_seen.append(r_seen); accs_zs.append(r_zs)
            margins.append(r_seen["_inter_brand_margin"])
        def avg(rs, k): return float(np.mean([r[k] for r in rs]))
        res[tag] = {
            "margin": float(np.mean(margins)),
            "seen": {k: avg(accs_seen, k) for k in accs_seen[0] if not k.startswith("_")},
            "zeroshot": {k: avg(accs_zs, k) for k in accs_zs[0] if not k.startswith("_")},
        }
        r = res[tag]
        cheap = ["colour_shift", "brightness", "jpeg", "blur", "grayscale", "layout_shift"]
        exp = ["logo_occlude", "logo_delete"]
        print(f"[{tag}]  inter-brand margin={r['margin']:.4f}")
        print(f"   SEEN      clean={r['seen']['identity']:.3f}   "
              f"cheap-attack mean={np.mean([r['seen'][a] for a in cheap]):.3f}   "
              f"expensive mean={np.mean([r['seen'][a] for a in exp]):.3f}")
        print(f"   ZERO-SHOT clean={r['zeroshot']['identity']:.3f}   "
              f"cheap-attack mean={np.mean([r['zeroshot'][a] for a in cheap]):.3f}   "
              f"expensive mean={np.mean([r['zeroshot'][a] for a in exp]):.3f}\n")

    b, d = res["baseline"], res["cost_weighted_aug"]
    cheap = ["colour_shift", "brightness", "jpeg", "blur", "grayscale", "layout_shift"]
    print(f"{'metric':34s} {'baseline':>10s} {'defended':>10s} {'delta':>9s}")
    print("-" * 68)
    rows = [("clean retrieval (seen)", b["seen"]["identity"], d["seen"]["identity"]),
            ("cheap-attack retrieval (seen)",
             np.mean([b["seen"][a] for a in cheap]), np.mean([d["seen"][a] for a in cheap])),
            ("clean retrieval (zero-shot)", b["zeroshot"]["identity"], d["zeroshot"]["identity"]),
            ("cheap-attack retrieval (zero-shot)",
             np.mean([b["zeroshot"][a] for a in cheap]), np.mean([d["zeroshot"][a] for a in cheap])),
            ("inter-brand margin (collapse check)", b["margin"], d["margin"])]
    for name, bv, dv in rows:
        print(f"{name:34s} {bv:10.4f} {dv:10.4f} {dv-bv:+9.4f}")
    Path("results/encoder_crucible.json").write_text(json.dumps(res, indent=2))
    print("\nsaved -> results/encoder_crucible.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
