"""Phase 2 on REAL screenshots: does redundancy emerge when brand identity is
genuinely distributed across logo, typography and layout?

The synthetic run was invalid: that corpus encoded brand identity almost entirely
in colour, so the 'cheap' attacks we trained against destroyed 100% of the signal
and the encoder collapsed to chance. Here each brand is represented by several
REAL captured pages of different types (login, checkout, dashboard, payment), so
brand identity is multiply encoded and there is something to redistribute onto.

Same two competing objectives:
  invariance   a cheaply degraded page embeds near its brand's anchor
  separation   distinct brands stay far apart (the analogue of clean accuracy)
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from phishnet.config import Config, set_seed
from phishnet.models.visual_siamese import VisualEncoder
from phishnet.train.train_visual import load_image
from phishnet.eval.visual_attacks import ATTACK_COSTS, sampling_probs, apply_attack

DEV = "cpu"
ROOT = Path("data/real_shots")
ATTACKS = list(ATTACK_COSTS)
CHEAP = ["colour_shift", "brightness", "jpeg", "blur", "grayscale", "layout_shift"]
EXPENSIVE = ["logo_occlude", "logo_delete"]


def load_corpus(min_n=3):
    rows = list(csv.DictReader((ROOT / "manifest.csv").open()))
    by = defaultdict(list)
    for r in rows:
        p = ROOT / r["file_name"]
        if p.exists():
            by[r["brand"]].append(r["file_name"])
    return {b: v for b, v in by.items() if len(v) >= min_n}


class Cache:
    def __init__(self, size): self.size, self.d = size, {}
    def __call__(self, p):
        if p not in self.d:
            self.d[p] = load_image(ROOT / p, self.size)
        return self.d[p]


def train_encoder(cfg, by, brands, cache, defended, seed=0, epochs=8, tau=0.12, margin=0.3):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    enc = VisualEncoder(embed_dim=cfg.visual.embed_dim, out_dim=cfg.fusion.d_model).to(DEV)
    opt = torch.optim.AdamW(enc.parameters(), lr=2e-4, weight_decay=1e-4)
    probs = sampling_probs(tau)
    bl = list(brands)
    for _ in range(epochs):
        random.shuffle(bl)
        for b in bl:
            pos_pool = by[b]
            if len(pos_pool) < 2:
                continue
            a_p, p_p = random.sample(pos_pool, 2)
            nb = random.choice([x for x in bl if x != b])
            n_p = random.choice(by[nb])
            a, p, n = cache(a_p), cache(p_p), cache(n_p)
            if defended:
                p = apply_attack(p, str(rng.choice(ATTACKS, p=probs)), rng)
            opt.zero_grad(set_to_none=True)
            z = F.normalize(enc(torch.stack([a, p, n])), dim=1)
            loss = F.relu((1 - (z[0]*z[1]).sum()) - (1 - (z[0]*z[2]).sum()) + margin)
            if loss.requires_grad:
                loss.backward(); opt.step()
    return enc.eval()


@torch.no_grad()
def evaluate(enc, by, brands, cache, rng, n_probe=2, bs=32):
    """Top-1 brand retrieval against the anchor index, clean and per attack."""
    anchors = torch.stack([cache(by[b][0]) for b in brands])
    A = F.normalize(enc(anchors), dim=1) if len(anchors) <= bs else torch.cat(
        [F.normalize(enc(anchors[i:i+bs]), dim=1) for i in range(0, len(anchors), bs)])
    inter = (1 - A @ A.T)
    margin = float(inter[~torch.eye(len(brands), dtype=bool)].mean())

    probes, labels = [], []
    for bi, b in enumerate(brands):
        for p in by[b][1:1+n_probe]:
            probes.append(p); labels.append(bi)
    out = {"_margin": margin, "_n_probe": len(probes)}
    for atk in ATTACKS:
        hits = 0
        for i in range(0, len(probes), bs):
            chunk = [apply_attack(cache(p), atk, rng) for p in probes[i:i+bs]]
            Z = F.normalize(enc(torch.stack(chunk)), dim=1)
            pred = (Z @ A.T).argmax(1).numpy()
            hits += int((pred == np.array(labels[i:i+bs])).sum())
        out[atk] = hits / max(len(probes), 1)
    return out


def main() -> int:
    cfg = Config()
    by = load_corpus()
    brands = sorted(by)
    cache = Cache(cfg.visual.image_size)
    rng = np.random.default_rng(0)
    random.Random(1337).shuffle(brands)
    holdout, seen = brands[:100], brands[100:]
    print(f"real corpus: {sum(len(v) for v in by.values())} images, {len(brands)} brands")
    print(f"  seen={len(seen)}  zero-shot holdout={len(holdout)}  "
          f"(chance = {1/len(seen):.3f} / {1/len(holdout):.3f})\n")

    res = {}
    for defended in (False, True):
        tag = "cost_weighted_aug" if defended else "baseline"
        S, Z = [], []
        for s in (0, 1):
            enc = train_encoder(cfg, by, seen, cache, defended, seed=s)
            S.append(evaluate(enc, by, seen, cache, rng))
            Z.append(evaluate(enc, by, holdout, cache, rng))
        def avg(rs, k): return float(np.mean([r[k] for r in rs]))
        res[tag] = {"margin": avg(S, "_margin"),
                    "seen": {k: avg(S, k) for k in S[0] if not k.startswith("_")},
                    "zeroshot": {k: avg(Z, k) for k in Z[0] if not k.startswith("_")}}
        r = res[tag]
        print(f"[{tag}]  inter-brand margin={r['margin']:.4f}")
        for split in ("seen", "zeroshot"):
            d = r[split]
            print(f"   {split:9s} clean={d['identity']:.3f}  "
                  f"cheap={np.mean([d[a] for a in CHEAP]):.3f}  "
                  f"expensive={np.mean([d[a] for a in EXPENSIVE]):.3f}")
        print()

    b, d = res["baseline"], res["cost_weighted_aug"]
    print(f"{'metric':38s} {'baseline':>9s} {'defended':>9s} {'delta':>9s}")
    print("-" * 68)
    rows = [("clean retrieval (seen)", b["seen"]["identity"], d["seen"]["identity"]),
            ("cheap-attack retrieval (seen)",
             np.mean([b["seen"][a] for a in CHEAP]), np.mean([d["seen"][a] for a in CHEAP])),
            ("clean retrieval (zero-shot)", b["zeroshot"]["identity"], d["zeroshot"]["identity"]),
            ("cheap-attack retrieval (zero-shot)",
             np.mean([b["zeroshot"][a] for a in CHEAP]), np.mean([d["zeroshot"][a] for a in CHEAP])),
            ("inter-brand margin (collapse check)", b["margin"], d["margin"])]
    for n, bv, dv in rows:
        print(f"{n:38s} {bv:9.4f} {dv:9.4f} {dv-bv:+9.4f}")

    print(f"\nper-attack (seen):  {'baseline':>9s} {'defended':>9s}")
    for a in ATTACKS:
        print(f"  {a:16s} {b['seen'][a]:9.3f} {d['seen'][a]:9.3f}")
    Path("results/real_crucible.json").write_text(json.dumps(res, indent=2))
    print("\nsaved -> results/real_crucible.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
