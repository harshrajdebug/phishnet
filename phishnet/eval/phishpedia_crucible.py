"""Visual crucible on Phishpedia logo crops, with validity gates in front.

Why this exists. Two earlier visual experiments were invalid. The synthetic
corpus encoded brand identity in colour; the real-screenshot corpus downscaled a
1366px page to 224px, which shrinks a ~200x100 wordmark to roughly 23x21 px and
left the baseline at 1-4% against a 0.4% chance floor. Neither could host a
defence experiment, and we only discovered that after spending the compute.

Two changes here. First, we crop to the detected logo region instead of
downscaling the whole page, so the brand mark fills the frame. Second, nothing
trains until the competence and corpus-validity gates pass on an *untrained*
control -- the probe is one forward pass and it is the whole point of running it
first.

Splits are family-disjoint. family_id is a phishing-kit hash, so pages sharing
one are near-duplicates; a random split would let the encoder memorise the kit
and report it as brand recognition.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .competence_gate import (competence_gate, corpus_validity_gate, skill,
                              split_leakage_gate)

DATA = Path("data/phishpedia")
SHOTS = DATA / "shots"
RESULTS = Path("results")
DEV = "cpu"          # deliberate: MPS produced wrong gradients here before

# Attacks re-defined for crops. On a full page "logo_occlude" masked the header
# band; on a crop the mark *is* the image, so a top-band mask means something
# else entirely. We mask central regions and keep the surrounding background,
# which keeps the corpus-validity gate meaningful: if identity survives removal
# of the mark, it is living in the background colour.
CROP_ATTACKS = ("identity", "colour_shift", "blur", "jpeg", "brightness",
                "layout_shift", "grayscale", "logo_occlude", "logo_delete")


def apply_crop_attack(x: torch.Tensor, name: str, rng: np.random.Generator) -> torch.Tensor:
    """x: (3,H,W) in [0,1]."""
    if name == "identity":
        return x
    if name == "colour_shift":
        g = torch.tensor(rng.uniform(0.6, 1.4, size=(3, 1, 1)), dtype=x.dtype)
        return (x * g).clamp(0, 1)
    if name == "brightness":
        return (x * float(rng.uniform(0.55, 1.45))).clamp(0, 1)
    if name == "grayscale":
        return x.mean(0, keepdim=True).repeat(3, 1, 1)
    if name == "blur":
        k = torch.ones(3, 1, 5, 5) / 25.0
        return F.conv2d(x[None], k, padding=2, groups=3)[0].clamp(0, 1)
    if name == "jpeg":
        q = float(rng.choice([8, 12, 16]))
        return (torch.round(x * q) / q).clamp(0, 1)
    if name == "layout_shift":
        dx, dy = int(rng.integers(-18, 19)), int(rng.integers(-18, 19))
        return torch.roll(x, shifts=(dy, dx), dims=(1, 2))

    H, W = x.shape[1], x.shape[2]
    if name == "logo_occlude":       # blank the central ~36% of the mark
        y = x.clone()
        y[:, int(0.32 * H):int(0.68 * H), int(0.32 * W):int(0.68 * W)] = 1.0
        return y
    if name == "logo_delete":        # blank the central 64%, keep the surround
        y = x.clone()
        y[:, int(0.18 * H):int(0.82 * H), int(0.18 * W):int(0.82 * W)] = 1.0
        return y
    raise ValueError(name)


# ---------------------------------------------------------------- corpus ----
def load_manifest() -> list[dict]:
    with open(DATA / "manifest.csv") as f:
        return list(csv.DictReader(f))


def crop_logo(path: Path, boxes: list, size: int = 128,
              pad: float = 0.12) -> torch.Tensor | None:
    """Top-confidence detected region, padded, as a (3,size,size) tensor.

    The boxes come from Phishpedia's RCNN and are logo *candidates*, not verified
    ground truth, so the top-1 box is a noisy proxy for the brand mark. We keep
    the aspect ratio by padding to a square rather than stretching -- the earlier
    pipeline's non-aspect-preserving resize is what turned wordmarks into smears.
    """
    if not boxes:
        return None
    x1, y1, x2, y2, _ = boxes[0]
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    W, H = im.size
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(bw, bh) * (0.5 + pad)          # square window, aspect preserved
    l, t = max(0, cx - half), max(0, cy - half)
    r, b = min(W, cx + half), min(H, cy + half)
    if r - l < 8 or b - t < 8:
        return None
    crop = im.crop((int(l), int(t), int(r), int(b))).resize((size, size), Image.BILINEAR)
    a = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255.0)
    return a.permute(2, 0, 1).contiguous()


def build_corpus(size: int = 128, min_per_brand: int = 10, limit: int | None = None,
                 dedup: bool = True):
    """Build the brand -> [(crop, family_id, file)] corpus.

    dedup removes crops that are *pixel-identical* to one already kept for that
    brand. This is not the same control as family_id and it is not optional:
    distinct phishing kits clone the same official logo, so different family_ids
    routinely yield byte-identical crops. Measured on this corpus, 37.3% of query
    crops had an exact twin in the gallery, which turns retrieval into a lookup of
    an identical image and inflates every score built on it.
    """
    rows = load_manifest()
    if limit:
        rows = rows[:limit]
    by_brand: dict[str, list] = defaultdict(list)
    seen_hash: dict[str, set] = defaultdict(set)
    n_nobox = n_fail = n_dup = 0
    for r in rows:
        boxes = json.loads(r["boxes"]) if r["boxes"] else []
        if not boxes:
            n_nobox += 1
            continue
        t = crop_logo(SHOTS / r["file"], boxes, size=size)
        if t is None:
            n_fail += 1
            continue
        if dedup:
            h = hashlib.md5(np.round(t.numpy(), 3).tobytes()).hexdigest()
            if h in seen_hash[r["brand"]]:
                n_dup += 1
                continue
            seen_hash[r["brand"]].add(h)
        by_brand[r["brand"]].append((t, r["family_id"], r["file"]))
    by_brand = {b: v for b, v in by_brand.items() if len(v) >= min_per_brand}
    print(f"corpus: {sum(len(v) for v in by_brand.values())} crops, "
          f"{len(by_brand)} brands  (no-box {n_nobox}, crop-fail {n_fail}, "
          f"exact-dup dropped {n_dup})", flush=True)
    return by_brand


def embed(model, X: torch.Tensor, bs: int = 128) -> torch.Tensor:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(F.normalize(model(X[i:i + bs].to(DEV)), dim=1).cpu())
    return torch.cat(out)


def top1_retrieval(gal: torch.Tensor, gal_y: np.ndarray,
                   qry: torch.Tensor, qry_y: np.ndarray) -> float:
    sim = qry @ gal.T
    pred = gal_y[sim.argmax(1).numpy()]
    return float((pred == qry_y).mean())


def make_splits(by_brand: dict, seed: int = 0, zs_frac: float = 0.25):
    """Brand-disjoint zero-shot holdout; family-disjoint gallery/query within."""
    rng = np.random.default_rng(seed)
    brands = sorted(by_brand)
    rng.shuffle(brands)
    n_zs = max(1, int(len(brands) * zs_frac))
    zs, seen = sorted(brands[:n_zs]), sorted(brands[n_zs:])

    def split(bl):
        g, gy, q, qy = [], [], [], []
        for b in bl:
            items = by_brand[b]
            fams = sorted({f for _, f, _ in items})
            rng.shuffle(fams)
            n_g = max(1, len(fams) // 2)
            gset = set(fams[:n_g])
            for t, f, _ in items:
                (g if f in gset else q).append(t)
                (gy if f in gset else qy).append(b)
        if not q:
            return None
        return (torch.stack(g), np.array(gy), torch.stack(q), np.array(qy))

    return split(seen), split(zs), seen, zs


# ------------------------------------------------------------------ probe ----
def untrained_control(size: int = 128):
    """Raw ImageNet features. This is the gate's reference, and it costs one pass."""
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    return m.to(DEV).eval()


def evaluate(model, split, rng, attacks=CROP_ATTACKS) -> dict:
    gal, gal_y, qry, qry_y = split
    gal_e = embed(model, gal)
    per = {}
    for a in attacks:
        Xa = torch.stack([apply_crop_attack(qry[i], a, rng) for i in range(len(qry))])
        per[a] = top1_retrieval(gal_e, gal_y, embed(model, Xa), qry_y)
    return per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--probe-only", action="store_true",
                    help="run the untrained control and the gates, then stop")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    by_brand = build_corpus(size=args.size, limit=args.limit)
    seen_split, zs_split, seen, zs = make_splits(by_brand, seed=args.seed)
    if seen_split is None:
        print("FATAL: no usable seen split"); sys.exit(1)

    n_cls = len(seen)
    chance = 1.0 / n_cls
    print(f"seen brands={n_cls} (chance={chance:.4f})  zero-shot brands={len(zs)}",
          flush=True)
    print(f"gallery={len(seen_split[0])} query={len(seen_split[2])}", flush=True)

    # Audit the split itself before believing any number computed on it. The
    # family_id control is sound and still let 37.3% of query crops through as
    # pixel-identical twins of gallery crops, so we fingerprint the tensors that
    # actually reach the model.
    def fp(X):
        return [hashlib.md5(np.round(X[i].numpy(), 3).tobytes()).hexdigest()
                for i in range(len(X))]

    leak = split_leakage_gate(fp(seen_split[0]), fp(seen_split[2]))
    print(f"\nseen split      -> {leak}", flush=True)
    gates = [leak]
    # The zero-shot split matters more, not less: duplicate crops there inflate
    # exactly the generalisation-to-unseen-brands claim the split exists to make.
    if zs_split is not None:
        zleak = split_leakage_gate(fp(zs_split[0]), fp(zs_split[2]))
        print(f"zero-shot split -> {zleak}", flush=True)
        gates.append(zleak)
    if not all(g.passed for g in gates):
        print("\nRefusing to report retrieval on a leaking split.", flush=True)
        sys.exit(1)

    print("\n--- untrained ImageNet control (the gate reference) ---", flush=True)
    ctrl = untrained_control(args.size)
    per_ctrl = evaluate(ctrl, seen_split, rng)
    for a in CROP_ATTACKS:
        print(f"   {a:<14} {per_ctrl[a]:.3f}", flush=True)

    cheap = [a for a in CROP_ATTACKS if a not in ("identity", "logo_occlude", "logo_delete")]
    ctrl_clean = per_ctrl["identity"]
    ctrl_att = float(np.mean([per_ctrl[a] for a in cheap]))
    print(f"\ncontrol: clean={ctrl_clean:.3f}  cheap-attack mean={ctrl_att:.3f}", flush=True)

    cv = corpus_validity_gate(per_ctrl)
    print(f"\nCORPUS VALIDITY (on untrained control): {cv}", flush=True)

    print(f"\ncontrol skill: clean={skill(ctrl_clean, chance):.3f}  "
          f"cheap-attack={skill(ctrl_att, chance):.3f}", flush=True)

    out = {"n_seen": n_cls, "n_zeroshot": len(zs), "chance": chance,
           "n_gallery": len(seen_split[0]), "n_query": len(seen_split[2]),
           "control_per_attack": per_ctrl, "control_clean": ctrl_clean,
           "control_cheap_mean": ctrl_att,
           "control_clean_skill": skill(ctrl_clean, chance),
           "control_cheap_skill": skill(ctrl_att, chance),
           "split_leakage_passed": all(g.passed for g in gates),
           "split_leakage_seen": leak.detail,
           "split_leakage_zeroshot": (gates[1].detail if len(gates) > 1 else None),
           "corpus_validity_passed": cv.passed,
           "corpus_validity_reasons": cv.reasons}
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "phishpedia_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {RESULTS/'phishpedia_probe.json'}", flush=True)

    if args.probe_only:
        print("\n--probe-only set; stopping before any training.", flush=True)
        return


if __name__ == "__main__":
    main()
