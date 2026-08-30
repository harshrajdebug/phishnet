"""Baseline encoder training on Phishpedia logo crops, judged by the gates.

The question this run answers is narrow and was set before it started: does
*standard* metric learning preserve the robustness that untrained ImageNet
features already have? On downscaled full pages it did not -- triplet training
doubled clean retrieval while degrading every attack, which is the failure the
competence gate exists to catch. Now that the corpus is high-resolution,
deduplicated logo crops that pass all three gates, we re-ask it cleanly. If
standard triplet loss still destroys robustness, the architecture is
texture-hypersensitive and we learn that before any cost-weighted defence is
layered on top and credited with the difference.

Split is family-disjoint three ways. The encoder must not train on the gallery it
is later evaluated against, or "retrieval" partly measures memorised enrolment
images rather than generalisation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .competence_gate import (competence_gate, corpus_validity_gate, skill,
                              split_leakage_gate)
from .phishpedia_crucible import (CROP_ATTACKS, DEV, RESULTS, apply_crop_attack,
                                  build_corpus, embed, top1_retrieval,
                                  untrained_control)

CHEAP = tuple(a for a in CROP_ATTACKS
              if a not in ("identity", "logo_occlude", "logo_delete"))


def make_splits3(by_brand: dict, seed: int = 0, zs_frac: float = 0.25):
    """Seen brands -> train / gallery / query by family (50/25/25).

    Zero-shot brands contribute no training data; their families split 50/50 into
    gallery and query so an unseen brand is enrolled from reference images alone.
    """
    rng = np.random.default_rng(seed)
    brands = sorted(by_brand)
    rng.shuffle(brands)
    n_zs = max(1, int(len(brands) * zs_frac))
    zs, seen = sorted(brands[:n_zs]), sorted(brands[n_zs:])

    tr_x, tr_y = [], []
    g, gy, q, qy = [], [], [], []
    for b in seen:
        items = by_brand[b]
        fams = sorted({f for _, f, _ in items})
        rng.shuffle(fams)
        n_tr = max(1, int(len(fams) * 0.5))
        n_g = max(1, int(len(fams) * 0.25))
        tr_f = set(fams[:n_tr])
        g_f = set(fams[n_tr:n_tr + n_g])
        for t, f, _ in items:
            if f in tr_f:
                tr_x.append(t); tr_y.append(b)
            elif f in g_f:
                g.append(t); gy.append(b)
            else:
                q.append(t); qy.append(b)

    zg, zgy, zq, zqy = [], [], [], []
    for b in zs:
        items = by_brand[b]
        fams = sorted({f for _, f, _ in items})
        rng.shuffle(fams)
        gset = set(fams[:max(1, len(fams) // 2)])
        for t, f, _ in items:
            if f in gset:
                zg.append(t); zgy.append(b)
            else:
                zq.append(t); zqy.append(b)

    seen_split = (torch.stack(g), np.array(gy), torch.stack(q), np.array(qy))
    zs_split = (torch.stack(zg), np.array(zgy), torch.stack(zq), np.array(zqy))
    return (torch.stack(tr_x), np.array(tr_y)), seen_split, zs_split, seen, zs


def make_encoder(dim: int = 128):
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Linear(512, dim)
    return m.to(DEV)


def batch_hard_triplet(emb: torch.Tensor, y: np.ndarray) -> torch.Tensor:
    """Hermans et al. batch-hard: hardest positive and hardest negative per anchor.

    Random triplets go slack almost immediately, which is why the mining pool size
    matters; this is the reason the corpus was grown before training started.
    """
    d = torch.cdist(emb, emb, p=2)
    uniq = {v: i for i, v in enumerate(dict.fromkeys(y))}
    lab = torch.as_tensor([uniq[v] for v in y], device=emb.device)
    same = lab[:, None] == lab[None, :]
    eye = torch.eye(len(y), dtype=torch.bool, device=emb.device)
    pos = same & ~eye
    neg = ~same
    valid = pos.any(1) & neg.any(1)
    if not valid.any():
        return emb.sum() * 0.0
    hardest_pos = d.masked_fill(~pos, -float("inf")).max(1).values
    hardest_neg = d.masked_fill(~neg, float("inf")).min(1).values
    return F.softplus(hardest_pos - hardest_neg)[valid].mean()


def pk_batches(y: np.ndarray, P: int, K: int, steps: int, rng):
    by = defaultdict(list)
    for i, b in enumerate(y):
        by[b].append(i)
    usable = [b for b, v in by.items() if len(v) >= K]
    for _ in range(steps):
        bs = rng.choice(usable, size=min(P, len(usable)), replace=False)
        idx = []
        for b in bs:
            idx += list(rng.choice(by[b], size=K, replace=False))
        yield np.array(idx)


def evaluate(model, split, rng) -> dict:
    gal, gal_y, qry, qry_y = split
    gal_e = embed(model, gal)
    out = {}
    for a in CROP_ATTACKS:
        Xa = torch.stack([apply_crop_attack(qry[i], a, rng) for i in range(len(qry))])
        out[a] = top1_retrieval(gal_e, gal_y, embed(model, Xa), qry_y)
    return out


def summarise(per: dict) -> tuple[float, float]:
    return per["identity"], float(np.mean([per[a] for a in CHEAP]))


def fingerprints(X):
    return [hashlib.md5(np.round(X[i].numpy(), 3).tobytes()).hexdigest()
            for i in range(len(X))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps-per-epoch", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    by_brand = build_corpus(size=128)
    (tr_x, tr_y), seen_split, zs_split, seen, zs = make_splits3(by_brand, seed=args.seed)
    chance = 1.0 / len(seen)
    print(f"train={len(tr_x)} crops over {len(set(tr_y))} brands", flush=True)
    print(f"seen: gallery={len(seen_split[0])} query={len(seen_split[2])} "
          f"(chance={chance:.4f})", flush=True)
    print(f"zero-shot: {len(zs)} brands, gallery={len(zs_split[0])} "
          f"query={len(zs_split[2])}", flush=True)

    # The encoder must not have trained on what it retrieves against.
    print("\nsplit leakage audit:", flush=True)
    for nm, a, b in (("train vs gallery", tr_x, seen_split[0]),
                     ("train vs query", tr_x, seen_split[2]),
                     ("gallery vs query", seen_split[0], seen_split[2]),
                     ("zs gallery vs query", zs_split[0], zs_split[2])):
        r = split_leakage_gate(fingerprints(a), fingerprints(b))
        print(f"  {nm:<22} -> {'PASS' if r.passed else 'FAIL'} "
              f"({r.detail['dup_frac']:.1%})", flush=True)
        if not r.passed:
            print("Refusing to train on a leaking split."); raise SystemExit(1)

    print("\n--- untrained ImageNet control ---", flush=True)
    ctrl = untrained_control()
    c_seen = evaluate(ctrl, seen_split, rng)
    c_zs = evaluate(ctrl, zs_split, rng)
    cc, ca = summarise(c_seen)
    zc, za = summarise(c_zs)
    print(f"  seen      clean={cc:.3f}  cheap={ca:.3f}", flush=True)
    print(f"  zero-shot clean={zc:.3f}  cheap={za:.3f}", flush=True)
    cv = corpus_validity_gate(c_seen)
    print(f"  {cv}", flush=True)

    print("\n--- training baseline (plain batch-hard triplet, no augmentation) ---",
          flush=True)
    model = make_encoder(args.dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for idx in pk_batches(tr_y, args.P, args.K, args.steps_per_epoch, rng):
            xb = tr_x[idx].to(DEV)
            emb = F.normalize(model(xb), dim=1)
            loss = batch_hard_triplet(emb, tr_y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        msg = (f"epoch {ep:3d}/{args.epochs}  loss={np.mean(losses):.4f}  "
               f"({time.time()-t0:.0f}s)")
        if ep % args.eval_every == 0 or ep == args.epochs:
            per = evaluate(model, seen_split, rng)
            cl, at = summarise(per)
            msg += f"  clean={cl:.3f} cheap={at:.3f}"
            hist.append({"epoch": ep, "loss": float(np.mean(losses)),
                         "clean": cl, "cheap": at})
        print(msg, flush=True)

    print("\n--- trained encoder ---", flush=True)
    t_seen = evaluate(model, seen_split, rng)
    t_zs = evaluate(model, zs_split, rng)
    tc, ta = summarise(t_seen)
    tzc, tza = summarise(t_zs)

    print(f"{'attack':<16}{'control':>10}{'trained':>10}{'delta':>10}", flush=True)
    for a in CROP_ATTACKS:
        print(f"  {a:<14}{c_seen[a]:>10.3f}{t_seen[a]:>10.3f}"
              f"{t_seen[a]-c_seen[a]:>+10.3f}", flush=True)
    print(f"\n  seen      clean={tc:.3f} (ctrl {cc:.3f})  "
          f"cheap={ta:.3f} (ctrl {ca:.3f})", flush=True)
    print(f"  zero-shot clean={tzc:.3f} (ctrl {zc:.3f})  "
          f"cheap={tza:.3f} (ctrl {za:.3f})", flush=True)

    gate = competence_gate(clean=tc, attacked=ta, chance=chance,
                           control_clean=cc, control_attacked=ca)
    print(f"\n{gate}", flush=True)

    out = {"chance": chance, "n_seen": len(seen), "n_zeroshot": len(zs),
           "n_train": len(tr_x), "control_seen": c_seen, "control_zs": c_zs,
           "trained_seen": t_seen, "trained_zs": t_zs, "history": hist,
           "control_clean": cc, "control_cheap": ca,
           "trained_clean": tc, "trained_cheap": ta,
           "trained_clean_skill": skill(tc, chance),
           "trained_cheap_skill": skill(ta, chance),
           "competence_passed": gate.passed, "competence_reasons": gate.reasons,
           "corpus_validity_passed": cv.passed,
           "args": vars(args)}
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "phishpedia_baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    torch.save(model.state_dict(), RESULTS / "phishpedia_baseline.pt")
    print(f"\nsaved -> {RESULTS/'phishpedia_baseline.json'}", flush=True)


if __name__ == "__main__":
    main()
