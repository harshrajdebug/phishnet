"""Cost-weighted attack augmentation, judged on zero-shot transfer.

Hypothesis under test. Standard triplet loss gained +0.174 clean on seen brands
but only +0.033 on unseen ones. The proposed explanation is lazy learning: the
cheapest brand-specific feature in a logo is colour, so the encoder indexes on it,
which simultaneously explains the model's specific weakness to grayscale (0.579)
and its failure to transfer. Cost-weighted augmentation destroys cheap colour
signal during training, so satisfying the triplet loss should force capacity onto
structural invariants -- shape, aspect, spatial relation -- and those are what an
unseen brand shares with a seen one.

Predictions, registered before the run:
  1. the seen-vs-unseen clean gap narrows
  2. zero-shot attacked retrieval beats the baseline's +0.027 over control
  3. grayscale robustness improves as a by-product

Two design points the result depends on.

*Selection cannot touch the test brands.* Brands are split three ways: train,
val-zero-shot (for choosing the temperature) and test-zero-shot (read once). The
alternative -- sweeping tau and keeping whatever wins on the zero-shot set -- is
test-set tuning wearing a different hat.

*Augmentation is not the same claim as cost-weighted augmentation.* A uniform
arm samples the identical attacks with the cost weighting removed. If the
uniform arm matches the cost-weighted one, the mechanism is augmentation and the
cost model is decoration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .competence_gate import competence_gate, skill, split_leakage_gate
from .phishpedia_crucible import (CROP_ATTACKS, DEV, RESULTS, apply_crop_attack,
                                  build_corpus, embed, top1_retrieval,
                                  untrained_control)
from .phishpedia_train import (CHEAP, batch_hard_triplet, fingerprints,
                               make_encoder, pk_batches, summarise)
from .visual_attacks import attack_cost


def sampling_probs(mode: str, tau: float) -> np.ndarray:
    """P(a) over CROP_ATTACKS. Cheap attacks should dominate a cost-weighted run."""
    if mode == "uniform":
        return np.ones(len(CROP_ATTACKS)) / len(CROP_ATTACKS)
    c = np.array([attack_cost(a) for a in CROP_ATTACKS])
    p = np.exp(-c / tau)
    return p / p.sum()


def split4(by_brand: dict, seed: int = 0, val_zs: int = 14, test_zs: int = 15):
    """train / val-zero-shot / test-zero-shot brands, family-disjoint within each."""
    rng = np.random.default_rng(seed)
    brands = sorted(by_brand)
    rng.shuffle(brands)
    te = sorted(brands[:test_zs])
    va = sorted(brands[test_zs:test_zs + val_zs])
    tr = sorted(brands[test_zs + val_zs:])

    def enrol(bl):
        g, gy, q, qy = [], [], [], []
        for b in bl:
            items = by_brand[b]
            fams = sorted({f for _, f, _ in items})
            rng.shuffle(fams)
            gset = set(fams[:max(1, len(fams) // 2)])
            for t, f, _ in items:
                (g if f in gset else q).append(t)
                (gy if f in gset else qy).append(b)
        return torch.stack(g), np.array(gy), torch.stack(q), np.array(qy)

    tr_x, tr_y = [], []
    g, gy, q, qy = [], [], [], []
    for b in tr:
        items = by_brand[b]
        fams = sorted({f for _, f, _ in items})
        rng.shuffle(fams)
        n_tr = max(1, int(len(fams) * 0.5))
        n_g = max(1, int(len(fams) * 0.25))
        trf, gf = set(fams[:n_tr]), set(fams[n_tr:n_tr + n_g])
        for t, f, _ in items:
            if f in trf:
                tr_x.append(t); tr_y.append(b)
            elif f in gf:
                g.append(t); gy.append(b)
            else:
                q.append(t); qy.append(b)
    seen_split = (torch.stack(g), np.array(gy), torch.stack(q), np.array(qy))
    return ((torch.stack(tr_x), np.array(tr_y)), seen_split,
            enrol(va), enrol(te), tr, va, te)


def evaluate(model, split, rng) -> dict:
    gal, gal_y, qry, qry_y = split
    gal_e = embed(model, gal)
    out = {}
    for a in CROP_ATTACKS:
        Xa = torch.stack([apply_crop_attack(qry[i], a, rng) for i in range(len(qry))])
        out[a] = top1_retrieval(gal_e, gal_y, embed(model, Xa), qry_y)
    return out


def train_arm(name, mode, tau, tr_x, tr_y, args, rng):
    torch.manual_seed(args.seed)
    model = make_encoder(args.dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    probs = sampling_probs(mode, tau) if mode != "none" else None
    if probs is not None:
        top = sorted(zip(CROP_ATTACKS, probs), key=lambda kv: -kv[1])[:4]
        print(f"  [{name}] attack mix: "
              + ", ".join(f"{a} {p:.2f}" for a, p in top), flush=True)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for idx in pk_batches(tr_y, args.P, args.K, args.steps_per_epoch, rng):
            xb = tr_x[idx]
            if probs is not None:
                picks = rng.choice(len(CROP_ATTACKS), size=len(idx), p=probs)
                xb = torch.stack([apply_crop_attack(xb[i], CROP_ATTACKS[picks[i]], rng)
                                  for i in range(len(idx))])
            emb = F.normalize(model(xb.to(DEV)), dim=1)
            loss = batch_hard_triplet(emb, tr_y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        if ep % 10 == 0 or ep == args.epochs:
            print(f"  [{name}] epoch {ep:3d}/{args.epochs} "
                  f"loss={np.mean(losses):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps-per-epoch", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    by_brand = build_corpus(size=128)
    (tr_x, tr_y), seen, val, test, trb, vab, teb = split4(by_brand, seed=args.seed)
    chance_seen = 1.0 / len(trb)
    print(f"train brands={len(trb)} ({len(tr_x)} crops)  "
          f"val-zs={len(vab)}  test-zs={len(teb)}", flush=True)
    print(f"seen gallery={len(seen[0])} query={len(seen[2])} | "
          f"val gallery={len(val[0])} query={len(val[2])} | "
          f"test gallery={len(test[0])} query={len(test[2])}", flush=True)

    print("\nsplit leakage audit:", flush=True)
    for nm, a, b in (("train vs seen-query", tr_x, seen[2]),
                     ("train vs val-query", tr_x, val[2]),
                     ("train vs test-query", tr_x, test[2]),
                     ("val gal vs query", val[0], val[2]),
                     ("test gal vs query", test[0], test[2])):
        r = split_leakage_gate(fingerprints(a), fingerprints(b))
        print(f"  {nm:<22} -> {'PASS' if r.passed else 'FAIL'} "
              f"({r.detail['dup_frac']:.1%})", flush=True)
        if not r.passed:
            raise SystemExit("Refusing to train on a leaking split.")

    print("\n--- untrained control ---", flush=True)
    ctrl = untrained_control()
    c = {k: evaluate(ctrl, s, rng) for k, s in
         (("seen", seen), ("val", val), ("test", test))}
    for k in ("seen", "val", "test"):
        cl, at = summarise(c[k])
        print(f"  {k:<5} clean={cl:.3f} cheap={at:.3f}", flush=True)

    arms = [("baseline", "none", 0.0),
            ("uniform_aug", "uniform", 0.0),
            ("cost_tau0.12", "cost", 0.12),
            ("cost_tau0.06", "cost", 0.06)]
    res = {}
    for name, mode, tau in arms:
        print(f"\n--- {name} ---", flush=True)
        m = train_arm(name, mode, tau, tr_x, tr_y, args, rng)
        r = {k: evaluate(m, s, rng) for k, s in
             (("seen", seen), ("val", val), ("test", test))}
        res[name] = r
        for k in ("seen", "val", "test"):
            cl, at = summarise(r[k])
            print(f"  {k:<5} clean={cl:.3f} cheap={at:.3f}", flush=True)
        torch.save(m.state_dict(), RESULTS / f"phishpedia_{name}.pt")

    # selection happens on val-zero-shot only
    cands = [a for a, _, _ in arms if a.startswith("cost")]
    pick = max(cands, key=lambda a: summarise(res[a]["val"])[1])
    print(f"\nselected on VAL zero-shot (cheap-attack): {pick}", flush=True)

    print("\n=== TEST zero-shot (read once) ===", flush=True)
    print(f"{'arm':<14}{'clean':>9}{'cheap':>9}{'gray':>9}{'seen-zs gap':>13}", flush=True)
    ctl_cl, ctl_at = summarise(c["test"])
    print(f"  {'control':<12}{ctl_cl:>9.3f}{ctl_at:>9.3f}"
          f"{c['test']['grayscale']:>9.3f}"
          f"{summarise(c['seen'])[0]-ctl_cl:>13.3f}", flush=True)
    for name, _, _ in arms:
        cl, at = summarise(res[name]["test"])
        gap = summarise(res[name]["seen"])[0] - cl
        print(f"  {name:<12}{cl:>9.3f}{at:>9.3f}"
              f"{res[name]['test']['grayscale']:>9.3f}{gap:>13.3f}", flush=True)

    b_cl, b_at = summarise(res["baseline"]["test"])
    p_cl, p_at = summarise(res[pick]["test"])
    print(f"\nprediction 1 (clean gap narrows): baseline gap "
          f"{summarise(res['baseline']['seen'])[0]-b_cl:+.3f} -> {pick} gap "
          f"{summarise(res[pick]['seen'])[0]-p_cl:+.3f}", flush=True)
    print(f"prediction 2 (zs attacked beats baseline +{b_at-ctl_at:.3f} over control): "
          f"{pick} {p_at-ctl_at:+.3f}", flush=True)
    print(f"prediction 3 (grayscale improves): baseline "
          f"{res['baseline']['test']['grayscale']:.3f} -> {pick} "
          f"{res[pick]['test']['grayscale']:.3f}", flush=True)
    u_cl, u_at = summarise(res["uniform_aug"]["test"])
    print(f"\ncontrol for the mechanism: uniform_aug zs cheap={u_at:.3f} vs "
          f"{pick} {p_at:.3f} (delta {p_at-u_at:+.3f})", flush=True)

    gate = competence_gate(clean=p_cl, attacked=p_at, chance=1.0 / len(teb),
                           control_clean=ctl_cl, control_attacked=ctl_at)
    print(f"\n{gate}", flush=True)

    with open(RESULTS / "phishpedia_defence.json", "w") as f:
        json.dump({"control": c, "arms": res, "selected": pick,
                   "n_train_brands": len(trb), "n_val": len(vab),
                   "n_test": len(teb), "args": vars(args),
                   "competence_passed": gate.passed}, f, indent=2)
    print(f"\nsaved -> {RESULTS/'phishpedia_defence.json'}", flush=True)


if __name__ == "__main__":
    main()
