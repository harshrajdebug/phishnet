"""Five-seed replication of the cost-weighted defence, paired within seed.

tau is frozen at 0.06, selected once on the validation zero-shot brands at seed 0
and not re-tuned. Re-selecting per seed would fit the hyperparameter to the
evaluation.

Why paired. The single-seed run let all arms share one advancing RNG, so each arm
saw a different batch sequence and different random attack parameters at
evaluation. Some of the margin between arms was therefore batch-order luck. Here
every arm inside a seed gets identical initialisation, identical batch indices and
identical eval-time attack draws, so the augmentation policy is the only thing
that differs and the arm-to-arm difference can be read as a paired effect.

The margin under test is small (+0.068 cost-weighted over uniform on zero-shot
attacked retrieval). This project has already had a single-run effect of similar
size evaporate across seeds, which is the entire reason for this run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .competence_gate import skill, split_leakage_gate
from .phishpedia_crucible import (CROP_ATTACKS, DEV, RESULTS, apply_crop_attack,
                                  build_corpus, embed, top1_retrieval,
                                  untrained_control)
from .phishpedia_defence import sampling_probs, split4
from .phishpedia_train import (CHEAP, batch_hard_triplet, fingerprints,
                               make_encoder, pk_batches, summarise)

ARMS = [("baseline", "none", 0.0),
        ("uniform_aug", "uniform", 0.0),
        ("cost_tau0.06", "cost", 0.06)]
CKPT = RESULTS / "phishpedia_seeds.json"


def evaluate_paired(model, split, eval_seed: int) -> dict:
    """Identical attack draws for every arm, so arms face the same test."""
    gal, gal_y, qry, qry_y = split
    gal_e = embed(model, gal)
    out = {}
    for a in CROP_ATTACKS:
        rng = np.random.default_rng(eval_seed)      # reset per attack, per arm
        Xa = torch.stack([apply_crop_attack(qry[i], a, rng) for i in range(len(qry))])
        out[a] = top1_retrieval(gal_e, gal_y, embed(model, Xa), qry_y)
    return out


def train_arm(name, mode, tau, tr_x, tr_y, args, seed):
    torch.manual_seed(seed)                      # identical init across arms
    model = make_encoder(args.dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch_rng = np.random.default_rng(10_000 + seed)   # identical batches
    aug_rng = np.random.default_rng(20_000 + seed)     # attack draws
    probs = sampling_probs(mode, tau) if mode != "none" else None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for idx in pk_batches(tr_y, args.P, args.K, args.steps_per_epoch, batch_rng):
            xb = tr_x[idx]
            if probs is not None:
                picks = aug_rng.choice(len(CROP_ATTACKS), size=len(idx), p=probs)
                xb = torch.stack([apply_crop_attack(xb[i], CROP_ATTACKS[picks[i]], aug_rng)
                                  for i in range(len(idx))])
            emb = F.normalize(model(xb.to(DEV)), dim=1)
            loss = batch_hard_triplet(emb, tr_y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        if ep % 15 == 0 or ep == args.epochs:
            print(f"    [{name}] ep {ep}/{args.epochs} loss={np.mean(losses):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps-per-epoch", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()

    by_brand = build_corpus(size=128)
    store = {}
    if CKPT.exists():
        store = json.loads(CKPT.read_text())
        print(f"resuming; seeds already done: {sorted(store)}", flush=True)

    for seed in args.seeds:
        if str(seed) in store:
            print(f"\n=== seed {seed} (cached) ===", flush=True)
            continue
        print(f"\n=== seed {seed} ===", flush=True)
        (tr_x, tr_y), seen, val, test, trb, vab, teb = split4(by_brand, seed=seed)
        print(f"  train {len(trb)} brands / {len(tr_x)} crops | "
              f"val-zs {len(vab)} | test-zs {len(teb)}", flush=True)

        ok = True
        for nm, a, b in (("train vs test-q", tr_x, test[2]),
                         ("train vs val-q", tr_x, val[2]),
                         ("test gal vs q", test[0], test[2])):
            r = split_leakage_gate(fingerprints(a), fingerprints(b))
            if not r.passed:
                print(f"  LEAK {nm}: {r.detail['dup_frac']:.1%} -- skipping seed",
                      flush=True)
                ok = False
        if not ok:
            continue

        ev = 50_000 + seed
        rec = {"n_train_brands": len(trb), "n_val": len(vab), "n_test": len(teb)}
        ctrl = untrained_control()
        rec["control"] = {k: evaluate_paired(ctrl, s, ev)
                          for k, s in (("seen", seen), ("val", val), ("test", test))}
        cc, ca = summarise(rec["control"]["test"])
        print(f"  control  test clean={cc:.3f} cheap={ca:.3f}", flush=True)

        rec["arms"] = {}
        for name, mode, tau in ARMS:
            m = train_arm(name, mode, tau, tr_x, tr_y, args, seed)
            rec["arms"][name] = {k: evaluate_paired(m, s, ev)
                                 for k, s in (("seen", seen), ("val", val),
                                              ("test", test))}
            cl, at = summarise(rec["arms"][name]["test"])
            print(f"  {name:<13} test clean={cl:.3f} cheap={at:.3f} "
                  f"gray={rec['arms'][name]['test']['grayscale']:.3f}", flush=True)

        store[str(seed)] = rec
        CKPT.write_text(json.dumps(store, indent=2))   # resumable
        print(f"  checkpointed -> {CKPT}", flush=True)

    report(store)


def _sd(v) -> float:
    return float(v.std(ddof=1)) if len(v) > 1 else 0.0


def report(store: dict) -> None:
    seeds = sorted(store, key=int)
    if not seeds:
        return
    print(f"\n\n=== {len(seeds)} seeds: mean +/- sd on TEST zero-shot ===", flush=True)

    def col(arm, field, split="test"):
        out = []
        for s in seeds:
            src = store[s]["control"] if arm == "control" else store[s]["arms"][arm]
            per = src[split]
            out.append(summarise(per)[0] if field == "clean" else
                       summarise(per)[1] if field == "cheap" else per[field])
        return np.array(out)

    names = ["control"] + [a for a, _, _ in ARMS]
    print(f"{'arm':<14}{'clean':>16}{'attacked':>16}{'grayscale':>16}", flush=True)
    for a in names:
        r = []
        for f in ("clean", "cheap", "grayscale"):
            v = col(a, f)
            r.append(f"{v.mean():.3f}+/-{_sd(v):.3f}")
        print(f"  {a:<12}{r[0]:>16}{r[1]:>16}{r[2]:>16}", flush=True)

    print("\n--- paired deltas (per seed, then averaged) ---", flush=True)
    pairs = [("cost_tau0.06", "uniform_aug", "cost-weighting vs plain augmentation"),
             ("cost_tau0.06", "baseline", "cost-weighting vs no augmentation"),
             ("uniform_aug", "baseline", "augmentation alone vs none")]
    for a, b, label in pairs:
        d = col(a, "cheap") - col(b, "cheap")
        g = col(a, "grayscale") - col(b, "grayscale")
        wins = int((d > 0).sum())
        se = _sd(d) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        t = d.mean() / se if se and se > 0 else float("nan")
        print(f"  {label}", flush=True)
        print(f"     attacked  {d.mean():+.4f} +/- {_sd(d):.4f}  "
              f"(wins {wins}/{len(d)}, t={t:+.2f})  per-seed "
              f"{[f'{x:+.3f}' for x in d]}", flush=True)
        print(f"     grayscale {g.mean():+.4f} +/- {_sd(g):.4f}  "
              f"(wins {int((g>0).sum())}/{len(g)})", flush=True)

    print("\n--- skill-normalised seen/unseen gap ---", flush=True)
    print("  (raw retrieval is not comparable: seen and test have different "
          "class counts)", flush=True)
    for a in names:
        gaps = []
        for s in seeds:
            src = store[s]["control"] if a == "control" else store[s]["arms"][a]
            ch_s = 1.0 / store[s]["n_train_brands"]
            ch_t = 1.0 / store[s]["n_test"]
            gaps.append(skill(summarise(src["seen"])[0], ch_s)
                        - skill(summarise(src["test"])[0], ch_t))
        gaps = np.array(gaps)
        print(f"  {a:<14}{gaps.mean():+.4f} +/- {_sd(gaps):.4f}", flush=True)

    with open(RESULTS / "phishpedia_seeds_summary.json", "w") as f:
        json.dump({"seeds": seeds}, f)
    print(f"\nfull per-seed record -> {CKPT}", flush=True)


if __name__ == "__main__":
    main()
