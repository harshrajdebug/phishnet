"""Section 4.1.1 and Table 4.1: H1, H2, and per-feature evasion.

H1  corr(reliance, 1/cost)            -- is reliance concentrated on cheap features?
H2  corr(evasion, reliance/cost)      -- does evasion track reliance priced by cost?

Pinned to CPU, and that is load-bearing rather than incidental. On MPS the
baseline reliance gradient returned all-NaN while the defended arms stayed
finite; np.mean over seeds propagated it silently, and a *completed* run reported
both correlations as `nan`. CPU reproduces H1 = +0.201 +/- 0.031 and
H2 = +0.774 +/- 0.045 over five seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phishnet.config import Config
from phishnet.data.datasets import build_url_corpus
from phishnet.features.lexical import FEATURE_NAMES, extract_batch
from phishnet.eval.cost_model import feature_costs
from phishnet.eval.feature_economics import (per_feature_evasion, reliance,
                                             scores, train)
from phishnet.eval.metrics import threshold_for_fpr

DEV = "cpu"
SEEDS = (0, 1, 2, 3, 4)


def main() -> int:
    cfg = Config()
    c = build_url_corpus(cfg, verbose=False)
    Xtr = extract_batch(c["train"].urls); ytr = c["train"].labels
    Xva = extract_batch(c["val"].urls);   yva = c["val"].labels
    Xte = extract_batch(c["test"].urls);  yte = c["test"].labels
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zva, Zte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    bt = Ztr[ytr == 0].mean(0)
    costs = feature_costs()

    h1, h2, evs = [], [], []
    for s in SEEDS:
        m = train(Ztr, ytr, DEV, costs=costs, lam=0.0, seed=s)
        rel = reliance(m, Zte, DEV)
        if not np.isfinite(rel).all():
            raise RuntimeError(f"non-finite reliance at seed {s}; refusing to "
                               "average NaN into a load-bearing correlation")
        thr = threshold_for_fpr(yva, scores(m, Zva, DEV), 0.01)
        ev = per_feature_evasion(m, Zte, yte, thr, bt, DEV)
        evv = np.array([ev[f] for f in FEATURE_NAMES])
        h1.append(np.corrcoef(rel, 1.0 / costs)[0, 1])
        h2.append(np.corrcoef(evv, rel / costs)[0, 1])
        evs.append(evv)
        print(f"seed {s}: H1={h1[-1]:+.3f}  H2={h2[-1]:+.3f}", flush=True)

    h1, h2, E = np.array(h1), np.array(h2), np.array(evs)
    print(f"\nH1 corr(reliance, 1/cost)       = {h1.mean():+.3f} +/- {h1.std(ddof=1):.3f}")
    print(f"H2 corr(evasion, reliance/cost) = {h2.mean():+.3f} +/- {h2.std(ddof=1):.3f}")

    mean, sdv = E.mean(0), E.std(0, ddof=1)
    print("\nTable 4.1 -- most effective single-feature evasions:")
    for i in np.argsort(-mean)[:6]:
        print(f"  {FEATURE_NAMES[i]:<24} evades={100*mean[i]:5.1f}% "
              f"+/-{100*sdv[i]:4.1f}  cost={costs[i]:.3f}")

    Path("results").mkdir(exist_ok=True)
    with open("results/h1_h2_cpu.json", "w") as f:
        json.dump({"H1": h1.tolist(), "H2": h2.tolist(),
                   "H1_mean": float(h1.mean()), "H1_sd": float(h1.std(ddof=1)),
                   "H2_mean": float(h2.mean()), "H2_sd": float(h2.std(ddof=1)),
                   "device": DEV, "n_seeds": len(SEEDS)}, f, indent=2)
    with open("results/per_feature_evasion_cpu.json", "w") as f:
        json.dump({"feature_names": FEATURE_NAMES, "evasion_mean": mean.tolist(),
                   "evasion_sd": sdv.tolist(), "costs": costs.tolist(),
                   "n_seeds": len(SEEDS), "device": DEV}, f, indent=2)
    print("\nsaved -> results/h1_h2_cpu.json, results/per_feature_evasion_cpu.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
