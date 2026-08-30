"""Cost-budgeted TRADES with a proper inner maximisation, plus Kaplan-Meier CTE.

Two corrections to the Phase-1 run:

1. The earlier cost-TRADES used a fixed greedy fill for the inner problem -- it
   spent the budget on the cheapest features regardless of their effect. That is
   not an argmax. Here the inner problem searches the *cost-feasible* subset space
   (all subsets of size <=3 with cost <= B) and selects the perturbation that
   maximises the KL divergence, approximated per step by sampling candidates.

2. Median CTE is biased when censoring differs across models (dropout censors
   34% of samples, baseline 9%). Kaplan-Meier treats un-evaded samples as
   right-censored and estimates the median without discarding or flooring them.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from phishnet.config import Config, set_seed
from phishnet.data.datasets import build_url_corpus
from phishnet.features.lexical import extract_batch
from phishnet.eval.cost_model import feature_costs
from phishnet.eval.metrics import compute, threshold_for_fpr
from phishnet.eval.feature_economics import MLP, scores
from phishnet.eval.mechanism_race import train_mech, exact_cte, DEV


def affordable_subsets(costs, budget, max_k=3):
    out = []
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(range(len(costs)), k):
            c = sum(costs[i] for i in combo)
            if c <= budget:
                out.append(combo)
    return out


def train_cost_trades(Ztr, ytr, costs, budget=0.15, beta=2.0, seed=0,
                      epochs=30, bs=512, n_cand=16):
    """TRADES whose inner max searches the cost-feasible perturbation set."""
    set_seed(seed)
    m = MLP(Ztr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    X = torch.tensor(Ztr, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    cands = affordable_subsets(costs, budget)
    rng = np.random.default_rng(seed)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            xb, yb = X[perm[i:i + bs]], y[perm[i:i + bs]]
            with torch.no_grad():
                p_clean = torch.sigmoid(m(xb))
                best_kl = torch.full((len(xb),), -1e9)
                best = xb.clone()
                # stochastic argmax over affordable perturbations
                for ci in rng.choice(len(cands), size=min(n_cand, len(cands)), replace=False):
                    trial = xb.clone()
                    for f in cands[ci]:
                        trial[:, f] = 0.0                 # benign centroid (z-space)
                    p_adv = torch.sigmoid(m(trial))
                    eps = 1e-6
                    kl = (p_clean * ((p_clean + eps) / (p_adv + eps)).log()
                          + (1 - p_clean) * ((1 - p_clean + eps) / (1 - p_adv + eps)).log())
                    upd = kl > best_kl
                    best[upd] = trial[upd]
                    best_kl[upd] = kl[upd]
            opt.zero_grad()
            out = m(xb)
            p_c = torch.sigmoid(out)
            p_a = torch.sigmoid(m(best))
            eps = 1e-6
            kl = (p_c * ((p_c + eps) / (p_a + eps)).log()
                  + (1 - p_c) * ((1 - p_c + eps) / (1 - p_a + eps)).log()).mean()
            (F.binary_cross_entropy_with_logits(out, yb) + beta * kl).backward()
            opt.step()
    return m.eval()


def km_median(times, events):
    """Kaplan-Meier median. events=1 observed evasion, 0 right-censored."""
    order = np.argsort(times)
    t, e = np.asarray(times)[order], np.asarray(events)[order]
    n_at_risk, S = len(t), 1.0
    for i, ti in enumerate(t):
        if e[i] == 1:
            S *= (1 - 1.0 / n_at_risk)
            if S <= 0.5:
                return float(ti)
        n_at_risk -= 1
    return float("nan")          # survival never reaches 0.5 -> median not reached


def main() -> int:
    cfg = Config()
    c = build_url_corpus(cfg, verbose=False)
    Xtr = extract_batch(c["train"].urls); ytr = c["train"].labels
    Xva = extract_batch(c["val"].urls);   yva = c["val"].labels
    Xte = extract_batch(c["test"].urls);  yte = c["test"].labels
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zva, Zte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    bt = Ztr[ytr == 0].mean(0); costs = feature_costs()
    BUD = 2.0
    seeds = [0, 1, 2]

    def report(name, make):
        f1s, kms, meds, czs = [], [], [], []
        for s in seeds:
            m = make(s)
            thr = threshold_for_fpr(yva, scores(m, Zva, DEV), 0.01)
            f1s.append(compute(yte, scores(m, Zte, DEV), thr)["f1"])
            cte, cz = exact_cte(m, Zte, yte, thr, costs, bt, budget=BUD)
            ev = (cte < BUD).astype(int)          # censored samples hit the ceiling
            kms.append(km_median(cte, ev)); meds.append(np.median(cte)); czs.append(cz)
        km = np.nanmean(kms)
        print(f"  {name:22s} F1={np.mean(f1s):.4f}  naive_median={np.mean(meds):.4f}  "
              f"KM_median={km:.4f}  censored={np.mean(czs):.1%}")
        return {"f1": float(np.mean(f1s)), "km": float(km),
                "median": float(np.mean(meds)), "censored": float(np.mean(czs))}

    print("Kaplan-Meier corrected CTE (censoring-aware)\n")
    out = {}
    out["baseline"] = report("baseline", lambda s: train_mech(Ztr, ytr, "baseline", costs, seed=s))
    out["cost_dropout"] = report("cost_dropout(0.50)",
                                 lambda s: train_mech(Ztr, ytr, "cost_dropout", costs, seed=s, drop_scale=0.50))
    print("\ncost-budgeted TRADES with true inner argmax:")
    for beta in (1.0, 2.0, 4.0):
        out[f"trades_b{beta}"] = report(f"cost_trades(beta={beta})",
                                        lambda s, b=beta: train_cost_trades(Ztr, ytr, costs, beta=b, seed=s))

    b = out["baseline"]
    print(f"\n{'model':22s} {'ΔF1 %':>9s} {'ΔKM-CTE %':>11s}")
    print("-" * 46)
    for k, v in out.items():
        if k == "baseline":
            continue
        print(f"{k:22s} {(v['f1']-b['f1'])/b['f1']*100:+8.2f}% "
              f"{(v['km']-b['km'])/b['km']*100:+10.1f}%")
    Path("results/trades_km.json").write_text(json.dumps(out, indent=2))
    print("\nsaved -> results/trades_km.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
