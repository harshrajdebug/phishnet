"""Race four defence mechanisms against the measured baseline.

Question: can we recover the robustness gain (+161-196% cost-to-evade) at
materially less than the measured -6% to -11% clean-F1 cost?

Phase-0 corrections applied throughout:
  * reliance and all gradients computed on CPU -- MPS was found to emit both NaN
    and materially wrong gradients (max |MPS-CPU| = 10.55 on identical weights)
  * cost-to-evade is EXACT by exhaustive enumeration over feature subsets of
    size <= 3 (4,525 subsets, one batched pass per sample), not greedy. Greedy
    was optimal on only 78.7% of samples with a p95 gap of +74%, and it errs by
    overstating evasion cost, which flatters the defence.
  * the raw 1/cost weighting (50x dynamic range) is replaced by a rank-based
    soft transform for every mechanism that needs per-feature weights.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phishnet.config import Config, set_seed
from phishnet.data.datasets import build_url_corpus
from phishnet.features.lexical import extract_batch
from phishnet.eval.cost_model import feature_costs
from phishnet.eval.metrics import compute, threshold_for_fpr
from phishnet.eval.feature_economics import MLP, scores

DEV = "cpu"          # deliberate: MPS autograd is unreliable for this graph


def soft_weights(costs: np.ndarray, hi: float = 5.0) -> np.ndarray:
    """Rank-based weights in [1, hi]: preserves the cost ordering but compresses
    the dynamic range of 1/cost that destabilised the loss landscape.

    Measured on the current price list, 1/cost spans 2.37..50.0, a 21.1x range;
    rank weighting compresses that to hi:1 (5x). The ordering is what the threat
    model asserts confidently, the exact ratios are not.
    """
    order = np.argsort(np.argsort(costs))            # 0 = cheapest
    r = order / max(len(costs) - 1, 1)               # 0..1, cheap -> 0
    return hi - (hi - 1.0) * r                       # cheap -> hi, expensive -> 1


def _loader(X, y, bs, shuffle=True):
    n = len(y)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield X[j], y[j]


def train_mech(Ztr, ytr, mech, costs, seed=0, epochs=30, bs=512,
               lam=0.05, beta=1.0, budget=0.15, drop_scale=0.35):
    """mech in {baseline, soft_penalty, cost_dropout, cost_trades, two_stage}"""
    set_seed(seed)
    m = MLP(Ztr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    X = torch.tensor(Ztr, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    w = torch.tensor(soft_weights(costs), dtype=torch.float32)
    # drop probability: cheap features dropped most, capped so nothing vanishes
    pdrop = torch.tensor(soft_weights(costs), dtype=torch.float32)
    pdrop = (pdrop - pdrop.min()) / (pdrop.max() - pdrop.min() + 1e-9) * drop_scale

    stage1 = epochs if mech != "two_stage" else epochs // 2
    for ep in range(epochs):
        for xb, yb in _loader(X, y, bs):
            opt.zero_grad()
            if mech == "cost_dropout":
                # stochastically hide cheap features: the model must build
                # redundant pathways onto expensive invariants
                mask = (torch.rand_like(xb) > pdrop).float()
                loss = F.binary_cross_entropy_with_logits(m(xb * mask), yb)
            elif mech == "cost_trades":
                # inner max under a COST BUDGET (not an Lp ball): greedily move
                # the cheapest features toward the benign centroid until spent
                with torch.no_grad():
                    xadv = xb.clone()
                    spent, order = 0.0, np.argsort(costs)
                    for fi in order:
                        if spent + costs[fi] > budget:
                            break
                        xadv[:, fi] = 0.0            # benign centroid in z-space
                        spent += costs[fi]
                p_clean = torch.sigmoid(m(xb))
                p_adv = torch.sigmoid(m(xadv))
                eps = 1e-6
                kl = (p_clean * ((p_clean + eps) / (p_adv + eps)).log()
                      + (1 - p_clean) * ((1 - p_clean + eps) / (1 - p_adv + eps)).log()).mean()
                loss = F.binary_cross_entropy_with_logits(m(xb), yb) + beta * kl
            elif mech in ("soft_penalty", "two_stage"):
                use_pen = (mech == "soft_penalty") or (ep >= stage1)
                if use_pen:
                    xb = xb.clone().requires_grad_(True)
                    out = m(xb)
                    g = torch.autograd.grad(out.sum(), xb, create_graph=True)[0]
                    pen = ((g ** 2) * w).sum(dim=1).mean()
                    loss = F.binary_cross_entropy_with_logits(out, yb) + lam * pen
                else:
                    loss = F.binary_cross_entropy_with_logits(m(xb), yb)
            else:
                loss = F.binary_cross_entropy_with_logits(m(xb), yb)
            loss.backward()
            opt.step()
        if mech == "two_stage" and ep == stage1 - 1:
            for gp in opt.param_groups:            # gentler LR for re-anchoring
                gp["lr"] = 5e-4
    return m.eval()


def exact_cte(m, X, y, thr, costs, benign_target, n_samples=120, budget=2.0):
    """Exact minimum-cost evasion by exhaustive enumeration over subsets k<=3."""
    subsets = []
    nf = len(costs)
    for k in (1, 2, 3):
        for combo in itertools.combinations(range(nf), k):
            subsets.append((float(sum(costs[i] for i in combo)), combo))
    subsets.sort(key=lambda t: t[0])
    costs_arr = np.array([s[0] for s in subsets])

    phish = X[y == 1]
    s0 = scores(m, phish, DEV)
    caught = phish[s0 >= thr][:n_samples]
    out, censored = [], 0
    for x in caught:
        V = np.repeat(x[None, :], len(subsets), axis=0)
        for r, (_, combo) in enumerate(subsets):
            for i in combo:
                V[r, i] = benign_target[i]
        sc = scores(m, V, DEV)
        hit = np.where(sc < thr)[0]
        if len(hit) == 0:
            censored += 1
            out.append(budget)
        else:
            out.append(costs_arr[hit[0]])
    return np.array(out), censored / max(len(caught), 1)


def reliance_cpu(m, X, n=3000):
    xb = torch.tensor(X[:n], dtype=torch.float32, requires_grad=True)
    g = torch.autograd.grad(m(xb).sum(), xb)[0].abs()
    g = torch.nan_to_num(g, 0.0).mean(0).numpy()
    return g / (g.sum() + 1e-12)


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
    print(f"corpus train={len(ytr):,} test={len(yte):,}   metric=EXACT CTE (k<=3)\n")

    seeds = [0, 1, 2]
    res = {}
    for mech in ("baseline", "soft_penalty", "cost_dropout", "cost_trades", "two_stage"):
        f1s, ctes, cens, rel_c = [], [], [], []
        for s in seeds:
            m = train_mech(Ztr, ytr, mech, costs, seed=s)
            thr = threshold_for_fpr(yva, scores(m, Zva, DEV), 0.01)
            f1s.append(compute(yte, scores(m, Zte, DEV), thr)["f1"])
            cte, cz = exact_cte(m, Zte, yte, thr, costs, bt)
            ctes.append(float(np.median(cte))); cens.append(cz)
            r = reliance_cpu(m, Zte)
            rel_c.append(float(np.corrcoef(r, 1.0 / costs)[0, 1]))
        res[mech] = {"f1": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
                     "cte": float(np.mean(ctes)), "cte_sd": float(np.std(ctes)),
                     "censored": float(np.mean(cens)),
                     "reliance_corr": float(np.nanmean(rel_c))}
        print(f"  {mech:14s} F1={res[mech]['f1']:.4f}±{res[mech]['f1_sd']:.4f}  "
              f"exactCTE={res[mech]['cte']:.4f}±{res[mech]['cte_sd']:.4f}  "
              f"corr={res[mech]['reliance_corr']:+.3f}")

    b = res["baseline"]
    print(f"\n{'mechanism':16s} {'ΔF1 %':>9s} {'ΔCTE %':>9s}   verdict")
    print("-" * 58)
    for k, v in res.items():
        if k == "baseline":
            continue
        df1 = (v["f1"] - b["f1"]) / b["f1"] * 100
        dct = (v["cte"] - b["cte"]) / b["cte"] * 100
        ok = "PASSES <2.5% budget" if (df1 > -2.5 and dct > 0) else "over budget"
        v["d_f1"], v["d_cte"] = df1, dct
        print(f"{k:16s} {df1:+8.2f}% {dct:+8.1f}%   {ok}")

    Path("results/mechanism_race.json").write_text(json.dumps(res, indent=2))
    print("\nsaved -> results/mechanism_race.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
