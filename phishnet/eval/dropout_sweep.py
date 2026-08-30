"""Table 4.3: the cost-weighted dropout Pareto frontier (protocol P2).

Reconstructed as a committed entry point. The original sweep was run ad hoc and
its result file (results/dropout_sweep.json) therefore had no script behind it,
which makes the paper's "no accuracy tax" claim unreproducible for a reviewer.

Protocol P2 has its own baseline (F1 0.9364, exact CTE 0.0595) and must not be
mixed with the P1 mechanism-race baseline (F1 0.9358, CTE 0.0642) -- see the
protocol note in Section 4.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phishnet.config import Config
from phishnet.data.datasets import build_url_corpus
from phishnet.features.lexical import extract_batch
from phishnet.eval.cost_model import feature_costs
from phishnet.eval.mechanism_race import DEV, exact_cte, scores, train_mech
from phishnet.eval.metrics import compute, threshold_for_fpr

SCALES = (0.15, 0.25, 0.35, 0.50, 0.65)
SEEDS = (0, 1, 2, 3, 4)   # recovered from the committed artifact's
                          # resist values, which are exact integers over 5*120


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

    def run(mech: str, **kw):
        f1s, ctes, res = [], [], []
        for s in SEEDS:
            m = train_mech(Ztr, ytr, mech, costs, seed=s, **kw)
            thr = threshold_for_fpr(yva, scores(m, Zva, DEV), 0.01)
            f1s.append(compute(yte, scores(m, Zte, DEV), thr)["f1"])
            cte, cz = exact_cte(m, Zte, yte, thr, costs, bt)
            ctes.append(float(np.median(cte)))
            res.append(float(cz))       # fraction whose evasion failed within budget
        return (float(np.mean(f1s)), float(np.std(f1s)),
                float(np.mean(ctes)), float(np.std(ctes)), float(np.mean(res)))

    b_f1, b_f1sd, b_cte, b_ctesd, b_res = run("baseline")
    print(f"baseline  F1={b_f1:.4f}+/-{b_f1sd:.4f}  CTE={b_cte:.4f}+/-{b_ctesd:.4f}  "
          f"resist={b_res:.1%}", flush=True)

    out = {"baseline": {"f1": b_f1, "cte": b_cte, "resist": b_res}, "sweep": {}}
    print(f"\n{'drop_scale':>10} {'F1':>18} {'dF1%':>8} {'exactCTE':>18} {'dCTE%':>9}")
    for s in SCALES:
        f1, f1sd, cte, ctesd, resist = run("cost_dropout", drop_scale=s)
        d_f1 = 100 * (f1 - b_f1) / b_f1
        d_cte = 100 * (cte - b_cte) / b_cte
        out["sweep"][str(s)] = {"f1": f1, "f1_sd": f1sd, "cte": cte,
                                "cte_sd": ctesd, "resist": resist,
                                "d_f1": d_f1, "d_cte": d_cte}
        print(f"{s:>10} {f1:.4f}+/-{f1sd:.4f} {d_f1:>+7.2f}% "
              f"{cte:.4f}+/-{ctesd:.4f} {d_cte:>+8.1f}%", flush=True)

    Path("results").mkdir(exist_ok=True)
    with open("results/dropout_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> results/dropout_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
