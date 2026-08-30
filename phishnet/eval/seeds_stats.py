"""Section 4.3 statistics: paired t-tests over the multi-seed replication.

Reads results/phishpedia_seeds.json (written by phishpedia_seeds.py) and emits
results/phishpedia_seeds_stats.json, the artifact every number in Section 4.3 is
audited against. All contrasts are two-sided paired t-tests over seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from phishnet.eval.phishpedia_train import summarise

ARMS = ("control", "baseline", "uniform_aug", "cost_tau0.06")
FIELDS = ("clean", "attacked", "grayscale", "logo_delete")
PAIRS = (("cost_tau0.06", "uniform_aug"), ("cost_tau0.06", "baseline"),
         ("uniform_aug", "baseline"), ("baseline", "control"),
         ("cost_tau0.06", "control"))


def main() -> int:
    src = Path("results/phishpedia_seeds.json")
    if not src.exists():
        raise SystemExit(f"{src} not found; run phishnet.eval.phishpedia_seeds first")
    d = json.loads(src.read_text())
    S = sorted(d, key=int)
    print(f"n seeds = {len(S)}")

    def get(arm, field, split="test"):
        out = []
        for s in S:
            per = (d[s]["control"] if arm == "control" else d[s]["arms"][arm])[split]
            out.append(summarise(per)[0] if field == "clean" else
                       summarise(per)[1] if field == "attacked" else per[field])
        return np.array(out)

    res = {"n_seeds": len(S), "seeds": S, "arms": {}, "contrasts": {}}
    for a in ARMS:
        res["arms"][a] = {f: {"mean": float(get(a, f).mean()),
                              "sd": float(get(a, f).std(ddof=1))} for f in FIELDS}

    print(f"\n{'contrast':<40}{'mean':>10}{'t':>8}{'p':>10}{'wins':>8}")
    for f in FIELDS:
        print(f"--- {f} ---")
        for a, b in PAIRS:
            x, y = get(a, f), get(b, f)
            t, p = stats.ttest_rel(x, y)
            dif = x - y
            res["contrasts"][f"{f}|{a}-{b}"] = {
                "mean": float(dif.mean()), "sd": float(dif.std(ddof=1)),
                "t": float(t), "p": float(p), "wins": int((dif > 0).sum()),
                "n": len(S), "per_seed": dif.tolist()}
            print(f"  {a+' - '+b:<38}{dif.mean():>+10.4f}{t:>+8.2f}"
                  f"{p:>10.5f}{(dif>0).sum():>5}/{len(S)}")

    with open("results/phishpedia_seeds_stats.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nsaved -> results/phishpedia_seeds_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
