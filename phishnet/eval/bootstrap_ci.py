"""Bootstrap 95% confidence intervals for the headline metrics.

Single-point scores hide sampling uncertainty. We resample each saved test-set
score/label vector with replacement (2,000 iterations) and report the 2.5th/97.5th
percentiles of F1, ROC-AUC and FPR. This is what turns "F1 = 0.966" into a claim a
reviewer can trust: F1 = 0.966 [0.961, 0.971].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def _fpr(y, s, thr):
    pred = (s >= thr).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return fp / max(fp + tn, 1)


def ci(scores_file: str, name: str, n_boot: int = 2000, seed: int = 1337) -> dict:
    d = np.load(scores_file, allow_pickle=True)
    s, y, thr = d["scores"], d["labels"], float(d["threshold"])
    rng = np.random.default_rng(seed)
    n = len(y)
    f1s, aucs, fprs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        f1s.append(f1_score(yb, (sb >= thr).astype(int)))
        aucs.append(roc_auc_score(yb, sb))
        fprs.append(_fpr(yb, sb, thr))
    def band(x):
        x = np.array(x)
        return [round(float(np.mean(x)), 4), round(float(np.percentile(x, 2.5)), 4),
                round(float(np.percentile(x, 97.5)), 4)]
    return {"name": name, "n": int(n), "f1": band(f1s), "roc_auc": band(aucs), "fpr": band(fprs)}


def main() -> int:
    out = Path("results")
    targets = [("url_test_scores.npz", "URL CNN"),
               ("text_test_scores.npz", "Text DistilBERT"),
               ("fusion_test_scores.npz", "Fusion (cross-attn)")]
    res = {}
    for f, name in targets:
        p = out / f
        if p.exists():
            res[name] = ci(str(p), name)
            r = res[name]
            print(f"  {name:22s} F1 {r['f1'][0]:.4f} [{r['f1'][1]:.4f}, {r['f1'][2]:.4f}]  "
                  f"ROC-AUC {r['roc_auc'][0]:.4f} [{r['roc_auc'][1]:.4f}, {r['roc_auc'][2]:.4f}]")
    (out / "bootstrap_ci.json").write_text(json.dumps(res, indent=2))
    print(f"saved -> {out/'bootstrap_ci.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
