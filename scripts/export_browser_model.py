"""Export a compact, browser-runnable URL classifier to JSON.

The deployed system uses a 300MB neural model behind a server. For a public,
install-free demo we need something that runs client-side in a few kilobytes of
JavaScript. We train a small Random Forest (few shallow trees) on the same 30
lexical URL features as the paper's classical baseline, on the same real corpora,
then serialise the trees plus the standardiser, the reputation allowlist, and the
keyword lexicons to one JSON file. The browser re-implements the identical feature
extraction and averages the trees. It is honest about what it is: the trained
lexical classifier (a strong baseline, Table 1), not the full multimodal model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phishnet.config import Config
from phishnet.data.datasets import build_url_corpus
from phishnet.features.lexical import (
    FEATURE_NAMES, BRANDS, RISKY_TLDS, FREE_HOSTS, SUSPICIOUS_TOKENS,
    registrable_domain,
)
from phishnet.eval.metrics import compute, threshold_for_fpr


def tree_to_dict(t) -> dict:
    """Flatten a sklearn tree into parallel arrays (compact, JS-friendly)."""
    tr = t.tree_
    # value[node] = [[neg, pos]]; store P(phishing).
    proba = (tr.value[:, 0, 1] / tr.value[:, 0, :].sum(axis=1)).astype(float)
    return {
        "feature": tr.feature.astype(int).tolist(),      # -2 = leaf
        "threshold": np.round(tr.threshold, 5).tolist(),
        "left": tr.children_left.astype(int).tolist(),
        "right": tr.children_right.astype(int).tolist(),
        "proba": np.round(proba, 4).tolist(),
    }


def main() -> int:
    from sklearn.ensemble import RandomForestClassifier

    cfg = Config()
    print("[data] building URL corpus (real phishing + harvested benign)")
    corpus = build_url_corpus(cfg, verbose=True)

    from phishnet.features.lexical import extract_batch
    Xtr = extract_batch(corpus["train"].urls); ytr = corpus["train"].labels
    Xva = extract_batch(corpus["val"].urls);   yva = corpus["val"].labels
    Xte = extract_batch(corpus["test"].urls);  yte = corpus["test"].labels

    # Standardise (store mean/std for the browser).
    mean = Xtr.mean(axis=0); std = Xtr.std(axis=0) + 1e-9
    Ztr, Zva, Zte = (Xtr - mean) / std, (Xva - mean) / std, (Xte - mean) / std

    # Compact forest: 40 shallow trees keep the JSON small (<1 MB) and inference
    # trivial, while retaining most of the full baseline's accuracy.
    rf = RandomForestClassifier(n_estimators=40, max_depth=10, min_samples_leaf=25,
                                class_weight="balanced", n_jobs=-1, random_state=1337)
    rf.fit(Ztr, ytr)
    s_va = rf.predict_proba(Zva)[:, 1]
    s_te = rf.predict_proba(Zte)[:, 1]
    thr = threshold_for_fpr(yva, s_va, 0.01)
    m = compute(yte, s_te, thr)
    print(f"[compact RF] F1={m['f1']:.4f} ROC-AUC={m['roc_auc']:.4f} "
          f"FPR={m['fpr']:.4f} thr={thr:.4f}")

    # Reputation allowlist: top registrable domains minus shared-hosting.
    raw = Path(cfg.data.raw_dir)
    doms, seen = [], set()
    for i, line in enumerate(raw.joinpath("benign_urls.txt").read_text().splitlines()):
        if i >= 6000:
            break
        d = registrable_domain(line.strip())
        if d and d not in seen and not any(h in d for h in FREE_HOSTS):
            seen.add(d); doms.append(d)

    model = {
        "meta": {"kind": "compact_random_forest", "n_trees": len(rf.estimators_),
                 "f1": round(m["f1"], 4), "roc_auc": round(m["roc_auc"], 4),
                 "fpr": round(m["fpr"], 4), "threshold": round(float(thr), 4)},
        "feature_names": list(FEATURE_NAMES),
        "mean": np.round(mean, 6).tolist(),
        "std": np.round(std, 6).tolist(),
        "trees": [tree_to_dict(t) for t in rf.estimators_],
        "lexicons": {"brands": list(BRANDS), "risky_tlds": list(RISKY_TLDS),
                     "free_hosts": list(FREE_HOSTS), "suspicious_tokens": list(SUSPICIOUS_TOKENS)},
        "reputation": doms,
    }
    out = Path("results/browser_model.json")
    out.write_text(json.dumps(model, separators=(",", ":")))
    print(f"exported {out} ({out.stat().st_size/1024:.0f} KB, {len(doms)} reputable domains)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
