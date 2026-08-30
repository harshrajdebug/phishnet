"""Train the URL branch and the classical baselines it is measured against."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phishnet.config import Config, ensure_dirs, resolve_device, set_seed
from phishnet.data.datasets import (URLDataset, apply_base_rate,
                                    build_url_corpus)
from phishnet.eval.metrics import compute, format_table, threshold_for_fpr
from phishnet.models.url_cnn import URLClassifier
from phishnet.train.common import predict, pos_weight_for, train_epochs


def forward(model, batch, device):
    x, y = batch
    x, y = x.to(device), y.to(device)
    return model(x), y


def run_classical_baselines(splits, out_dir: Path) -> dict:
    """Random Forest and Logistic Regression over the hand-engineered features.

    These are not strawmen -- on URL-only phishing detection a well-tuned RF over
    good lexical features is a genuinely strong baseline, and reporting it keeps
    the neural result honest.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from phishnet.features.lexical import extract_batch

    print("  extracting lexical features...")
    Xtr = extract_batch(splits["train"].urls)
    Xva = extract_batch(splits["val"].urls)
    Xte = extract_batch(splits["test"].urls)
    ytr, yva, yte = splits["train"].labels, splits["val"].labels, splits["test"].labels

    scaler = StandardScaler().fit(Xtr)
    results = {}

    rf = RandomForestClassifier(n_estimators=300, max_depth=24, n_jobs=-1,
                                class_weight="balanced", random_state=1337)
    rf.fit(Xtr, ytr)
    s_va, s_te = rf.predict_proba(Xva)[:, 1], rf.predict_proba(Xte)[:, 1]
    thr = threshold_for_fpr(yva, s_va, 0.01)
    results["random_forest"] = compute(yte, s_te, thr)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(scaler.transform(Xtr), ytr)
    s_va, s_te = (lr.predict_proba(scaler.transform(Xva))[:, 1],
                  lr.predict_proba(scaler.transform(Xte))[:, 1])
    thr = threshold_for_fpr(yva, s_va, 0.01)
    results["logistic_regression"] = compute(yte, s_te, thr)

    import joblib
    joblib.dump({"rf": rf, "lr": lr, "scaler": scaler}, out_dir / "classical_baselines.joblib")

    from phishnet.features.lexical import FEATURE_NAMES
    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda x: -x[1])
    print("  top RF features: " + ", ".join(f"{k}={v:.3f}" for k, v in imp[:8]))
    results["_rf_feature_importance"] = {k: float(v) for k, v in imp}
    return results


def main() -> int:
    cfg = Config()
    ensure_dirs(cfg)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    out_dir = Path("results")
    print(f"device={device}")

    print("[data] building URL corpus")
    splits = build_url_corpus(cfg)

    train_ds = URLDataset(splits["train"], cfg.url.max_len)
    val_ds = URLDataset(splits["val"], cfg.url.max_len)
    test_ds = URLDataset(splits["test"], cfg.url.max_len)
    tl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                    num_workers=cfg.train.num_workers, drop_last=True)
    vl = DataLoader(val_ds, batch_size=256)
    el = DataLoader(test_ds, batch_size=256)

    print("[train] URL 1D-CNN")
    model = URLClassifier(embed_dim=cfg.url.embed_dim, num_filters=cfg.url.num_filters,
                          kernel_sizes=tuple(cfg.url.kernel_sizes), out_dim=cfg.fusion.d_model,
                          dropout=cfg.url.dropout)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    model, _ = train_epochs(model, tl, vl, device, cfg, forward,
                            tag="url_cnn", out_dir=out_dir,
                            pos_weight=pos_weight_for(splits["train"].labels, device))

    # Operating threshold is chosen on validation, then frozen for test.
    s_va, y_va = predict(model, vl, device, forward)
    thr = threshold_for_fpr(y_va, s_va, 0.01)
    s_te, y_te = predict(model, el, device, forward)

    results = {"url_cnn": compute(y_te, s_te, thr)}

    # The same model re-scored at a deployment-realistic prevalence.
    br = cfg.data.eval_base_rate
    rs = apply_base_rate(splits["test"], br)
    idx = {u: i for i, u in enumerate(splits["test"].urls)}
    keep = [idx[u] for u in rs.urls if u in idx]
    results[f"url_cnn@base_rate_{br}"] = compute(y_te[keep], s_te[keep], thr)

    print("\n[baselines] classical models")
    results.update(run_classical_baselines(splits, out_dir))

    printable = {k: v for k, v in results.items() if not k.startswith("_")}
    print("\n" + format_table(printable))
    (out_dir / "url_results.json").write_text(json.dumps(results, indent=2))
    np.savez(out_dir / "url_test_scores.npz", scores=s_te, labels=y_te,
             urls=np.array(splits["test"].urls, dtype=object), threshold=thr)
    print(f"\nsaved -> {out_dir/'url_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
