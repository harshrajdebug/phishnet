"""Fine-tune the DistilBERT branch on message text."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phishnet.config import Config, ensure_dirs, resolve_device, set_seed
from phishnet.data.datasets import TextDataset, build_text_corpus
from phishnet.eval.metrics import compute, format_table, threshold_for_fpr
from phishnet.models.text_encoder import TextClassifier
from phishnet.train.common import predict, pos_weight_for, train_epochs


def forward(model, batch, device):
    ids, mask, y = batch
    ids, mask, y = ids.to(device), mask.to(device), y.to(device)
    return model(ids, mask), y


def run_tfidf_baseline(corpus) -> dict:
    """TF-IDF + linear SVM.

    Worth reporting because phishing lures are lexically stereotyped ("verify
    your account", "your mailbox will be closed"), and a bag-of-words model
    captures a surprising amount of that. If DistilBERT cannot beat it, the
    transformer is not earning its latency budget.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    (Xtr, ytr), (Xva, yva), (Xte, yte) = corpus["train"], corpus["val"], corpus["test"]
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True,
                          min_df=2, strip_accents="unicode")
    Atr = vec.fit_transform(Xtr)
    clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)
    clf.fit(Atr, ytr)
    s_va = clf.predict_proba(vec.transform(Xva))[:, 1]
    s_te = clf.predict_proba(vec.transform(Xte))[:, 1]
    thr = threshold_for_fpr(yva, s_va, 0.01)
    return compute(yte, s_te, thr)


def main() -> int:
    from transformers import AutoTokenizer

    cfg = Config()
    ensure_dirs(cfg)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    out_dir = Path("results")
    print(f"device={device}")

    print("[data] building text corpus")
    corpus = build_text_corpus(cfg)

    tok = AutoTokenizer.from_pretrained(cfg.text.model_name)
    ds = {k: TextDataset(v[0], v[1], tok, cfg.text.max_len) for k, v in corpus.items()}
    tl = DataLoader(ds["train"], batch_size=16, shuffle=True, drop_last=True)
    vl = DataLoader(ds["val"], batch_size=32)
    el = DataLoader(ds["test"], batch_size=32)
    # Held-out recent-lure test: current-theme messages the model never trained
    # on. This is the honest measure of whether the recent-lure upgrade helps.
    rl = DataLoader(ds["recent_test"], batch_size=32) if "recent_test" in ds else None

    print("[train] DistilBERT text classifier")
    model = TextClassifier(model_name=cfg.text.model_name, out_dim=cfg.fusion.d_model,
                           dropout=cfg.text.dropout, freeze_layers=cfg.text.freeze_layers)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {trainable:,}")

    # A small number of epochs at BERT learning rates; more overfits 6k messages.
    cfg.train.epochs = min(cfg.train.epochs, 4)
    model, _ = train_epochs(model, tl, vl, device, cfg, forward,
                            lr=cfg.train.bert_lr, tag="text_distilbert",
                            out_dir=out_dir,
                            pos_weight=pos_weight_for(corpus["train"][1], device))

    s_va, y_va = predict(model, vl, device, forward)
    thr = threshold_for_fpr(y_va, s_va, 0.01)
    s_te, y_te = predict(model, el, device, forward)

    results = {"distilbert": compute(y_te, s_te, thr)}
    # Detection on the held-out recent-lure split, at the SAME frozen threshold.
    if rl is not None:
        s_re, y_re = predict(model, rl, device, forward)
        results["distilbert@recent_lures"] = compute(y_re, s_re, thr)
    print("\n[baseline] TF-IDF + linear SVM")
    results["tfidf_svm"] = run_tfidf_baseline(corpus)

    print("\n" + format_table(results))
    (out_dir / "text_results.json").write_text(json.dumps(results, indent=2))
    np.savez(out_dir / "text_test_scores.npz", scores=s_te, labels=y_te, threshold=thr)
    print(f"\nsaved -> {out_dir/'text_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
