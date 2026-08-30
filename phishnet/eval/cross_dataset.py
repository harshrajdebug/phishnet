"""Cross-source generalisation of the URL branch.

A domain-disjoint split proves the model does not memorise hosts. It does not
prove the model generalises across *data sources*: two feeds can share collection
biases (a crawler's URL-shape preferences, a de-duplication rule) that a
within-corpus split cannot expose. The stronger test trains on one phishing source
and tests on a different one it never saw. We train on the Phishing.Database ACTIVE
list and test on the OpenPhish live feed (benign held fixed from Tranco, split by
domain), and report the drop, if any, versus the within-corpus number.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from phishnet.config import Config, resolve_device, set_seed
from phishnet.features.url_encoding import encode_batch
from phishnet.features.lexical import registrable_domain
from phishnet.models.url_cnn import URLClassifier, URLEncoder
from phishnet.eval.metrics import compute, threshold_for_fpr
from phishnet.train.common import train_epochs


def _load(path):
    return [u for u in Path(path).read_text(errors="ignore").splitlines() if u.startswith(("http://", "https://"))]


def _domain_split(urls, labels, frac_test=0.3, seed=1337):
    """Split so no registrable domain crosses train/test."""
    import random
    rng = random.Random(seed)
    by_dom = {}
    for u, y in zip(urls, labels):
        by_dom.setdefault(registrable_domain(u), []).append((u, y))
    doms = list(by_dom); rng.shuffle(doms)
    n_test = int(len(doms) * frac_test)
    test_d = set(doms[:n_test])
    tr = [(u, y) for d in doms[n_test:] for u, y in by_dom[d]]
    te = [(u, y) for d in test_d for u, y in by_dom[d]]
    return tr, te


def _forward(model, batch, device):
    x, y = batch
    return model(x.to(device)), y.to(device).float()


def run_once(train_phish, test_phish, benign, cfg, device, tag):
    set_seed(cfg.train.seed)
    # Benign: split by domain, use disjoint halves for train and test.
    b_tr, b_te = _domain_split(benign, [0] * len(benign))
    Xtr_u = [u for u, _ in b_tr] + train_phish
    ytr = np.array([0] * len(b_tr) + [1] * len(train_phish))
    Xte_u = [u for u, _ in b_te] + test_phish
    yte = np.array([0] * len(b_te) + [1] * len(test_phish))

    def loader(urls, y, shuffle):
        X = torch.tensor(encode_batch(urls, cfg.url.max_len))
        return DataLoader(TensorDataset(X, torch.tensor(y)), batch_size=cfg.train.batch_size,
                          shuffle=shuffle)
    # small val slice from train
    n = len(ytr); idx = np.random.RandomState(0).permutation(n); nv = int(n * 0.1)
    vi, ti = idx[:nv], idx[nv:]
    tl = loader([Xtr_u[i] for i in ti], ytr[ti], True)
    vl = loader([Xtr_u[i] for i in vi], ytr[vi], False)
    el = loader(Xte_u, yte, False)

    enc = URLEncoder(embed_dim=cfg.url.embed_dim, num_filters=cfg.url.num_filters,
                     kernel_sizes=cfg.url.kernel_sizes, dropout=cfg.url.dropout, out_dim=cfg.url.out_dim)
    model = URLClassifier(enc).to(device)
    cfg.train.epochs = 4
    model, _ = train_epochs(model, tl, vl, device, cfg, _forward, lr=cfg.train.lr,
                            tag=tag, out_dir=Path("results"))

    @torch.no_grad()
    def score(loader):
        model.eval(); ss=[]; ys=[]
        for x, y in loader:
            ss.append(torch.sigmoid(model(x.to(device))).cpu().numpy()); ys.append(y.numpy())
        return np.concatenate(ss), np.concatenate(ys)
    s_va, y_va = score(vl); thr = threshold_for_fpr(y_va, s_va, 0.01)
    s_te, y_te = score(el)
    return compute(y_te, s_te, thr)


def main() -> int:
    cfg = Config(); device = resolve_device(cfg.train.device)
    raw = Path(cfg.data.raw_dir)
    op = _load(raw / "openphish_feed.txt")
    db = _load(raw / "phishing_links_active.txt")
    benign = [u for u in raw.joinpath("benign_urls.txt").read_text().splitlines() if u][:40000]
    import random
    random.Random(1).shuffle(db)
    db = db[:40000]
    print(f"OpenPhish: {len(op)}  Phishing.Database: {len(db)}  benign: {len(benign)}")

    # Cross-source: train on DB, test on OpenPhish (disjoint sources).
    print("[cross] train=Phishing.Database  test=OpenPhish")
    cross = run_once(db, op, benign, cfg, device, tag="xds_cross")
    # Within-source control: train and test both from DB (domain-disjoint).
    print("[within] train/test both Phishing.Database (domain-disjoint)")
    tr, te = _domain_split(db, [1] * len(db))
    within = run_once([u for u, _ in tr], [u for u, _ in te], benign, cfg, device, tag="xds_within")

    res = {"cross_source_train_db_test_openphish": cross, "within_source_db": within}
    print(f"\n  within-source  F1={within['f1']:.4f} ROC-AUC={within['roc_auc']:.4f} FPR={within['fpr']:.4f}")
    print(f"  cross-source   F1={cross['f1']:.4f} ROC-AUC={cross['roc_auc']:.4f} FPR={cross['fpr']:.4f}")
    Path("results/cross_dataset_results.json").write_text(json.dumps(res, indent=2))
    print("saved -> results/cross_dataset_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
