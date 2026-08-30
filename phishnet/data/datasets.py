"""Dataset construction with leakage control.

Two failure modes dominate published phishing benchmarks, and both inflate
scores enough to make comparisons meaningless:

1. Domain leakage. Feeds contain many URLs per host. A random row-level split
   puts sibling URLs from the same host on both sides, so the model can memorise
   hosts and still look like it generalises. We split on the registrable domain,
   so no domain is ever in more than one partition.

2. Structural artefacts. Pairing a phishing feed (full URLs) with a domain
   ranking list (bare hosts) makes URL length alone almost perfectly separating.
   We measured 98.9% vs 0.0% path presence on our own raw sources. The benign
   side is therefore drawn from harvested same-origin deep links, and
   `structural_report` re-measures the residual gap so it is stated, not hidden.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from torch.utils.data import Dataset

from phishnet.features.lexical import registrable_domain
from phishnet.features.url_encoding import encode_url


@dataclass
class Split:
    urls: list[str]
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.urls)

    @property
    def pos_rate(self) -> float:
        return float(self.labels.mean()) if len(self.labels) else 0.0


def _domain_bucket(domain: str, n_buckets: int = 1000) -> int:
    """Stable hash of a domain to a bucket.

    Hashing rather than shuffling means the split is reproducible across runs
    and machines without carrying a manifest file, and adding new data never
    moves an existing domain to a different partition.
    """
    h = hashlib.sha256(domain.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n_buckets


def domain_disjoint_split(urls: list[str], labels: np.ndarray,
                          val_frac: float = 0.15, test_frac: float = 0.15):
    """Partition by registrable domain so no domain spans two splits."""
    n_buckets = 1000
    val_cut = int(n_buckets * (1 - val_frac - test_frac))
    test_cut = int(n_buckets * (1 - test_frac))

    parts: dict[str, tuple[list[str], list[int]]] = {
        "train": ([], []), "val": ([], []), "test": ([], [])
    }
    for u, y in zip(urls, labels):
        b = _domain_bucket(registrable_domain(u) or u, n_buckets)
        key = "train" if b < val_cut else ("val" if b < test_cut else "test")
        parts[key][0].append(u)
        parts[key][1].append(int(y))
    return {k: Split(v[0], np.array(v[1], dtype=np.int64)) for k, v in parts.items()}


def apply_base_rate(split: Split, base_rate: float, seed: int = 1337) -> Split:
    """Down-sample positives to a target prevalence.

    A balanced test set reports precision as if every second page were phishing.
    Real browsing is nowhere near that, and precision is the metric a user
    actually experiences (it is the false-alarm rate). Evaluating at a realistic
    base rate is the difference between a usable warning and alert fatigue.
    """
    rng = random.Random(seed)
    pos = [i for i, y in enumerate(split.labels) if y == 1]
    neg = [i for i, y in enumerate(split.labels) if y == 0]
    if not pos or not neg:
        return split
    # Keep all negatives, sample positives to hit the target rate.
    n_pos = int(round(base_rate * len(neg) / max(1e-9, 1 - base_rate)))
    n_pos = min(n_pos, len(pos))
    keep = sorted(rng.sample(pos, n_pos) + neg)
    return Split([split.urls[i] for i in keep], split.labels[keep])


def structural_report(urls_pos: list[str], urls_neg: list[str]) -> dict:
    """Quantify how far a trivial classifier could get on structure alone."""
    def stats(us: list[str]) -> dict:
        lens, paths, depths = [], [], []
        for u in us:
            p = urlparse(u)
            lens.append(len(u))
            has = bool((p.path or "").strip("/")) or bool(p.query)
            paths.append(has)
            depths.append(len([x for x in (p.path or "").split("/") if x]))
        return {"mean_len": float(np.mean(lens)) if lens else 0.0,
                "has_path_rate": float(np.mean(paths)) if paths else 0.0,
                "mean_depth": float(np.mean(depths)) if depths else 0.0}

    a, b = stats(urls_pos), stats(urls_neg)
    # Best achievable accuracy from a single threshold on URL length --
    # the cheapest shortcut available to the model.
    L = np.array([len(u) for u in urls_pos + urls_neg])
    Y = np.array([1] * len(urls_pos) + [0] * len(urls_neg))
    best = 0.0
    for t in np.percentile(L, np.arange(1, 100)):
        # Try the threshold in both polarities and keep whichever separates better.
        acc = max(float(((L > t) == Y).mean()), float(((L <= t) == Y).mean()))
        best = max(best, acc)
    return {"phishing": a, "benign": b, "length_threshold_accuracy": best}


def build_url_corpus(cfg, use_deep_benign: bool = True, verbose: bool = True):
    """Assemble the URL corpus and split it without leakage."""
    raw = Path(cfg.data.raw_dir)
    rng = random.Random(cfg.train.seed)

    phish = [u for u in raw.joinpath("phish_urls.txt").read_text().splitlines() if u]
    deep = raw / "benign_deep_urls.txt"
    if use_deep_benign and deep.exists() and deep.stat().st_size > 0:
        benign = [u for u in deep.read_text().splitlines() if u]
        benign_src = "harvested same-origin deep links"
    else:
        benign = [u for u in raw.joinpath("benign_urls.txt").read_text().splitlines() if u]
        benign_src = "Tranco bare domains (ARTEFACT-PRONE)"

    rng.shuffle(phish)
    rng.shuffle(benign)
    phish = phish[: cfg.data.max_phish_urls]
    benign = benign[: cfg.data.max_benign_urls]

    if verbose:
        print(f"  phishing: {len(phish):,}   benign: {len(benign):,}  [{benign_src}]")
        rep = structural_report(phish[:20000], benign[:20000])
        print(f"  structural check -> phishing has_path={rep['phishing']['has_path_rate']:.3f} "
              f"benign has_path={rep['benign']['has_path_rate']:.3f}")
        print(f"  length-threshold-only accuracy: {rep['length_threshold_accuracy']:.3f}")

    urls = phish + benign
    labels = np.array([1] * len(phish) + [0] * len(benign), dtype=np.int64)
    splits = domain_disjoint_split(urls, labels, cfg.data.val_frac, cfg.data.test_frac)

    if verbose:
        for k, s in splits.items():
            print(f"  {k:5s} n={len(s):7,}  pos_rate={s.pos_rate:.3f}")
        # Assert the property we claim, rather than trusting the construction.
        doms = {k: {registrable_domain(u) for u in s.urls} for k, s in splits.items()}
        overlap = (doms["train"] & doms["test"]) | (doms["train"] & doms["val"])
        print(f"  domain overlap across splits: {len(overlap)} (expected 0)")
    return splits


class URLDataset(Dataset):
    def __init__(self, split: Split, max_len: int = 200):
        self.urls = split.urls
        self.labels = split.labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, i: int):
        return (torch.from_numpy(encode_url(self.urls[i], self.max_len)),
                torch.tensor(float(self.labels[i])))


class TextDataset(Dataset):
    """Tokenised message bodies. Tokenisation happens once, up front -- doing it
    in __getitem__ dominated epoch time on a corpus this small."""

    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_len: int = 256):
        enc = tokenizer(texts, truncation=True, padding="max_length",
                        max_length=max_len, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.input_ids[i], self.attention_mask[i], self.labels[i]


def build_text_corpus(cfg, verbose: bool = True, include_recent: bool = True):
    """Message-body corpus for the text branch.

    Two registers, combined so the classifier learns intent rather than surface
    style. (1) Classic email: phishing (Nazario, 2003-2007) vs benign
    (SpamAssassin ham, 2002). (2) Recent lures reflecting 2024-2026 scam themes
    (see `recent_lures.py`), added because the classic corpus predates the entire
    current threat vocabulary (MFA codes, tariff refunds, crypto drainers,
    package-redelivery fees, SIM-swap).

    A fixed fraction of the recent lures is held out as a `recent_test` split that
    the model never trains on, so we can report detection on current-theme
    messages specifically -- the honest measure of the upgrade. `recent_test` is
    additive; `train`/`val`/`test` keep the same meaning for every other caller.
    """
    raw = Path(cfg.data.raw_dir)
    sep = "\n<<<DOC>>>\n"
    phish = [t.strip() for t in raw.joinpath("phish_emails.txt").read_text().split(sep) if t.strip()]
    benign = [t.strip() for t in raw.joinpath("benign_emails.txt").read_text().split(sep) if t.strip()]

    rng = random.Random(cfg.train.seed)

    # Pull the recent lures and split off a held-out recent-only test set first,
    # before anything recent touches the training pool.
    recent_test = None
    r_phish_train, r_legit_train = [], []
    if include_recent:
        try:
            from phishnet.data.recent_lures import load_recent
            rec = load_recent(raw)
            rp, rl = rec["phish"], rec["legit"]
            rng.shuffle(rp)
            rng.shuffle(rl)
            hp, hl = int(len(rp) * 0.25), int(len(rl) * 0.25)  # 25% held out
            recent_test = (rp[:hp] + rl[:hl],
                           np.array([1] * hp + [0] * hl, dtype=np.int64))
            r_phish_train, r_legit_train = rp[hp:], rl[hl:]
        except Exception as e:  # never let the recent set break the core pipeline
            if verbose:
                print(f"  (recent lures unavailable: {e})")

    phish = phish + r_phish_train
    benign = benign + r_legit_train
    rng.shuffle(phish)
    rng.shuffle(benign)
    texts = phish + benign
    labels = np.array([1] * len(phish) + [0] * len(benign), dtype=np.int64)

    idx = list(range(len(texts)))
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(n * cfg.data.test_frac)
    n_val = int(n * cfg.data.val_frac)
    test_i, val_i, train_i = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]

    def take(ii):
        return [texts[i] for i in ii], labels[ii]

    out = {"train": take(train_i), "val": take(val_i), "test": take(test_i)}
    if recent_test is not None:
        out["recent_test"] = recent_test
    if verbose:
        print(f"  phishing (classic+recent) {len(phish):,}  benign {len(benign):,}")
        for k, (t, y) in out.items():
            print(f"  {k:11s} n={len(t):6,}  pos_rate={y.mean():.3f}")
    return out
