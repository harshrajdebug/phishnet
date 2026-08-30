"""Assembly of aligned multimodal samples for fusion training.

The honest problem: no public corpus gives all three modalities observed
*together* for the same attack. OpenPhish gives URLs, Nazario gives lure text,
and screenshots must be rendered. So a multimodal sample has to be assembled.

We do this by *class-consistent assembly*: a phishing sample draws its URL, its
lure text and its screenshot each from the phishing side of a real corpus; a
benign sample draws all three from benign corpora. This is defensible because in
a real campaign the malicious URL, the malicious lure and the cloned page ARE
parts of one attack -- they are simply collected through different sensors. What
the assembly cannot claim is that a *specific* observed triple co-occurred; it
claims only that each component is real and correctly labelled.

Two consequences we design around rather than hide:
  * The task must not be solvable by any single modality, or fusion is untested.
    Modality dropout (in the fusion layer) and the per-modality ablation in the
    benchmark both guard against this.
  * A fraction of samples are deliberately left partial (a modality marked
    absent) to match deployment, where the extension often has a URL and a
    screenshot but no email context.

This is stated plainly in the paper's limitations. It is the standard fallback
in multimodal phishing work when naturally-aligned data does not exist, and the
unimodal results (trained/tested on fully real, un-assembled data) remain the
load-bearing empirical claims.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MMSample:
    url: str
    text: str | None
    image: str | None      # screenshot filename, or None if absent
    label: int
    present: tuple[bool, bool, bool]   # (url, visual, text)


def assemble(cfg, n_per_class: int = 6000, seed: int = 1337,
             partial_frac: float = 0.35, verbose: bool = True):
    """Build train/val/test lists of MMSample with domain-disjoint URL splits."""
    from phishnet.data.datasets import domain_disjoint_split, Split
    from phishnet.features.lexical import registrable_domain

    raw = Path(cfg.data.raw_dir)
    sd = Path(cfg.data.screenshot_dir)
    rng = random.Random(seed)

    phish_urls = [u for u in raw.joinpath("phish_urls.txt").read_text().splitlines() if u]
    deep = raw / "benign_deep_urls.txt"
    benign_urls = [u for u in (deep if deep.exists() else raw / "benign_urls.txt")
                   .read_text().splitlines() if u]
    sep = "\n<<<DOC>>>\n"
    phish_txt = [t.strip() for t in raw.joinpath("phish_emails.txt").read_text().split(sep) if t.strip()]
    benign_txt = [t.strip() for t in raw.joinpath("benign_emails.txt").read_text().split(sep) if t.strip()]
    # Fold in the recent-lure set so the fused model, like the standalone text
    # branch, sees current-theme message vocabulary rather than only 2000s email.
    try:
        from phishnet.data.recent_lures import load_recent
        rec = load_recent(raw)
        phish_txt += rec["phish"]
        benign_txt += rec["legit"]
    except Exception:
        pass

    manifest = json.loads((sd / "manifest.json").read_text())
    phish_imgs = [s["path"] for s in manifest["samples"] if s["label"] == 1]
    benign_imgs = [s["path"] for s in manifest["samples"] if s["label"] == 0]

    rng.shuffle(phish_urls); rng.shuffle(benign_urls)
    rng.shuffle(phish_txt); rng.shuffle(benign_txt)

    def make(urls, texts, imgs, label, other_texts, other_imgs, mismatch_frac=0.25):
        out = []
        for i in range(min(n_per_class, len(urls))):
            # Every sample always has a URL (the one modality always present at
            # inference). Text and visual are present with probability, and
            # `partial_frac` of samples drop at least one to model deployment.
            has_text = rng.random() > 0.15
            has_vis = rng.random() > 0.15
            if rng.random() < partial_frac:
                # force at least one modality absent
                if rng.random() < 0.5:
                    has_text = False
                else:
                    has_vis = False
            # Cross-class assembly. In a fraction of samples the text and/or the
            # screenshot are drawn from the *opposite* class while the label stays
            # with the URL. This is the fix for a concrete failure of pure
            # class-consistent assembly: with URL and text always agreeing, the
            # fusion learns text is decisive and lets a benign-looking message veto
            # a malicious URL (false negative) or a lure-worded message condemn a
            # benign URL (false positive). Teaching the model that the destination
            # (URL/visual) sets the label while text merely supports it removes the
            # text-veto behaviour. The URL is never swapped -- it is the ground
            # truth for whether a click is dangerous.
            t_pool, v_pool = texts, imgs
            if rng.random() < mismatch_frac:
                if rng.random() < 0.7 and other_texts:
                    t_pool = other_texts          # opposite-class text, URL label kept
                elif other_imgs:
                    v_pool = other_imgs
            out.append(MMSample(
                url=urls[i],
                text=(rng.choice(t_pool) if has_text else None),
                image=(rng.choice(v_pool) if has_vis else None),
                label=label,
                present=(True, has_vis, has_text),
            ))
        return out

    phish = make(phish_urls, phish_txt, phish_imgs, 1, benign_txt, benign_imgs)
    benign = make(benign_urls, benign_txt, benign_imgs, 0, phish_txt, phish_imgs)
    all_samples = phish + benign

    # Split on the URL's registrable domain so the same domain never crosses
    # partitions -- identical leakage discipline to the unimodal URL split.
    urls = [s.url for s in all_samples]
    labels = np.array([s.label for s in all_samples])
    idx_splits = domain_disjoint_split(urls, labels, cfg.data.val_frac, cfg.data.test_frac)
    url_to_split = {}
    for name, sp in idx_splits.items():
        for u in sp.urls:
            url_to_split[u] = name

    splits: dict[str, list[MMSample]] = {"train": [], "val": [], "test": []}
    for s in all_samples:
        splits[url_to_split.get(s.url, "train")].append(s)

    if verbose:
        for k, v in splits.items():
            pos = sum(s.label for s in v)
            vis = sum(s.present[1] for s in v)
            txt = sum(s.present[2] for s in v)
            print(f"  {k:5s} n={len(v):6,} pos={pos/max(len(v),1):.3f} "
                  f"visual_present={vis/max(len(v),1):.3f} text_present={txt/max(len(v),1):.3f}")
    return splits
