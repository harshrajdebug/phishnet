# PhishNet

**A Cross-Modal Attention Framework for Explainable Real-Time Phishing Detection**

PhishNet reads a suspicious page the way an analyst does — it looks at the URL,
at what the page *looks* like, and at what the message *says* — and it fuses
those three views with a cross-modal attention mechanism so that each signal is
interpreted in the context of the others. It then explains its verdict in plain
language and serves the whole thing behind a Chrome extension.

The design is a response to the 2026 threat landscape. APWG counted **971,181
unique phishing attacks in Q1 2026**, up 13.8% over the previous quarter, spread
across **766 distinct impersonated brands** ([APWG Q1 2026][apwg]). Detection
kits now cloak aggressively against automated crawlers — serving benign decoys
unless the visitor's referrer and TLS fingerprint match a real victim
([Netcraft][netcraft], [Abnormal][blacksite]) — which is precisely why PhishNet
is built to run **client-side, on the page the user actually sees**, rather than
from a scanner the kit can detect and dodge.

---

## Why multimodal, and why attention

Single-signal detectors each have a blind spot that attackers design around:

| Signal | What it catches | How it's evaded |
|---|---|---|
| URL blacklist | known-bad domains | a brand-new domain (zero-day) |
| URL lexical model | look-alike strings | a clean-looking URL on a compromised host |
| Visual match | brand clones | a pixel-perfect clone is *identical* to the real page |
| Text/NLP | lure language | a page with almost no text |

The visual case is the instructive one. A perfect clone of a bank login page is,
by construction, visually indistinguishable from the real page — so "does this
look like a bank" is not a useful question. **"Does this look exactly like Brand
X while being served from a domain that is not X's"** is. That question is
inherently *relational*: it can only be answered by reading the visual signal
against the URL signal. Naive fusion (concatenate the embeddings, classify)
forces each modality to be scored in isolation first, which throws that relation
away. Cross-modal attention keeps it: each modality's representation is allowed
to attend to — and be modulated by — the others before any decision is made.

This repository tests that claim directly: the fusion ablation trains
cross-attention and concatenation on the **same frozen encoders** and the
**same data**, so any gap is attributable to the fusion mechanism alone.

## Architecture

```
         URL  ──▶ Char 1D-CNN ─────┐
                                   │
   Screenshot ──▶ Siamese ResNet ──┼──▶ Cross-Modal Attention ──▶ verdict
                  + brand index    │      (+ gated pooling)         + gates
                                   │
      Message ──▶ DistilBERT ──────┘
                                            │
                       SHAP (URL) + Grad-CAM (visual) ──▶ explanation
```

- **URL branch** — character-level 1D-CNN with parallel kernel widths (3–6).
  Character-level because phishing hostnames are built from non-words
  (`paypa1-secure-verify`) where any word vocabulary is out-of-vocabulary exactly
  where the signal is.
- **Visual branch** — Siamese ResNet-18 trained with a triplet objective into a
  metric space where *same-brand* pages are close. Detection is a nearest-anchor
  lookup, so a new brand is added with **one reference screenshot, no
  retraining** — the zero-shot property that matters when 766 brands are hit per
  quarter.
- **Text branch** — DistilBERT (≈97% of BERT's accuracy at ~60% of the runtime,
  which the latency budget requires), mean-pooled, lower layers frozen.
- **Fusion** — two layers of cross-modal attention with learned modality-type
  embeddings, an explicit learned token for absent modalities, and gated pooling.
  **Modality dropout** during training stops the fused model from collapsing into
  a single-modality model with extra parameters.
- **Explainability** — SHAP over human-readable URL features and Grad-CAM over
  the screenshot (differentiating the cosine similarity to the matched brand, the
  quantity the decision actually rests on).
- **Reputation short-circuit** (serving only) — before the model runs, a URL
  served from a top-ranked *registrable* domain is passed as legitimate, exactly
  as Safe Browsing / SmartScreen do. Shared-hosting domains (`netlify.app`,
  `duckdns.org`, `000webhostapp.com`) are excluded from trust, so their abusive
  subdomains still reach the model. This is kept out of the reported metrics: the
  tables measure the model, not the allowlist.

Per-modality operating thresholds are calibrated on validation (`ablation.py`)
and selected at inference by which modalities are present, so a URL-only scan is
not judged against a threshold tuned for all three modalities.

## Data — and the artefact we refused to ship

All corpora are real and fetched live (`phishnet/data/acquire.py`):

| Role | Source | Size |
|---|---|---|
| Phishing URLs | OpenPhish live feed + Phishing.Database ACTIVE | ~789k |
| Benign URLs | Tranco top-1M | 200k |
| Phishing text | Nazario phishing corpus | 2,236 |
| Benign text | SpamAssassin public ham | 3,900 |
| Screenshots | rendered brand anchors + look-alike clones | see `render.py` |

The standard way to build this dataset — phishing feed vs. a domain ranking list
— produces a **fatal artefact**: 98.9% of phishing URLs carry a path while 0.0%
of the bare-domain benign URLs do. A single length threshold then scores ~99%,
and every "result" is really measuring that. We measured it, then fixed it:
`phishnet/data/harvest_benign.py` fetches each benign homepage once and extracts
same-origin deep links, giving the benign class the **same structural shape** as
the phishing class (147k benign deep URLs harvested). `datasets.py` re-measures
the residual gap on every run and prints it, so the shortcut cannot hide.

We also **never render live phishing pages** — they serve exploits and cloak
against headless crawlers. The visual positives are locally generated look-alike
login templates (brand colour + layout, no copyrighted logos), which keeps the
pipeline safe and lets us vary clone fidelity for the zero-shot experiment.

Splits are **domain-disjoint** (`domain_disjoint_split`): no registrable domain
ever appears in more than one partition, so the model cannot memorise hosts and
pass it off as generalisation. The property is asserted, not assumed.

## Metrics — why not accuracy

At a realistic 2% base rate, predicting "benign" always scores 98% accuracy. The
metrics that decide whether a browser extension is usable are **false-positive
rate at a target detection rate** (the alert-fatigue number) and **PR-AUC**.
Operating thresholds are always chosen on validation, never on test.

### Headline results

| Model / setting | F1 | ROC-AUC | PR-AUC | notes |
|---|---|---|---|---|
| URL: char 1D-CNN (vs RF 0.935 / LR 0.804) | 0.966 [0.964, 0.969] | 0.995 | 0.996 | 95% bootstrap CI; FPR\@TPR95 = 1.75% |
| Text on **recent lures**: before → after recent data | 0.70 → **1.00** | **0.31 → 1.00** | — | old model below chance on 2026 lures |
| Visual **zero-shot** (unseen brands): 10 → **24** brands | 0.59 → **0.78** | 0.85 → **1.00** | 0.74 → **1.00** | richer index ⇒ better generalisation |
| Modality ablation: URL-only → **all three** | 0.969 → **0.995** | → 1.000 | → 1.000 | multimodal gain, robust |
| Fusion mechanism (4 runs): cross-attn vs concat | 0.9949 ± .003 vs 0.9944 ± .001 | — | — | statistically **tied** |
| **Adversarial**: worst evasion (URL shortener) | — | — | — | 26.8% evade; 5 of 6 transforms resisted |
| **Cross-source** (train DB → test OpenPhish) | 0.75 | **1.00** | 1.00 | ranking transfers, threshold doesn't |

Two results carry the paper. **(1) Adding modalities helps robustly** — F1 rises
from 0.969 (URL alone) to 0.995 (all three), well outside noise. **(2) The fusion
*mechanism* does not matter**: a single run suggested cross-attention beat
concatenation 5.5× on FPR, but across four independent runs the two are
statistically indistinguishable (each wins half the runs). We report that negative
result rather than quoting the lucky run. Around them: a text model trained only on
pre-2009 email scores **below chance (ROC-AUC 0.31)** on current-theme lures and a
recent-lure set fixes it; the URL branch resists 5 of 6 evasions (shorteners are
the weakness the other modalities cover); and across sources the representation
transfers (ROC-AUC 1.0) though the threshold must be recalibrated. The honest
caveats — a synthetic visual corpus, current-*theme* (not captured) recent lures,
and assembled multimodal samples — are in the paper's Limitations, not hidden.

## Install & run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# full pipeline (acquire → harvest → render → train → benchmark → figures)
bash scripts/run_all.sh

# or a single stage
python -m phishnet.train.train_url
python -m phishnet.eval.benchmark
```

### Try it

```bash
python -m phishnet.cli demo
python -m phishnet.cli scan "http://paypa1-secure-verify.tk/account/login.php" \
    --text "Your account is suspended. Verify now."
```

### Browser extension (automatic, always-on)

The extension (`extension/`, MV3) runs continuously and needs no manual input:

1. **Start the local model server** (leave it running):
   ```bash
   bash scripts/serve.sh          # serves the API on 127.0.0.1:8000
   ```
2. **Load the extension once** in Chrome: `chrome://extensions` → enable
   **Developer mode** → **Load unpacked** → select the `extension/` folder.

From then on, every page you visit is scanned automatically. On navigation the
extension runs an instant URL-only scan (before the page paints); after load it
captures a screenshot and the visible text and runs the full multimodal scan. A
phishing page is **blocked with a full-screen interstitial** you must dismiss
consciously. The toolbar badge shows the live verdict (`✓` / `?` / `!`), and the
popup shows the modality gate weights, the reasons, server status, and toggles for
**always-on** and **aggressive** mode. Aggressive mode (default on) lowers the
decision bar to catch ~91% of phishing at a ~1.6% false-positive cost on obscure
sites; the top-10k domains are never flagged (reputation stage). If the server is
down the extension fails open — it never traps you offline.

## Tests

```bash
python -m pytest tests/ -q
```

The tests lock in the invariants that would silently invalidate results if
broken: leakage-free splitting, encoding round-trip, fusion missing-modality
masking, token-boundary brand matching, and metric behaviour under imbalance.

## Layout

```
phishnet/
  data/       acquire, harvest_benign, render, datasets, multimodal
  features/   url_encoding, lexical
  models/     url_cnn, text_encoder, visual_siamese, fusion, phishnet
  explain/    url_shap, gradcam
  train/      train_url, train_text, train_visual, train_fusion, common
  eval/       metrics, benchmark, make_figures
  serve/      inference, api
extension/    manifest + service worker + content script + popup
paper/        research paper + figures
```

## Authors

UPES School of Computer Science, B.Tech CSE (CSF) — Project group:
Harsh Raj (ML Pipeline & NLP), Aryan Kumar (Visual Module & Extension),
Ayush Prajapati (Explainability & UI), Ujjawal Jain (Data & Testing).
Mentor: Dr. Swati Rastogi.

[apwg]: https://docs.apwg.org/reports/apwg_trends_report_q1_2026.pdf
[netcraft]: https://www.netcraft.com/blog/detecting-cloaking-geofencing-evasion
[blacksite]: https://abnormal.ai/blog/blacksite-aitm-phishing-kit-cloaked-gg
