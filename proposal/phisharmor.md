# PhishArmor: Cost-Aware Adversarial Robustness for Phishing Webpage Detection

**A Research Proposal**

**Harsh Raj, Aryan Kumar, Ayush Prajapati, Ujjawal Jain**
School of Computer Science, University of Petroleum and Energy Studies, Dehradun, India
Mentor: Dr. Swati Rastogi

---

## Executive Summary

Visual phishing detectors published at top security venues collapse to near-zero
detection under manipulations an attacker can apply for free. Deleting the brand
logo drives PhishIntention to **0%** detection; changing a font drives it to
**2.15%**; a gradient-based perturbation to **3.64%** [1]. A parallel line of work
shows that **over 80% of successful evasions concentrate on just three low-cost
features**, meaning robustness is governed by the economics of the attack surface
rather than by model capacity [2]. Meanwhile the newest state-of-the-art system,
PhishAgent (AAAI 2025), evaluates robustness only against prompt injection and
typosquatting, explicitly setting visual attacks aside as *"less common due to
their visibility"* [3] — an assumption the evidence in [1] contradicts.

The field has produced many attacks and **no defence**. The authors of [1] state
the gap directly: there is *no defence against coordinated manipulation of logos,
layouts, and textual content simultaneously.*

**PhishArmor** proposes to fill it. The central claim is that a detector's
robustness is determined not by its capacity but by *which features it relies on,
weighted by what those features cost an attacker to forge*. We contribute (i) a
formal attacker-cost model that prices each evasion in money, effort **and lost
victim conversion**; (ii) a defence combining cost-weighted adversarial training
with a penalty on reliance upon cheap features; and (iii) **RobustPhishBench**, the
first cost-parameterised robustness benchmark evaluating Phishpedia,
PhishIntention, PhishAgent and PhishArmor on equal terms, reporting a new metric —
**median cost-to-evade** — in place of the conventional robust-accuracy-at-epsilon.

---

## 1. Background and Motivation

Phishing remains the dominant initial-access vector, and the detection literature
has converged on three families:

| Family | Representatives | Weakness under attack |
|---|---|---|
| Reference-based visual | Phishpedia, PhishIntention | Depends on logo presence and appearance |
| Screenshot similarity | VisualPhishNet, EMD | Brand identification 7–26% on real data |
| Multimodal / agentic | PhishAgent, KnowPhish | Untested against visual manipulation |

Every family shares an unexamined assumption: that the visual and structural
signals it reads are *stable*. An adversary chooses those signals. The question
that matters operationally is therefore not "how accurate is the detector on a
static benchmark" but **"what does it cost an attacker to make this detector
wrong, and can we raise that price?"**

### 1.1 Why the assumption fails

The robustness evaluation in [1] applies thirteen visible manipulations plus
white- and black-box perturbations to three logo-based and two screenshot-based
detectors. The results are not graceful degradation; they are collapse:

| Attack | Effect on PhishIntention |
|---|---|
| Logo deletion | detection → **0%** |
| Font change | detection → **2.15%** |
| Location shift | detection → **2.73%** |
| PGD / FGSM (white-box) | detection → **3.64%** |
| Unseen brands | **13–18%** absolute drop |
| Real-world vs curated data | **20.7%** decline |

### 1.2 Why no defence exists yet

Three structural reasons, each of which our design addresses:

1. **Attacks are cheaper to publish than defences.** Generating an adversarial
   logo is a weekend project; a defence must hold across attack families.
2. **Robust-accuracy-at-epsilon does not fit this threat model.** An `L∞` ball is
   the wrong abstraction for "delete the logo" or "shorten the URL". Without a
   metric that captures *semantic* attacks, defences cannot be compared.
3. **Reference-based detectors have no training loop to harden.** Phishpedia is
   deliberately not trained on phishing data, so standard adversarial training
   does not apply to it. A defence must therefore operate at the *representation*
   and *fusion* level, not only at the classifier.

---

## 2. Research Gap and Questions

**Gap.** No published defence addresses coordinated, semantic, low-cost
manipulation of phishing webpages, and no benchmark prices attacks by what they
cost the adversary.

We ask:

- **RQ1** — Can evasions be assigned a principled cost that predicts which
  attacks adversaries actually use in the wild?
- **RQ2** — Does training a detector with perturbations sampled *inversely
  proportional to attacker cost* yield greater robustness per unit of clean
  accuracy lost than uniform adversarial training?
- **RQ3** — Does explicitly penalising reliance on cheap features raise the
  median cost-to-evade without a proportional loss of clean performance?
- **RQ4** — Under a unified cost-parameterised attack suite, how do Phishpedia,
  PhishIntention, PhishAgent and PhishArmor compare?

**Hypothesis.** Robustness follows feature economics. A detector regularised away
from cheap features will trade a small amount of clean accuracy for a large
increase in cost-to-evade, and this trade will dominate uniform adversarial
training on the robustness–accuracy frontier.

---

## 3. Proposed Approach

### 3.1 The attacker-cost model

Each evasion `a` receives a cost vector `c(a) = (c_money, c_effort, c_conversion)`:

- **c_money** — registration, hosting, certificates, proxy or residential IP rental
- **c_effort** — engineering time and skill, whether kit-automatable
- **c_conversion** — *the novel axis*: how much the manipulation reduces the
  victim's likelihood of being deceived. Deleting the brand logo defeats a
  logo-based detector at zero monetary cost but also removes the visual cue the
  victim relies on, so the attack is partly self-defeating.

Scalar cost is a weighted sum, `cost(a) = wᵀc(a)`, with `w` calibrated against
observed frequencies in live feeds (OpenPhish, Phishing.Database) — an attack that
is cheap under the model should be common in the wild. **That calibration is
RQ1's falsifiable test.**

### 3.2 Defence mechanism 1 — cost-weighted adversarial training

Rather than sampling perturbations uniformly, sample the attack applied to each
training example with probability inversely proportional to its cost:

```
P(a) ∝ exp( −cost(a) / τ )

for (x, y) in loader:
    a      ← sample_attack(P)          # cheap attacks dominate
    x_adv  ← a(x)
    loss   ← CE( f(x_adv), y ) + λ · reliance_penalty(f, x)
```

The temperature `τ` interpolates between uniform adversarial training (`τ → ∞`)
and training exclusively on the single cheapest attack (`τ → 0`), giving a clean
ablation axis.

### 3.3 Defence mechanism 2 — cheap-feature reliance penalty

Robustness is a property of *what the model attends to*. We penalise input-gradient
mass on features that are cheap to manipulate:

```
g              = ∂f(x) / ∂x
reliance_penalty = Σᵢ  (1 / cost(featureᵢ)) · gᵢ²
```

This pushes the decision onto expensive invariants — domain registration and
infrastructure characteristics, TLS and hosting provenance, and cross-modal
consistency between the rendered brand identity and the serving domain, which an
attacker cannot forge without actually controlling the brand's domain.

### 3.4 Defence mechanism 3 — expensive-invariant anchored fusion

Our prior work established that a benign-looking modality can *veto* a malicious
one when fusion is trained on class-consistent data. PhishArmor generalises the
fix: the fusion layer is trained so that **the cheapest modality can never
override the most expensive one**, enforced by a monotonicity constraint on the
gate assigned to low-cost modalities.

### 3.5 RobustPhishBench and the cost-to-evade metric

We replace robust-accuracy-at-epsilon with an economically meaningful metric:

```
CTE(model, x) = min { cost(a) : model( a(x) ) = benign }
```

The **median cost-to-evade** across a test set, and the full CTE distribution,
answer the question a defender actually has: *how much must an adversary spend to
get past this system?* We report CTE alongside conventional metrics so results
remain comparable to prior work.

---

## 4. Implementation Architecture

### 4.1 System overview

```
                        ┌───────────────────────────────┐
                        │        DATA LAYER             │
                        │  Phishpedia 30k (real shots)  │
                        │  PhishIntention 50k pages     │
                        │  OpenPhish live feed          │
                        │  Tranco benign + deep links   │
                        └───────────────┬───────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
┌───────▼────────┐           ┌──────────▼─────────┐          ┌──────────▼────────┐
│   COST MODEL   │           │   ATTACK SUITE     │          │  DETECTOR ZOO     │
│ taxonomy.py    │──prices──▶│ visual.py          │─attacks─▶│ phishpedia_wrap   │
│ estimator.py   │           │ perturbation.py    │          │ phishintention_wr │
│ conversion.py  │           │ html.py / url.py   │          │ phishagent_wrap   │
│                │           │ compose.py (multi) │          │ phisharmor.py     │
└───────┬────────┘           └──────────┬─────────┘          └──────────┬────────┘
        │                               │                               │
        │  P(a) ∝ exp(−cost/τ)          │                               │
        └───────────────┬───────────────┘                               │
                        │                                               │
              ┌─────────▼──────────┐                          ┌─────────▼────────┐
              │   DEFENCE TRAINER  │                          │  BENCH RUNNER    │
              │ cost_weighted_at   │─────── trains ──────────▶│ runner.py        │
              │ reliance_penalty   │                          │ metrics.py (CTE) │
              │ anchor_fusion      │                          │ report.py        │
              └────────────────────┘                          └─────────┬────────┘
                                                                        │
                                                              ┌─────────▼────────┐
                                                              │ RESULTS          │
                                                              │ CTE curves,      │
                                                              │ robustness       │
                                                              │ frontier, CIs    │
                                                              └──────────────────┘
```

### 4.2 Repository structure

```
phisharmor/
├── configs/
│   ├── default.yaml               # paths, seeds, hyper-parameters
│   ├── cost_weights.yaml          # w for (money, effort, conversion)
│   └── attacks.yaml               # attack registry + parameters
├── phisharmor/
│   ├── cost/
│   │   ├── taxonomy.py            # Evasion enum + cost vectors
│   │   ├── estimator.py           # cost(a) -> scalar; calibration to feed data
│   │   └── conversion.py          # victim-conversion penalty model
│   ├── attacks/
│   │   ├── base.py                # Attack protocol: apply(sample) -> sample
│   │   ├── visual.py              # logo delete/blur/shift/recolour/font swap
│   │   ├── perturbation.py        # FGSM, PGD, black-box ViT transfer
│   │   ├── html.py                # DOM mutation, prompt injection, obfuscation
│   │   ├── url.py                 # shortener, TLD swap, homoglyph, padding
│   │   └── compose.py             # coordinated multi-component attacks
│   ├── defense/
│   │   ├── cost_weighted_at.py    # sampler + training loop
│   │   ├── reliance_penalty.py    # input-gradient regulariser
│   │   └── anchor_fusion.py       # monotone gate constraint
│   ├── models/
│   │   ├── base.py                # Detector protocol (uniform interface)
│   │   ├── phishpedia_wrap.py     # subprocess/API adapter
│   │   ├── phishintention_wrap.py
│   │   ├── phishagent_wrap.py
│   │   └── phisharmor.py          # our detector (extends PhishNet)
│   ├── bench/
│   │   ├── runner.py              # orchestration, caching, resumability
│   │   ├── metrics.py             # CTE, robust acc, frontier, bootstrap CI
│   │   └── report.py              # tables + figures
│   └── data/
│       ├── loaders.py             # Phishpedia / PhishIntention adapters
│       └── splits.py              # brand-disjoint + domain-disjoint splits
├── scripts/
│   ├── fetch_benchmarks.sh        # pull 30k/50k datasets + weights
│   ├── run_bench.py
│   └── train_defense.py
├── tests/
└── paper/
```

### 4.3 The uniform detector interface

Every detector — ours and each baseline — is wrapped behind one protocol, so the
benchmark treats them identically and adding a future system costs one file:

```python
class Detector(Protocol):
    name: str
    def predict(self, sample: Sample) -> Verdict:
        """Sample carries url, html, screenshot; Verdict carries
        label, score, and optionally the identified target brand."""
```

`Sample` is the single currency of the system: attacks map `Sample → Sample`,
detectors map `Sample → Verdict`. This keeps attacks, detectors and metrics fully
decoupled and independently testable.

### 4.4 Key algorithm — cost-to-evade search

CTE is a constrained minimisation over a discrete attack space, solved greedily
with a cost-ordered frontier (exact for single attacks, near-optimal for
compositions):

```
def cost_to_evade(model, sample, attacks, budget):
    frontier = PriorityQueue()            # ordered by cumulative cost
    frontier.push(cost=0, applied=[], s=sample)
    seen = set()
    while frontier and frontier.peek().cost <= budget:
        node = frontier.pop()
        if model.predict(node.s).label == "benign":
            return node.cost, node.applied      # cheapest successful evasion
        for a in attacks:
            if a in node.applied: continue      # no repeats
            child = a.apply(node.s)
            frontier.push(node.cost + cost(a), node.applied + [a], child)
    return budget, None                          # censored observation
```

Runs that exhaust the budget are recorded as **censored**, and the CTE
distribution is estimated with survival analysis (Kaplan–Meier) rather than by
discarding them — this avoids the bias that would arise from dropping the most
robust samples.

### 4.5 Technology stack

| Layer | Choice | Rationale |
|---|---|---|
| Modelling | PyTorch 2.x + torchvision | Baselines are PyTorch; MPS/CUDA portable |
| Detector baselines | Phishpedia, PhishIntention (official repos) | Reproducibility; both ship CPU setup |
| Eval scaffolding | PhishingEval harness | Existing multi-detector evaluation code |
| Vision backbones | ResNet-18/50, ViT-B/16 | Matches baselines; ViT for transfer attacks |
| Text | DistilBERT / MobileBERT | Latency budget for client-side operation |
| Attacks | torchattacks + custom semantic ops | Standard `Lp` attacks plus our semantic suite |
| Rendering | Playwright (headless Chromium) | Deterministic screenshots for manipulated DOM |
| Statistics | scipy, lifelines | Bootstrap CIs and censored CTE estimation |
| Orchestration | Hydra configs, DVC or plain manifests | Reproducible sweeps |

### 4.6 Reuse from existing PhishNet work

A substantial fraction of the scaffolding already exists and transfers directly:

| Existing asset | Role in PhishArmor |
|---|---|
| Adversarial evaluation harness (six URL transforms) | Seeds `attacks/url.py` |
| Multimodal fusion with gating and modality dropout | Base for `anchor_fusion.py` |
| Domain-disjoint splitting, bootstrap CIs, multi-seed protocol | `bench/metrics.py` discipline |
| Cross-class assembly fix for the text-veto failure | Precursor to §3.4; a motivating result |
| Live OpenPhish / Tranco ingestion | Cost-model calibration data |
| FastAPI serving + activity dashboard | Field-study instrumentation |

---

## 5. Experimental Design

### 5.1 Datasets

| Dataset | Scale | Use |
|---|---|---|
| Phishpedia benchmark | 30k phishing + 30k benign, real screenshots | Primary evaluation |
| PhishIntention | ~50k webpages | Cross-dataset generalisation |
| OpenPhish + Phishing.Database | live, continuous | Cost-model calibration, temporal drift |
| Tranco + harvested deep links | 200k | Benign control, false-positive measurement |

Splits are **brand-disjoint** (no brand in both train and test) as well as
domain-disjoint, so zero-shot brand robustness is measured honestly.

### 5.2 Baselines

Phishpedia, PhishIntention, VisualPhishNet, PhishAgent, plus URL-only references
(URLNet, Random Forest on lexical features) for the URL attack family.

### 5.3 Metrics

- **Clean** — accuracy, precision, recall, F1, ROC-AUC, PR-AUC
- **Robust** — detection rate under each attack family, and under composition
- **Economic** — median cost-to-evade, CTE survival curves, cost-at-90%-evasion
- **Trade-off** — the clean-accuracy / CTE frontier as `λ` and `τ` vary
- **Deployment** — false-positive rate at a realistic base rate; latency

All headline numbers carry **bootstrap 95% confidence intervals** and are reported
across **≥ 5 seeds**, following the protocol already established in our prior work.

### 5.4 Ablations

Cost weighting on/off · reliance penalty on/off · `λ` sweep · `τ` sweep · each
cost axis removed in turn (notably `c_conversion`, to test whether the novel term
earns its place) · anchored fusion on/off.

---

## 6. Timeline

| Phase | Weeks | Deliverable |
|---|---|---|
| 0 — Setup | 1–3 | Baselines reproduced; benchmark data pulled; detector wrappers pass smoke tests |
| 1 — Cost model + attacks | 4–10 | Attack suite implemented; cost model calibrated against live-feed frequencies (**RQ1**) |
| 2 — Defence | 11–22 | Cost-weighted AT, reliance penalty, anchored fusion; ablations (**RQ2, RQ3**) |
| 3 — Benchmark | 23–30 | RobustPhishBench across all detectors; CTE curves; CIs (**RQ4**) |
| 4 — Writing | 31–36 | Manuscript, artefact release, submission |

Phases 1 and 2 overlap deliberately: the attack suite must be stable before the
defence trains against it, but cost calibration can continue in parallel.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Adversarial training costs too much clean accuracy | High | Report the frontier honestly; the contribution is the *shape* of the trade-off, not a free lunch |
| Baseline reproduction (PhishIntention) proves brittle | Medium | Use the PhishingEval harness; failing that, report on the baselines that do reproduce and document the rest |
| A competing defence is published mid-project | Medium | RobustPhishBench and the CTE metric stand independently; pivot emphasis to the benchmark |
| Compute limits at 30k-scale adversarial training | Medium | Subsample with stratification; free Colab/Kaggle GPU; cache attacked samples on disk |
| Cost model judged arbitrary by reviewers | Medium | Calibrate against observed wild frequencies and publish the calibration as a falsifiable test |

---

## 8. Expected Contributions

1. **The first defence** against coordinated, low-cost, semantic manipulation of
   phishing webpages — filling a gap stated explicitly in [1].
2. **A formal attacker-cost model** including victim-conversion cost, calibrated
   against live phishing telemetry.
3. **Cost-to-evade**, an economically meaningful robustness metric with censored
   estimation, replacing robust-accuracy-at-epsilon for semantic threat models.
4. **RobustPhishBench**, a reusable open benchmark evaluating four detector
   families on equal terms.
5. An **empirical correction** to the assumption in [3] that visual attacks are
   impractical.

### Target venues

| Tier | Venue | Fit |
|---|---|---|
| Q1 journal | *Computers & Security* (Elsevier) | Strong — systems + evaluation focus |
| Q1 journal | *IEEE TIFS* / *TDSC* | Strong if the defence result is decisive |
| A\* conference | USENIX Security, ACM CCS | Reach; the benchmark strengthens the case |
| Fallback | *DTRAP*, ESORICS, DIMVA | Solid, realistic |

---

## 9. Resources Required

- **Compute** — one CUDA GPU (16 GB) for adversarial training at full scale; the
  Apple-silicon development machine suffices for the pipeline and reduced-scale runs
- **Storage** — ~150 GB for benchmark datasets, cached attacked samples, checkpoints
- **Data access** — all datasets are public; no partnership required
- **Ethics** — no live phishing pages are served or visited; all manipulation is
  applied offline to archived samples. No human-subject data is collected.

---

## References

[1] Evaluating the Effectiveness and Robustness of Visual Similarity-based
Phishing Detection Models, 2024. arXiv:2405.19598

[2] Robustness, Cost, and Attack-Surface Concentration in Phishing Detection,
2026. arXiv:2603.19204

[3] Cao et al., PhishAgent: A Robust Multimodal Agent for Phishing Webpage
Detection, AAAI 2025. arXiv:2408.10738

[4] Lin et al., Phishpedia: A Hybrid Deep Learning Based Approach to Visually
Identify Phishing Webpages, USENIX Security 2021.

[5] Liu et al., Inferring Phishing Intention via Webpage Appearance and Dynamics
(PhishIntention), USENIX Security 2022.

[6] Abdelnabi et al., VisualPhishNet: Zero-Day Phishing Website Detection by
Visual Similarity, ACM CCS 2020.

[7] Zhang et al., CrawlPhish: Large-scale Analysis of Client-side Cloaking
Techniques in Phishing, IEEE S&P 2021.

[8] Attacking Logo-Based Phishing Website Detectors with Adversarial
Perturbations, ESORICS 2023.

[9] PhishOracle: Generating Adversarial Phishing Web Pages, 2024.

[10] From ML to LLM: Evaluating the Robustness of Phishing Web Page Detection
Models against Adversarial Attacks, ACM DTRAP, 2025.
