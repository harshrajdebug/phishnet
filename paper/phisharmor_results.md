# 4. Evaluation and Results

> Every figure below is reproducible from `results/`. §4.3 reports a completed
> **thirteen-seed** paired replication (`results/phishpedia_seeds_stats.json`);
> all paired contrasts are two-sided paired *t*-tests over seeds, $df=12$.

**Protocol note, stated once.** Three separate lexical runs exist with different
baselines and configurations. Mixing them would produce inconsistencies a careful
reader will catch, so each claim below cites one protocol and stays inside it:

| Protocol | Baseline | CTE estimator | Used for |
|---|---|---|---|
| **P1** mechanism race, 5 seeds | F1 0.9358, CTE 0.0642 | exhaustive $k\le3$ | mechanism comparison (§4.1.2) |
| **P2** dropout sweep, 5 seeds | F1 0.9364, CTE 0.0595 | exhaustive $k\le3$ | Pareto frontier (§4.1.3) |
| **P3** censoring-aware | F1 0.9358, KM 0.0650 | Kaplan–Meier | survival analysis (§4.1.4) |

All lexical gradient computations run on CPU. On MPS the baseline reliance
gradient returned all-NaN while defended arms stayed finite, and averaging over
seeds propagated that silently into a *completed* run that reported H1 and H2 as
`nan`. The evaluation harness now raises on non-finite reliance rather than
averaging it.

---

## 4.1 The accuracy tax and feature redundancy (lexical analysis)

### 4.1.1 The attack surface is where reliance meets cheapness

A detector's exposure is not its reliance on a feature, nor that feature's
manipulation cost, but the ratio. We test this directly. Input-gradient reliance
is only weakly concentrated on cheap features, $\mathrm{corr}(\text{reliance},
1/\text{cost}) = +0.201 \pm 0.031$ over five seeds, so an audit that stopped at
"does the model use cheap features?" would find little. But per-feature evasion
success tracks the ratio strongly: $\mathrm{corr}(\text{evasion},
\text{reliance}/\text{cost}) = \mathbf{+0.774 \pm 0.045}$.

The gap between $+0.201$ and $+0.774$ is the contribution. Reliance alone is a
weak predictor of exploitability; reliance priced by manipulation cost is a
strong one.

**Table 4.1. Most effective single-feature evasions (5 seeds, mean ± sd).**

| Feature | Evasion | Cost |
|---|---|---|
| `num_slashes` | 20.1% ± 1.9 | 0.020 |
| `path_depth` | 8.9% ± 1.5 | 0.020 |
| `hostname_length` | 7.5% ± 1.0 | 0.170 |
| `special_ratio` | 5.4% ± 2.2 | 0.028 |
| `hostname_entropy` | 5.2% ± 0.6 | 0.198 |
| `num_subdomains` | 4.9% ± 1.1 | 0.038 |

The two most damaging manipulations are also the two cheapest in the table.

### 4.1.2 The coercive defence and its accuracy tax (P1)

The intuitive defence is coercive: penalise gradient mass on cheap features and
force the model onto expensive ones. It works as designed and is unusable.

**Table 4.2. Mechanism comparison, five seeds (P1).**

| Mechanism | F1 | ΔF1 | CTE | ΔCTE | corr(reliance, 1/cost) |
|---|---|---|---|---|---|
| baseline | 0.9358 |, | 0.064 |, | +0.202 |
| gradient penalty | 0.8200 | **−12.37%** | 0.195 | +203.9% | **−0.498** |
| two-stage | 0.8167 | −12.73% | 0.190 | +196.1% | −0.336 |
| cost-budgeted TRADES | 0.9287 | −0.76% | 0.101 | +57.8% | +0.014 |
| **cost-weighted dropout** | **0.9305** | **−0.56%** | **0.313** | **+388.3%** | **+0.103** |

The result inverts the intuition. The gradient penalty drove reliance strongly
negative ($-0.498$): it genuinely relocated the model onto expensive features, as
instructed. It also destroyed 12.4% of clean F1 and still bought less
cost-to-evade than dropout. Cost-weighted dropout barely moved reliance at all
($+0.202 \to +0.103$) and delivered nearly twice the robustness for a 0.56%
accuracy cost.

**Suppressing reliance is therefore not the mechanism.** Depriving a model of a
cheap feature it needs is coercive: the feature was load-bearing, and removing
its gradient removes accuracy with it. Dropping the feature *stochastically*
leaves it available when present while forcing the model to build redundant
pathways that survive its absence. Robustness comes from redundancy, not from
abstinence; and the two are separable only because we measured reliance
directly and found the better mechanism barely changed it.

### 4.1.3 The Pareto frontier (P2)

Sweeping the dropout scale traces the trade-off. The frontier is monotone in
robustness and remarkably flat in accuracy at the low end.

**Table 4.3. Cost-weighted dropout frontier (P2, baseline F1 0.9364, CTE 0.0595).**

| Drop scale | ΔF1 | ΔCTE |
|---|---|---|
| 0.15 | **+0.01%** | +265.1% |
| 0.25 | −0.10% | +305.5% |
| 0.35 | −0.55% | +381.9% |
| **0.50** | **−0.88%** | **+456.7%** |
| 0.65 | −1.23% | +461.8% |

At a drop scale of 0.15 the accuracy change is $+0.01\%$. Statistically
indistinguishable from the baseline, and *positive in sign*, while cost-to-evade
rises 265%. There is no accuracy tax at all at that operating point. Our headline
configuration (0.50) accepts a 0.88% F1 reduction for a 457% increase in
cost-to-evade. Beyond 0.65 the frontier flattens: robustness saturates while
accuracy continues to erode.

The frontier is also insensitive to the cost weights themselves. Perturbing the
money/effort/conversion weights by ±10%, ±20% and ±30% moves median CTE only
between 0.222 and 0.225, so the result does not depend on precise pricing.

### 4.1.4 Survival analysis: the median is right-censored (P3)

Cost-to-evade is a time-to-event quantity, and a naive median is biased when
evasion fails within budget. Censoring is not negligible and it is *differential*:
9.4% for the baseline against 38.3% for cost-weighted dropout, the defence works
precisely by making evasion fail more often, so the arm we care about is the one
whose median is most understated. Comparing naive medians therefore understates
the defence.

**Table 4.4. Kaplan–Meier corrected cost-to-evade (P3).**

| Model | F1 | ΔF1 | KM median CTE | ΔCTE | Censored |
|---|---|---|---|---|---|
| baseline | 0.9358 |, | 0.065 |, | 9.4% |
| cost-weighted dropout | 0.9263 | −1.01% | 0.371 | +470.5% | 38.3% |
| cost-budgeted TRADES (β=2) | 0.9117 | −2.57% | 0.413 | **+534.6%** | 33.6% |

Under censoring-aware estimation the same defence is worth +470.5% rather than
+388.3%, and cost-budgeted TRADES buys the most robustness of any mechanism at a
2.57% accuracy cost. We report both the naive and corrected estimates because the
correction changes the ranking's magnitude, not its order.

---

## 4.2 Visual representation and the lazy-learning trap

### 4.2.1 The resolution reality, reported as a failure

Our first two visual experiments produced clean-looking tables that measured
nothing, and we report this because the failure is the finding.

On full-page screenshots downscaled to 224×224, the trained encoder reached 0.299
clean top-1 retrieval against a 0.0043 chance floor and **0.017 under cheap
attack**. Indistinguishable from noise. Worse, it was *less robust than doing
nothing*: untrained ImageNet features scored 0.032 under the same attacks, and the
trained model lost on every attack individually (JPEG 0.032 → 0.013, logo
occlusion 0.091 → 0.007) while doubling clean accuracy (0.134 → 0.299). Judged on
clean accuracy the baseline looked like it was learning.

The cause is representational. A 1366px-wide page carries a brand wordmark of
roughly 200×100 px; at 224×224 that becomes approximately **23×21 px**. Triplet
loss had no brand structure left to fit and fitted noise instead.

Cropping to the detected logo region rather than downscaling the page changes the
input and nothing else. On 3,162 logo crops over 78 brands, untrained ImageNet
features alone reach 0.717 clean against a 0.0169 chance floor. Standard
batch-hard triplet training, identical loss, backbone and optimiser to the run
that failed, then improves retrieval under **all nine attacks**, clean 0.656 →
0.829 and cheap-attack mean 0.556 → 0.766, with robustness rising monotonically
across 40 epochs and plateauing rather than collapsing.

**The architecture was never texture-hypersensitive.** The earlier conclusion that
metric learning destroys robustness was a property of the representation.

### 4.2.2 Corpus construction, and a leak that two gates missed

Deduplication is not optional here and the obvious key is insufficient. Phishpedia
samples carry a `family_id` phishing-kit hash, and splitting on it is sound: across
75 brands our gallery and query shared zero families. **37.3% of query crops were
nonetheless pixel-identical to a gallery crop.** `family_id` deduplicates the
*page*; distinct kits clone the *same official logo*, so the cropped artefact
repeats verbatim. Retrieval degenerated into looking up an identical image,
inflating clean top-1 from 0.668 to 0.859.

Removing exact-duplicate crops discards **61%** of what we harvest and is applied
throughout. The general lesson is that a semantically meaningful deduplication key
does not imply the artefact fed to the model is unique: fingerprint the tensors,
not the records.

### 4.2.3 Standard triplet loss is a lazy learner

The clearest evidence is a signed effect, not a small delta. On brands held out of
training entirely, standard triplet loss scores **0.377 ± 0.088 on grayscale inputs
while untrained ImageNet features score 0.619 ± 0.056**. A paired difference of
**−0.242 (t = −14.2, p < 10⁻⁵, losing on 13 of 13 seeds)**. The effect is not
confined to grayscale: on the cheap-attack mean the trained baseline also falls
below the untrained control (0.699 vs 0.732, **t = −5.95, p = 0.00007**, losing on
12 of 13 seeds). Training did not merely
fail to acquire colour-invariance, it *destroyed* robustness the initialisation
already had, reliably and on every seed.

This is what over-indexing on colour looks like. Colour is the cheapest
brand-discriminative signal in a logo, so a network minimising triplet loss over a
fixed brand set reaches for it, and a representation built on colour cannot
survive its removal. The same lazy strategy explains why the encoder transfers so
poorly: seen brands gain +0.204 clean and +0.224 attacked over the untrained
control, while unseen brands gain only **+0.029 clean and are +0.033 worse under
attack** (13 seeds; $p = 0.001$ and $p = 0.0001$ respectively). What training buys
is largely brand-specific separation rather than a transferable notion of a brand
mark.

---

## 4.3 Cost-aware augmentation in metric space

### 4.3.1 Design

All arms within a seed share initialisation, batch sequence and evaluation-time
attack draws, so the augmentation policy is the only difference and arm-to-arm
differences are paired. This is not a cosmetic choice. An unpaired single-seed run
put cost weighting **+0.068** ahead of uniform augmentation; the paired
thirteen-seed estimate of the same quantity is **+0.014**. Roughly four fifths of
the original margin was batch-order and evaluation-draw noise, and the effect that
remains is real but an order of magnitude smaller than the unpaired run implied.

The temperature $\tau = 0.06$ was selected once on validation brands and then
frozen; test brands are read once. A **uniform** arm samples the identical attacks
with cost weighting removed, because "augmentation helps" and "*cost-weighted*
augmentation helps" are different claims and only the second is ours.

**Table 4.5. Zero-shot test retrieval, 13 seeds (mean ± sd).**

| Arm | clean | attacked | grayscale | logo_delete |
|---|---|---|---|---|
| untrained control | 0.801 ± 0.037 | 0.732 ± 0.038 | 0.619 ± 0.056 | 0.427 ± 0.063 |
| baseline (no aug) | 0.831 ± 0.034 | 0.699 ± 0.036 | 0.377 ± 0.088 | 0.383 ± 0.048 |
| uniform augmentation | 0.815 ± 0.034 | 0.768 ± 0.032 | 0.609 ± 0.039 | 0.544 ± 0.054 |
| cost-weighted (τ=0.06) | 0.827 ± 0.036 | 0.782 ± 0.034 | 0.635 ± 0.052 | 0.373 ± 0.044 |

### 4.3.2 Semantic augmentation is necessary, and the effect is large

Both augmented arms beat the unaugmented baseline on zero-shot attacked retrieval
on **all 13 seeds**: uniform **+0.069 (t = 12.09, p < 10⁻⁵)**, cost-weighted
**+0.083 (t = 18.53, p < 10⁻⁵)**. On grayscale the gains are **+0.232** and
**+0.258**, both 13/13, both $p < 10⁻⁵$.

Read against §4.2.3 this is the paper's clearest visual claim. Standard metric
learning leaves an encoder *less* robust on unseen brands than no training at all;
semantic augmentation is what makes zero-shot visual robustness possible, and its
necessity is established far more strongly than any difference between
augmentation policies.

### 4.3.3 Colour dependency is repaired, not surpassed

Cost-weighted augmentation lifts grayscale retrieval on unseen brands from 0.377
to **0.635**. We are precise about the ceiling: 0.635 against the untrained
control's 0.619 is a difference of +0.016 with $p = 0.32$, statistically
indistinguishable even at 13 seeds. Augmentation **recovers** the colour
robustness that training destroyed; it does not exceed what the ImageNet
initialisation already supplied. Prediction 3 is confirmed in direction and
bounded in magnitude.

### 4.3.4 The mechanism margin, and what thirteen seeds were needed to see

Cost weighting beats uniform augmentation on zero-shot attacked retrieval by
**+0.0144 ± 0.0155**, winning on **11 of 13 seeds**, **t = 3.35, p = 0.006**. On
grayscale the margin is **+0.0263 ± 0.0429**, 10/13 seeds, **t = 2.21, p = 0.047**.
Both are significant; neither is large.

The path to this number is itself the methodological point. An unpaired single-seed
run reported +0.068. Pairing initialisation, batches and evaluation draws cut that
to +0.0099 at five seeds, where it was *not* separable from zero (p = 0.12) at 33%
power. Only at thirteen seeds. The sample size the observed effect size
$d_z = 0.93$ requires for 80% power, does the effect resolve, and it resolves
*upward*, to +0.0144. Two conclusions follow, and they point in opposite
directions. Four fifths of the original margin was an artefact of unpaired
evaluation. But the residue is real, and an analysis that had stopped at five
seeds would have wrongly reported a null.

We are correspondingly careful about magnitude. Cost weighting improves zero-shot
attacked retrieval by roughly **1.4 percentage points** over uniform augmentation, against the **6.9 points** that augmentation itself contributes over no
augmentation at all (§4.3.2). The dominant effect by a factor of five is that
attacks are simulated during training; *pricing* those attacks is a real but
second-order refinement on top of it. A reader deciding what to implement should
read that ordering as the finding.

This yields a **modality contrast** rather than a uniform endorsement. In the
discrete lexical setting cost weighting was not a refinement but a necessity: its
undifferentiated comparator, a uniform gradient penalty, cost **12.4%** of clean F1
(§4.1.2), a catastrophic price no amount of robustness would justify. In the
continuous visual embedding space the undifferentiated comparator is nearly free. Uniform augmentation costs 1.6 points of clean zero-shot retrieval, not
significantly different from baseline (p = 0.068); and cost weighting buys 1.4
points of attacked retrieval on top. The value of pricing manipulations is
therefore real in both modalities but differs by an order of magnitude in
importance, and we report where it is load-bearing and where it is a refinement.

Prediction 1 is **refuted**. Using the skill-normalised gap
$\frac{s-p_{\text{chance}}}{1-p_{\text{chance}}}$, required because the seen and
unseen tasks have different class counts, the seen/unseen gap does not narrow under
augmentation: baseline +0.037, cost-weighted +0.046, uniform +0.057. Augmentation
substantially improves unseen-brand robustness *under attack* without closing the
generalisation gap on clean retrieval, and we found no evidence that cost weighting
forces a representation that transfers better in the clean case.

### 4.3.5 The economic trade-off, measured

The place cost weighting differs most sharply from uniform augmentation is exactly
where the cost model says it should. On `logo_delete`, an attack that erases the
brand mark, uniform augmentation scores 0.544 and cost weighting 0.373, a paired
difference of **−0.172 (t = −10.36, p < 10⁻⁵, losing on 13 of 13 seeds)**.

The sharpest form of the result is a **null**. On `logo_delete`, cost-weighted
augmentation is statistically indistinguishable from *no augmentation whatsoever*
(−0.010, **p = 0.55**), while uniform augmentation gains +0.162 (p < 10⁻⁵). Cost
weighting spends essentially **nothing** defending an attack that destroys the
victim's own recognition cue; and nevertheless achieves *higher* aggregate
zero-shot robustness than the arm that spends 11% of its training budget there
(§4.3.4). The allocation is not merely defensible on threat-model grounds; on this
corpus it is also the better-performing one.

We state the cost of that allocation plainly. Against the untrained control,
cost-weighted augmentation is significantly *worse* on `logo_delete` (−0.054,
p = 0.027): the policy does not simply decline to improve there, it gives ground.
An adversary willing to abandon the brand cue entirely; and thereby most of the
campaign's conversion. Would find this model easier to evade than an
off-the-shelf ImageNet encoder. Whether that trade is correct depends entirely on
whether the conversion penalty in §2.2 reflects real attacker economics, which is
a modelling assumption and not something these experiments establish.

### 4.3.6 A structural caveat on cost-weighted evaluation

Attack cost and feature destructiveness are intrinsically coupled, which bounds
what any cost-weighted defence can claim. Measuring the trained encoder against
our price list gives $\mathrm{corr}(\text{cost}, \text{retained retrieval}) =
-0.765$ across all attacks and $-0.871$ across cheap ones: the model is most robust
exactly where attacks are cheapest. This is tempting to read as learned economic
sense, and we think that reading is wrong. A JPEG recompression is cheap *because*
the victim still recognises the logo, which necessarily means structural signal
survives for a detector to read; deleting the wordmark destroys the victim's cue
and the detector's evidence in one stroke. Cost and destructiveness are coupled by
construction, so a defence that reallocates capacity toward cheap attacks pushes on
an ordering the task already supplies.

This is the strongest argument against our own framing. It is why we evaluate on
zero-shot transfer rather than on robustness to cheap attacks; and, in light of
§4.3.4, why we report the visual cost model as a contrast with the lexical setting
rather than as a second demonstration of its superiority.

---

## 4.4 Limitations of visual-only detection

Cropping to the logo region is what made the visual task learnable, but it can
only act where a logo is found. Running Phishpedia's own released detector over its
own 30k benchmark, **4,208 of 29,496 samples (14.27%) yield no logo region at
all**, an empty coordinate file. Any method requiring a detected brand mark is
therefore capped at **85.7% recall on this corpus before a single evasion attack
is attempted**.

We state the bound precisely: it is a property of the detector-and-data pair, not
a theoretical limit of logo matching, and a stronger detector would move it. It
nonetheless bounds the entire visual arm, and it is the clearest quantitative
argument for why brand-mark matching must sit inside a multimodal system rather
than issue verdicts alone. On those 14% of pages the URL and text branches carry
the decision unaided: which is the architectural claim §4.1 and §4.2 jointly
support: the modalities fail in different places, and the cost structure that
governs their defence differs between them.
