# PhishArmor: Adversarial Robustness for Phishing Detection as an Economic Problem

<div class="byline">
<b>Harsh Raj, Aryan Kumar, Ayush Prajapati, Ujjawal Jain</b><br/>
School of Computer Science, University of Petroleum and Energy Studies, Dehradun, India<br/>
Mentor: Dr. Swati Rastogi
</div>

---


# Abstract

Adversarial robustness for phishing detection is usually posed geometrically, as
invariance inside an $L_p$ ball. We argue it is better posed economically. The
manipulations a phishing kit performs are semantic rather than norm-bounded, and
some evasions defeat the attacker along with the detector: removing the brand
token from a URL beats brand matching for free, and removes the cue the victim
needs in order to be deceived. We price 30 lexical and 8 visual manipulations on
three axes — money, engineering effort, and *victim conversion* — and evaluate
defences by cost-to-evade. The third axis carries the framework: ablating it moves
the three deception-bearing features from ranks 3–5 (among the cheapest) to 27–29
of 30, which is the difference between pricing what an attacker *can* do and what
an attacker *will* do.

Our results are substantially corrective. Feature reliance alone barely predicts
exploitability (r = +0.201 ± 0.031), while reliance *priced by cost* predicts it
strongly (r = +0.774 ± 0.045), so reliance-only auditing is a blind spot. The
intuitive defence — penalising gradient mass on cheap features — relocates the
model onto expensive features exactly as instructed, costs **12.4% of clean F1**,
and still buys less robustness than cost-weighted stochastic *dropout*, which
barely perturbs reliance and yields **+457% cost-to-evade for 0.88% F1**; at a
gentler setting the accuracy change is **+0.01%** for +265%. Robustness here comes
from redundancy, not abstinence.

In the visual modality, standard metric learning is a lazy learner. It
over-indexes on colour and leaves the encoder *less* robust on unseen brands than
untrained ImageNet features (grayscale −0.242, t = −14.2, p < 10⁻⁵, losing on
all 13 seeds). Semantic augmentation repairs this (+0.069 zero-shot attacked
retrieval, p < 10⁻⁵ over 13 paired seeds), and weighting that augmentation by
attacker cost adds a further **+0.014 (p = 0.006)** — real, but a fifth the size of
the augmentation effect itself, and visible only after pairing and thirteen seeds:
the same quantity read +0.068 unpaired and +0.010 at five seeds. Cost weighting is
also the more *efficient* allocation: on brand-erasing attacks it is
indistinguishable from no augmentation at all (p = 0.55), spending nothing on
manipulations that blind the victim while still achieving higher aggregate
robustness than the arm that does.

Two methodological contributions support these results. Experiment-validity gates
reject two of our own visual experiments before they can be reported, including a
baseline that clean accuracy alone would have passed. And a semantically sound
deduplication key — the phishing-kit identifier — still left **37.3%** of
evaluation crops pixel-identical to gallery items, inflating retrieval from 0.668
to 0.859: a deduplication key defined over records does not guarantee that the
artefact reaching the model is unique.

---

---

# 1. Introduction

A phishing detector is not evaluated by an adversary with a perturbation budget.
It is evaluated by one with a business model. That distinction is the subject of
this paper.

## 1.1 The problem: brittleness to manipulations that cost nothing

Visual and lexical phishing detectors are structurally brittle to changes an
attacker can make for free. Recompressing a screenshot, shifting a page's palette,
lengthening a URL path, or adding a subdomain requires no money, no engineering
time, and no infrastructure — the attacker already owns the DNS zone and the
document. We show this brittleness is not incidental. Trained with a standard
metric-learning objective on brand logos, an encoder scores **0.377** top-1
retrieval on grayscale inputs from unseen brands while *untrained* ImageNet
features score **0.619** (13 paired seeds, $p < 10^{-5}$). Training did not fail to acquire colour-invariance; it
destroyed robustness the initialisation already possessed, because colour is the
cheapest brand-discriminative signal available and the objective had no reason to
look past it.

## 1.2 The gap: $L_p$ balls do not describe semantic attacks

The standard robustness toolkit — adversarial training, TRADES, gradient penalties
— defines the threat as a norm ball around an input. That framing is a poor fit
here for two reasons.

First, the operations a phishing kit actually performs are *semantic*, not
$L_p$-bounded. Swapping a TLD, dropping a brand token, or re-rendering a page in a
different font can be enormous in pixel or feature distance while being trivial
for the attacker; conversely a large-$L_p$ perturbation may be impossible to
deploy in a real kit. Distance in representation space is uncorrelated with
difficulty in the attacker's world.

Second, and less obviously, the norm-ball framing has no way to express that
**some evasions defeat the attacker along with the detector**. Removing the brand
name from a URL beats brand-matching for free — and removes the cue the victim
relies on to be deceived. An attacker who strips every deceptive signal has evaded
detection by ceasing to phish. No $L_p$ budget can represent that, because the
constraint is not on the perturbation's size but on its effect on a third party.

## 1.3 The contribution: robustness as an economic problem

We argue that adversarial robustness in phishing is an economic problem before it
is a geometric one, and we make that operational. We price each manipulation on
three axes — money, engineering effort, and *victim conversion* — and evaluate
defences by **cost-to-evade (CTE)**: the cheapest bundle of manipulations that
flips a detected phish to benign. The third axis is the one the literature omits
and the one that does the work. It is what makes an evasion self-defeating, and
without it a cost-aware defence spends its capacity in exactly the wrong place
(§2.2).

Our claims are deliberately bounded. We do not claim a universal defence, and we
report two results that cut against our own framing: the coupling between attack
cost and feature destructiveness (§4.3.6), which limits what cost-weighting can demonstrate, and a modality
contrast in which cost weighting is a necessity in one setting and only a
second-order refinement in the other (§4.3.4).

## 1.4 Key findings

- **Pricing reliance predicts evasion; reliance alone does not.** Input-gradient
  reliance correlates only weakly with feature cheapness ($+0.201 \pm 0.031$),
  while per-feature evasion success correlates strongly with reliance *divided by*
  cost ($+0.774 \pm 0.045$). Auditing a model on reliance alone is a blind spot.

- **The intuitive defence is the wrong one.** Penalising gradient mass on cheap
  features does relocate the model onto expensive ones, and costs **12.4%** of
  clean F1 to buy less robustness than a cheaper mechanism. Cost-weighted
  *stochastic dropout* barely changes reliance and delivers **+388%** CTE for a
  0.56% F1 cost. Robustness comes from redundancy, not abstinence.

- **There is an operating point with no accuracy tax.** At a drop scale of 0.15
  the change in clean F1 is $+0.01\%$ — indistinguishable from baseline and
  positive in sign — while cost-to-evade rises **265%**.

- **Standard metric learning is a lazy learner.** It over-indexes on colour, which
  simultaneously explains a grayscale vulnerability *below the untrained control*
  and a near-total failure to transfer: seen brands gain $+0.204$ clean over that
  control, unseen brands only $+0.029$ — and on unseen brands *under attack* the
  trained encoder is $-0.033$ **worse** than the untrained one.

- **Logo matching cannot stand alone.** Running Phishpedia's own released detector
  over its own benchmark, 14.27% of samples yield no logo region at all, capping
  any logo-dependent method at **85.7% recall** before an attack is attempted.

---

# 2. Threat Model and the Economics of Evasion

## 2.1 The attacker's dilemma

We model an attacker who controls the URL, the hosting, and the rendered page, and
who is subject to one constraint the detection literature rarely encodes: the
campaign must still deceive a human. Every manipulation is therefore charged
against two budgets simultaneously — what it costs to perform, and what it costs
in victim conversion.

These budgets are not independent, and their coupling is structural. A
manipulation that leaves the victim's recognition intact necessarily leaves
structural signal for a detector to read; a manipulation that destroys the
victim's cue destroys the detector's evidence in the same stroke. JPEG
recompression is cheap on both budgets *because* the logo remains recognisable.
Deleting the wordmark is expensive on the conversion budget *because* nothing
remains to recognise. We quantify this coupling in §4.3.6 and treat it as a limit
on what any cost-weighted defence can claim, rather than as evidence for one.

The defender's objective follows. It is **not** to be robust to every semantic
manipulation. Robustness to `logo_delete` protects against an adversary who has
already abandoned the campaign's purpose. The objective is to **maximise the cost
an attacker must pay to evade, subject to preserving clean accuracy** — which
means deliberately declining to spend capacity on manipulations a rational
adversary will not choose.

## 2.2 The cost vector

Each of the 30 lexical features is priced on three axes in $[0,1]$:

$$c_i = \mathbf{w}^\top (c^{\text{money}}_i,\; c^{\text{effort}}_i,\;
c^{\text{conversion}}_i), \qquad \mathbf{w} = (0.45,\, 0.20,\, 0.35)$$

with a floor of $0.02$ so that $1/c_i$ stays finite for manipulations that are
free. Prices are grounded in what a 2026 attacker faces: TLS certificates are free,
subdomains are free because the attacker owns the zone, a fresh domain on a cheap
TLD is roughly \$1–10, and abandoning free hosting for reputable infrastructure
costs both money and an abuse-resistant identity. The resulting scale runs from
$0.020$ (path edits) to $0.423$ (leaving free hosting).

**The conversion axis is load-bearing, and we can show it by removing it.** Setting
$c^{\text{conversion}}$ to zero and renormalising re-prices the three features
that carry the deception itself:

| Feature | $c_i$ with conversion | without | rank with | rank without |
|---|---|---|---|---|
| `brand_outside_domain` | 0.325 | 0.020 | 29 / 30 | 5 / 30 |
| `num_suspicious_tokens` | 0.272 | 0.020 | 28 / 30 | 3 / 30 |
| `brand_in_path` | 0.255 | 0.020 | 27 / 30 | 4 / 30 |

(rank 1 = cheapest). Without the third axis, dropping the brand name from the URL
prices as among the *cheapest* evasions available, and a cost-aware defence would
correctly conclude it should spend most of its capacity defending against it —
against an attack that ends the campaign. The conversion axis moves these three
from ranks 3–5 to ranks 27–29. This is the difference between pricing what an
attacker *can* do and pricing what an attacker *will* do.

## 2.3 Rank-linear weighting

Defence mechanisms that weight features by cheapness cannot use $1/c_i$ directly.
Across our price list $1/c_i$ spans $2.37$ to $50.0$, a **21.1×** dynamic range,
and that spread destabilises the loss landscape: a handful of free features
dominate every gradient and training becomes high-variance.

We therefore weight by *rank* rather than by reciprocal cost. With features ordered
cheapest-first at normalised rank $r_i \in [0,1]$,

$$\omega_i = h - (h-1)\, r_i, \qquad h = 5$$

which preserves the cost ordering exactly while compressing the dynamic range from
21.1× to **5×**. The ordering is what the threat model asserts confidently; the
precise ratios are not, and the defence should not be sensitive to them. It is not:
perturbing $\mathbf{w}$ by ±10%, ±20% and ±30% moves median cost-to-evade only
between $0.222$ and $0.225$ (§4.1.3), so the results depend on the ranking of
manipulations by cost, not on the exact prices we assign.

## 2.4 Cost-to-evade

We report CTE as the total price of the cheapest bundle of manipulations that
moves a detected phishing sample below the operating threshold, searched greedily
by score reduction per unit cost under a budget. Because evasion sometimes fails
within budget, CTE is a right-censored time-to-event quantity and we estimate its
median with a Kaplan–Meier curve. Censoring is both substantial and *differential*
— 9.4% for the baseline against 38.3% for the defended model — so naive medians
systematically understate the defence, whose entire mechanism is to make evasion
fail more often (§4.1.4).

---

# 3. System and Method

## 3.1 Detectors and corpora

**Lexical.** The lexical detector is a three-layer MLP ($30 \to 64 \to 32 \to 1$,
ReLU) over the hand-designed URL features of §2.2, trained with AdamW
($\eta = 3\times10^{-3}$, weight decay $10^{-4}$, batch 512, 30 epochs) on 84,082
URLs and evaluated on 20,337 held out under a **domain-disjoint** split, so no
registrable domain appears in both. The model is deliberately small and
differentiable: input-gradient reliance and the cost-weighted penalties below are
only well-defined on a differentiable detector, and the questions asked here are
about *mechanism*, not about maximising absolute F1.

Operating thresholds are selected on validation at a 1% false-positive budget and
frozen before test. All lexical gradient computations run on CPU (§4, protocol
note).

**Visual.** The visual detector is a ResNet-18 initialised from ImageNet with its
classifier replaced by a linear projection to a 128-dimensional embedding,
$\ell_2$-normalised, trained with batch-hard triplet loss (Hermans et al.) using a
softplus margin. Batches are $P{\times}K$ with $P=8$ brands and $K=4$ crops, 40
steps per epoch for 30 epochs, Adam at $10^{-4}$.

The corpus is built from the Phishpedia benchmark. Rather than downscaling a
1366px page — which renders a wordmark at roughly $23\times21$ px and destroys the
signal (§4.2.1) — we crop to the highest-confidence detected logo region, pad to a
square to preserve aspect ratio, and resize to $128\times128$. After removing
exact-duplicate crops (§3.4) this yields **3,162 crops over 78 brands**. Splits are
**family-disjoint**: `family_id` is a phishing-kit hash, so pages sharing one are
near-duplicates, and brands held out for zero-shot evaluation contribute no
training data at all.

## 3.2 Defence mechanisms

All mechanisms weight features by the rank-linear $\omega_i \in [1,5]$ of §2.3
rather than by $1/c_i$ directly.

### 3.2.1 Cost-weighted dropout (permissive)

Feature $i$ is dropped independently per example with probability

$$p_i = \frac{\omega_i - \min_j \omega_j}{\max_j \omega_j - \min_j \omega_j}\cdot s,$$

so the cheapest feature is dropped with probability equal to the drop scale $s$
and the most expensive is never dropped. Training minimises the ordinary
cross-entropy on the masked input. Nothing is forbidden: a cheap feature remains
fully available whenever it survives the mask, and the model is free to use it.
What it cannot do is depend on it *exclusively*, because it is absent on a
predictable fraction of steps. This is the sense in which the mechanism is
**permissive** rather than **coercive**, and §4.1.2 shows the distinction is the
whole result.

### 3.2.2 Cost-budgeted TRADES

Standard TRADES takes its inner maximum over an $L_p$ ball, which §1.2 argues is
the wrong feasible set. We replace it with the set of manipulations the attacker
can *afford*: all feature subsets $S$ with $|S| \le 3$ and $\sum_{i\in S} c_i \le
B$. At $B = 0.15$ this is **1,345 of the 4,525 subsets** with $k \le 3$. The inner
step selects the affordable subset maximising the KL divergence between the clean
and perturbed predictive distributions — a stochastic argmax over 16 sampled
candidates per batch — and the outer objective adds $\beta \cdot \mathrm{KL}$ to
the classification loss. Perturbed features are moved to the benign centroid
(zero in standardised space), which is where an attacker aims.

### 3.2.3 Coercive baselines

Two comparators implement the intuitive alternative. The **gradient penalty** adds
$\lambda \sum_i \omega_i (\partial f/\partial x_i)^2$ to the loss, directly
suppressing gradient mass on cheap features. The **two-stage** variant trains
normally for half the epochs and then applies the same penalty, testing whether
the accuracy cost is an optimisation artefact rather than intrinsic. It is not
(§4.1.2).

### 3.2.4 Cost-weighted attack augmentation (visual)

In pixel space, features and attacks diverge: a patch has no economic value but
the semantic operation altering it does. We therefore price **attacks** rather than
features, and sample them during training with

$$P(a) \propto \exp(-\mathrm{cost}(a)/\tau),$$

applying one sampled manipulation per training crop. The identity attack has cost
zero and so retains the largest share; at $\tau = 0.06$ roughly 37% of crops are
left unmodified and the mass concentrates on cheap manipulations. The **uniform**
comparator samples the same eight manipulations with equal probability, which
isolates the contribution of cost weighting from that of augmentation itself.

## 3.3 Estimating cost-to-evade

Cost-to-evade is the total price of the cheapest bundle of manipulations that
moves a detected phishing sample below the operating threshold. We search greedily
by *score reduction per unit cost* rather than cheapest-first, over up to 12
manipulations within a budget of 4.0, on the first 250 correctly-detected phishing
samples. An earlier cheapest-first variant censored 85% of samples at the ceiling,
pinning the median at the budget and rendering the metric blind to improvement.

Because evasion sometimes fails within budget, CTE is **right-censored** and we
estimate its median with a Kaplan–Meier curve. This is not a stylistic preference.
Censoring is both substantial and *differential* — 9.4% for the baseline against
38.3% for the defended model — because the defence works precisely by making
evasion fail. A naive median therefore truncates the defended model's most
successful outcomes and systematically understates it (§4.1.4).

## 3.4 Experiment-validity gates

A robustness result is meaningful only if the baseline could perform the task and
the corpus and split can support the claim. We make these preconditions
executable, run them before any defence is trained, and report that they reject
two of our own experiments.

**Competence gate.** Performance is normalised as skill above chance,
$\mathrm{skill}(s) = (s - p_{\text{chance}})/(1 - p_{\text{chance}})$, so a
balanced binary task ($p=0.5$) and an $N$-way retrieval task ($p \approx 0.02$)
are judged on one axis; a raw multiple-of-chance rule is not scale-invariant and is
unsatisfiable for binary problems. We require skill $\ge 0.20$ clean and $\ge 0.05$
under cheap attack, below which no robustness signal remains for a defence to
move. The decisive third condition compares against an **untrained** reference: the
trained baseline must beat raw ImageNet features *under attack*, not merely on
clean data. This is the check that clean accuracy alone would have passed and that
caught our full-page screenshot baseline (§4.2.1).

**Corpus-validity gate.** A corpus can defeat a competent model in a way competence
cannot detect. The gate asserts an *ordering*: removing the brand mark must degrade
retrieval at least as much as a photometric perturbation, since logo matching rests
on the premise that the wordmark carries identity. An inverted ordering means the
model is a colour detector and any robustness number describes the renderer. Our
synthetic corpus failed exactly this test while passing every competence check.

**Split-leakage gate.** The first two gates both passed on a corpus in which 37.3%
of query crops were pixel-identical to gallery crops, despite a `family_id` split
with zero violations across 75 brands. The gate fingerprints the tensors that
reach the model — not the records they came from — and fails any split whose query
set duplicates more than 2% of the gallery. It is applied to the train/gallery,
train/query, gallery/query and zero-shot splits alike, since duplicate crops in a
zero-shot set inflate precisely the generalisation claim that set exists to make.

---

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
is only weakly concentrated on cheap features — $\mathrm{corr}(\text{reliance},
1/\text{cost}) = +0.201 \pm 0.031$ over five seeds — so an audit that stopped at
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
| baseline | 0.9358 | — | 0.064 | — | +0.202 |
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
abstinence — and the two are separable only because we measured reliance
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

At a drop scale of 0.15 the accuracy change is $+0.01\%$ — statistically
indistinguishable from the baseline, and *positive in sign* — while cost-to-evade
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
9.4% for the baseline against 38.3% for cost-weighted dropout — the defence works
precisely by making evasion fail more often, so the arm we care about is the one
whose median is most understated. Comparing naive medians therefore understates
the defence.

**Table 4.4. Kaplan–Meier corrected cost-to-evade (P3).**

| Model | F1 | ΔF1 | KM median CTE | ΔCTE | Censored |
|---|---|---|---|---|---|
| baseline | 0.9358 | — | 0.065 | — | 9.4% |
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
attack** — indistinguishable from noise. Worse, it was *less robust than doing
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
batch-hard triplet training — identical loss, backbone and optimiser to the run
that failed — then improves retrieval under **all nine attacks**, clean 0.656 →
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
while untrained ImageNet features score 0.619 ± 0.056** — a paired difference of
**−0.242 (t = −14.2, p < 10⁻⁵, losing on 13 of 13 seeds)**. The effect is not
confined to grayscale: on the cheap-attack mean the trained baseline also falls
below the untrained control (0.699 vs 0.732, **t = −5.95, p = 0.00007**, losing on
12 of 13 seeds). Training did not merely
fail to acquire colour-invariance — it *destroyed* robustness the initialisation
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
control's 0.619 is a difference of +0.016 with $p = 0.32$ — statistically
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
power. Only at thirteen seeds — the sample size the observed effect size
$d_z = 0.93$ requires for 80% power — does the effect resolve, and it resolves
*upward*, to +0.0144. Two conclusions follow, and they point in opposite
directions. Four fifths of the original margin was an artefact of unpaired
evaluation. But the residue is real, and an analysis that had stopped at five
seeds would have wrongly reported a null.

We are correspondingly careful about magnitude. Cost weighting improves zero-shot
attacked retrieval by roughly **1.4 percentage points** over uniform augmentation
— against the **6.9 points** that augmentation itself contributes over no
augmentation at all (§4.3.2). The dominant effect by a factor of five is that
attacks are simulated during training; *pricing* those attacks is a real but
second-order refinement on top of it. A reader deciding what to implement should
read that ordering as the finding.

This yields a **modality contrast** rather than a uniform endorsement. In the
discrete lexical setting cost weighting was not a refinement but a necessity: its
undifferentiated comparator, a uniform gradient penalty, cost **12.4%** of clean F1
(§4.1.2), a catastrophic price no amount of robustness would justify. In the
continuous visual embedding space the undifferentiated comparator is nearly free —
uniform augmentation costs 1.6 points of clean zero-shot retrieval, not
significantly different from baseline (p = 0.068) — and cost weighting buys 1.4
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
where the cost model says it should. On `logo_delete` — an attack that erases the
brand mark — uniform augmentation scores 0.544 and cost weighting 0.373, a paired
difference of **−0.172 (t = −10.36, p < 10⁻⁵, losing on 13 of 13 seeds)**.

The sharpest form of the result is a **null**. On `logo_delete`, cost-weighted
augmentation is statistically indistinguishable from *no augmentation whatsoever*
(−0.010, **p = 0.55**), while uniform augmentation gains +0.162 (p < 10⁻⁵). Cost
weighting spends essentially **nothing** defending an attack that destroys the
victim's own recognition cue — and nevertheless achieves *higher* aggregate
zero-shot robustness than the arm that spends 11% of its training budget there
(§4.3.4). The allocation is not merely defensible on threat-model grounds; on this
corpus it is also the better-performing one.

We state the cost of that allocation plainly. Against the untrained control,
cost-weighted augmentation is significantly *worse* on `logo_delete` (−0.054,
p = 0.027): the policy does not simply decline to improve there, it gives ground.
An adversary willing to abandon the brand cue entirely — and thereby most of the
campaign's conversion — would find this model easier to evade than an
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
zero-shot transfer rather than on robustness to cheap attacks — and, in light of
§4.3.4, why we report the visual cost model as a contrast with the lexical setting
rather than as a second demonstration of its superiority.

---

## 4.4 Limitations of visual-only detection

Cropping to the logo region is what made the visual task learnable, but it can
only act where a logo is found. Running Phishpedia's own released detector over its
own 30k benchmark, **4,208 of 29,496 samples (14.27%) yield no logo region at
all** — an empty coordinate file. Any method requiring a detected brand mark is
therefore capped at **85.7% recall on this corpus before a single evasion attack
is attempted**.

We state the bound precisely: it is a property of the detector-and-data pair, not
a theoretical limit of logo matching, and a stronger detector would move it. It
nonetheless bounds the entire visual arm, and it is the clearest quantitative
argument for why brand-mark matching must sit inside a multimodal system rather
than issue verdicts alone. On those 14% of pages the URL and text branches carry
the decision unaided — which is the architectural claim §4.1 and §4.2 jointly
support: the modalities fail in different places, and the cost structure that
governs their defence differs between them.

---

# 5. Related Work

## 5.1 Visual and lexical phishing detection

Modern phishing detection increasingly relies on reference-based visual matching
to overcome the brittleness of lexical-only classifiers. Phishpedia
[1] introduced a hybrid approach that first runs an object detector to
isolate identity logos, then uses a Siamese network to match those crops against a
trusted brand database [1]. This achieves high identification accuracy
without training on phishing samples, but it does not model the
adversary.

We adopt the Phishpedia pipeline as our visual substrate and use its released
benchmark and detector. Our contribution is adversarial rather than architectural:
we show that the metric-learning objective at the centre of this design
inadvertently optimises for cheap, texture-level signal — principally colour — and
that this leaves the encoder *less* robust on unseen brands than its own untrained
initialisation (§4.2.3). We also report a bound inherited from the reference-based
design itself: on 14.27% of the benchmark's own samples the detector finds no logo
at all (§6.1).

## 5.2 Adversarial robustness and the $L_p$ abstraction

The standard framework defines robustness as invariance within an $L_p$ ball
around the input. TRADES [2] explicitly optimises the theoretical
trade-off between clean accuracy and boundary robustness. But $L_p$
bounds map poorly onto the operations a phishing kit performs. A semantic
manipulation — swapping a font, recolouring a template, deleting a wordmark — can
be enormous in $L_2$ while costing the attacker nothing, and conversely a
small-norm perturbation may be undeployable in a real kit. Distance in
representation space is uncorrelated with difficulty in the attacker's world.

We adapt the TRADES formulation into **cost-budgeted TRADES**, replacing the $L_p$
neighbourhood with an explicitly enumerated set of economically feasible
manipulation subsets (§3.2.2). The outer objective is unchanged; only the feasible
set of the inner maximisation moves, from a norm ball to what the attacker can
afford.

## 5.3 Cost-aware and semantic attacks

Cost-sensitive machine learning has generally concerned *feature acquisition*
costs borne by the defender at inference time, not manipulation costs borne by an
adversary. Work on semantic adversarial examples has largely focused on generating
perceptually realistic perturbations rather than on whether deploying them is
economically rational.

Our contribution is to structure the adversary's problem as a **dual-budget**
optimisation: every manipulation is charged both against what it costs to perform
and against what it costs in victim conversion. To our knowledge the second budget
is not modelled in existing phishing-robustness work, and §2.2 shows it is not a
refinement but a reordering — without it, the three manipulations that carry the
deception price among the cheapest available.

---

# 6. Discussion and Limitations

## 6.1 The 85.7% recall bound of logo matching

Visual reference matching is structurally capped by the presence of a detectable
brand mark. Running Phishpedia's own released detector over its own 30k benchmark,
**4,208 of 29,496 samples (14.27%)** yield an empty coordinate file: no logo is
present to crop. Any logo-dependent architecture is therefore bounded at **85.7%
recall** before an adversary attempts anything.

We are careful about the scope of this claim. It is a property of the
detector-and-data pair, not a theoretical limit of logo matching, and a stronger
detector would move it. It nonetheless bounds our entire visual arm, and it is the
clearest quantitative argument for why visual matching cannot issue verdicts
alone: on those 14% of pages the URL and text branches must carry the decision
unaided.

## 6.2 The cost/destructiveness coupling

The most fundamental limit on our own framing is developed in §4.3.6 and
summarised here. Attack cost and feature destructiveness are coupled by
construction: a manipulation is cheap on the conversion budget precisely when it
leaves the victim's recognition intact, which necessarily leaves structural signal
for a detector to read. Measured against our price list, retained retrieval
correlates $-0.765$ with attack cost overall and $-0.871$ across cheap attacks.

The consequence is methodological. Robustness to cheap attacks is a *weak* test of
a cost-aware defence, because the ordering such a defence seeks to induce is
partly supplied by the task itself. This is why our primary visual metric is
zero-shot transfer to unseen brands, which the coupling does not determine.

## 6.3 Two distinct corpus failures, and a leak in the reference index

Our early visual experiments failed in two *different* ways, and conflating them
would obscure both.

The **synthetic** corpus failed the corpus-validity gate, not the competence gate.
It supported a perfectly competent model; the problem was that it encoded brand
identity almost entirely in colour, so deleting the wordmark cost 0.19 top-1
retrieval while a colour shift cost 0.62 — an ordering inverted from real brand
pages. Any robustness result on it would have described our renderer.

The **real full-page screenshot** corpus failed the competence gate, for an
unrelated reason: at 224×224 a 1366px page renders its wordmark at roughly
$23\times21$ px, and the baseline sat at 0.017 under attack against a 0.0043
chance floor while being less robust than untrained ImageNet features. Cropping to
the logo region — same loss, backbone and optimiser — resolved it entirely
(§4.2.1). The failure was representational, not architectural.

Separately, our audits found that a semantically meaningful deduplication key is
insufficient. Distinct phishing kits clone the *same official brand asset*, so
splitting on the kit hash left **37.3% of query crops pixel-identical to crops in
the gallery** — the reference index they are retrieved against — even though
gallery and query shared no `family_id` across 75 brands. We stress the precise
location of this leak: train-to-query overlap was 0.0%, so this is contamination
of the *retrieval index*, not classical train/test contamination, which is exactly
why a record-level split did not catch it. Uncorrected it inflated clean top-1
from 0.668 to 0.859, degenerating retrieval into an exact-match lookup.
Tensor-level fingerprinting is necessary; a deduplication key defined over records
does not guarantee the artefact reaching the model is unique.

## 6.4 Heuristic cost weighting

The values in our cost vector are grounded in 2026 attacker economics but
ultimately encode our operational judgement rather than measured telemetry from
live campaigns. Two things mitigate this. We use rank-linear weighting, which
compresses the 21.1× dynamic range of $1/c_i$ and preserves only the *ordering*
the threat model asserts confidently (§2.3). And perturbing the cost weights by
±10%, ±20% and ±30% moves median cost-to-evade only between **0.222 and 0.225**,
against a per-condition spread of ±0.008–0.013 — the defence depends on the
ordinal ranking of manipulations, not on their absolute prices. We note this is a
sensitivity analysis rather than a hypothesis test; we did not test the
differences for significance, and report the range observed.

Eliciting manipulation costs from observed campaign telemetry, rather than
assigning them, remains the natural next step, as does extending cost-to-evade to
the joint multimodal decision, where an attacker must pay across modalities
simultaneously and the cheapest bundle may exploit whichever modality the defender
has left unpriced.

---

# 7. Conclusion

We set out to test whether pricing an attacker's manipulations changes how a
phishing detector should be defended. It does, but not uniformly, and not always
in the direction we expected.

**What the cost model buys.** Pricing is what makes exploitability predictable.
The gap between r = +0.201 for reliance alone and r = +0.774 for reliance divided
by cost is the empirical case for the framework: a model audit that asks only
"which features does this model use?" cannot anticipate where it will be evaded.
The conversion axis is what keeps that pricing honest, and the ablation in §2.2 is
the sharpest evidence — without it, a cost-aware defence would rationally conclude
that its scarce capacity belongs on the manipulation that ends the campaign.

**What the defence looks like.** The mechanism that works is not the one the
framing suggests. Coercing a model off cheap features succeeds on its own terms
and is unusable, costing 12.4% of clean F1. Removing those features
*stochastically* leaves reliance almost unchanged and buys far more robustness,
because the model builds redundant pathways rather than abandoning a signal it
needs. There exists an operating point at which robustness rises 265% for no
measurable accuracy cost at all. We regard "redundancy, not abstinence" as the
transferable lesson, and note that it was only visible because we measured
reliance directly and found the better mechanism barely moved it.

**How far it transfers.** In the continuous visual embedding space the same
machinery earns its keep, but on a different scale. Semantic augmentation is
emphatically necessary — without it, metric learning leaves an encoder worse than
its own initialisation on unseen brands. Weighting that augmentation by attacker
cost adds a further significant but modest gain (+0.014, p = 0.006), roughly a
fifth of what augmentation itself contributes. So pricing is load-bearing in the
lexical setting, where its absence costs 12.4% of clean F1, and a second-order
refinement in the visual one. Cost weighting is additionally the more efficient
allocation: it reaches higher aggregate robustness while spending nothing on
brand-erasing attacks, being indistinguishable there from no augmentation at all.
It also gives ground there relative to an untrained encoder, a trade whose
correctness rests on the conversion penalty reflecting real attacker economics.

**A limit on our own framing.** Attack cost and feature destructiveness are
coupled by construction. A manipulation is cheap in conversion terms precisely
when it leaves the victim's recognition intact, which necessarily leaves signal
for a detector; one that erases the victim's cue erases the detector's evidence.
Any defence that reallocates capacity toward cheap attacks therefore pushes on an
ordering the task already supplies, and cost-weighted robustness on cheap attacks
is a weak test of the framework. This is why we evaluate on zero-shot transfer,
and why we present the visual result as a contrast.

**Scope.** Our visual arm inherits a hard ceiling: 14.27% of the Phishpedia
benchmark yields no detectable logo region, capping any logo-dependent method at
85.7% recall before an attack is attempted. That bound is a property of the
detector-and-data pair rather than of logo matching in principle, but it is the
clearest argument in this paper for why brand-mark matching must sit inside a
multimodal system rather than issue verdicts alone. The lexical results rest on a
small differentiable model over 30 hand-designed features, and the cost table,
while grounded in 2026 attacker economics and shown insensitive to ±30% weight
perturbation, encodes our judgement rather than measured attacker behaviour.
Eliciting those prices from observed campaign data is the natural next step, as is
extending cost-to-evade to the joint multimodal decision, where an attacker must
pay across modalities at once and the cheapest bundle may exploit exactly the
modality a defender has left unpriced.

**On negative results.** Three of this paper's findings are corrections to earlier
results of our own: a fusion mechanism advantage that vanished across seeds, a
visual defence experiment whose baseline could not perform the task, and a
cost-weighting margin that shrank fivefold once evaluation was properly paired —
and that a five-seed analysis would then have wrongly dismissed as null.
We report them because each was invisible under the analysis that first produced
it, and because a harness able to reject its authors' own experiments is a
stronger claim to rigour than a table those checks would have refused.

---

# References

[1] Y. Lin, R. Liu, D. M. Divakaran, J. Y. Ng, Q. Z. Chan, Y. Lu, Y. Si, F. Zhang,
and J. S. Dong. "Phishpedia: A Hybrid Deep Learning Based Approach to Visually
Identify Phishing Webpages." In *30th USENIX Security Symposium (USENIX Security
21)*, pp. 3793–3810, 2021.

[2] H. Zhang, Y. Yu, J. Jiao, E. P. Xing, L. El Ghaoui, and M. I. Jordan.
"Theoretically Principled Trade-off between Robustness and Accuracy." In
*Proceedings of the 36th International Conference on Machine Learning (ICML)*,
PMLR 97:7472–7482, 2019.
