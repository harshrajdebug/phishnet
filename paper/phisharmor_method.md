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
