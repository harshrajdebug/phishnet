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

# References

[1] Y. Lin, R. Liu, D. M. Divakaran, J. Y. Ng, Q. Z. Chan, Y. Lu, Y. Si, F. Zhang,
and J. S. Dong. "Phishpedia: A Hybrid Deep Learning Based Approach to Visually
Identify Phishing Webpages." In *30th USENIX Security Symposium (USENIX Security
21)*, pp. 3793–3810, 2021.

[2] H. Zhang, Y. Yu, J. Jiao, E. P. Xing, L. El Ghaoui, and M. I. Jordan.
"Theoretically Principled Trade-off between Robustness and Accuracy." In
*Proceedings of the 36th International Conference on Machine Learning (ICML)*,
PMLR 97:7472–7482, 2019.
