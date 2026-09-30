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
inadvertently optimises for cheap, texture-level signal, principally colour, and
that this leaves the encoder *less* robust on unseen brands than its own untrained
initialisation (§4.2.3). We also report a bound inherited from the reference-based
design itself: on 14.27% of the benchmark's own samples the detector finds no logo
at all (§6.1).

## 5.2 Adversarial robustness and the $L_p$ abstraction

The standard framework defines robustness as invariance within an $L_p$ ball
around the input. TRADES [2] explicitly optimises the theoretical
trade-off between clean accuracy and boundary robustness. But $L_p$
bounds map poorly onto the operations a phishing kit performs. A semantic
manipulation, swapping a font, recolouring a template, deleting a wordmark, can
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
refinement but a reordering, without it, the three manipulations that carry the
deception price among the cheapest available.

---

# 6. Discussion and Limitations

**A recall ceiling, not just an accuracy one.** Reference matching is capped by
whether a brand mark is present at all. Running Phishpedia's own detector across
its own 30k benchmark, 4,208 of 29,496 samples (14.27%) return an empty
coordinate file. Any logo-dependent architecture is therefore bounded at 85.7%
recall before an adversary attempts anything. The bound belongs to this
detector-and-data pair rather than to logo matching in principle, and a stronger
detector would move it, but it still bounds our whole visual arm. On those pages
the URL and text branches carry the decision alone (§4.4).

**Cost and destructiveness are coupled.** This is the sharpest limit on our own
framing. Retained retrieval correlates -0.765 with attack cost overall and -0.871
across cheap attacks, because a manipulation is cheap in conversion terms exactly
when it spares the victim's recognition, which also spares the detector's
evidence. The consequence is methodological rather than empirical: robustness to
cheap attacks is a weak test of a cost-aware defence, since the ordering it seeks
to induce is partly handed to it by the task. Zero-shot transfer, which the
coupling does not determine, is the metric we rely on instead (§4.3.6).

**Two corpus failures, and a leak in the reference index.** Our early visual
experiments failed in two unrelated ways, and merging them would obscure both.
The synthetic corpus failed corpus validity, not competence: it hosted a
perfectly capable model, but encoded brand identity almost entirely in colour, so
deleting the wordmark cost 0.19 top-1 retrieval while a colour shift cost 0.62.
That ordering is inverted from real brand pages, and any robustness result on it
would have described our renderer. The real full-page corpus failed competence
instead, for a reason of resolution: at 224x224 a 1366px page renders its
wordmark at roughly 23x21 px, leaving the baseline at 0.017 under attack against
a 0.0043 chance floor and less robust than untrained ImageNet features. Cropping
to the logo region, with loss and optimiser unchanged, resolved it (§4.2.1).

Separately, a semantically meaningful deduplication key proved insufficient.
Distinct kits clone the same official asset, so splitting on the phishing-kit hash
still left 37.3% of query crops pixel-identical to crops in the gallery they are
retrieved against, even though gallery and query shared no family_id across 75
brands. We stress where that leak sits: train-to-query overlap was 0.0%, so this
contaminates the retrieval index rather than the training set, which is exactly
why a record-level split missed it. Uncorrected it inflated clean top-1 from
0.668 to 0.859.

**Heuristic prices.** Our cost vector is grounded in 2026 attacker economics but
encodes operational judgement, not telemetry. Rank-linear weighting preserves only
the ordering the threat model asserts confidently (§2.3), and perturbing the
weights by +/-10%, +/-20% and +/-30% moves median cost-to-evade between 0.222 and
0.225 against a per-condition spread of +/-0.008-0.013. That is a sensitivity
analysis, not a hypothesis test; we did not test the differences for significance.
Eliciting manipulation costs from observed campaign telemetry, rather than
assigning them, remains the natural next step, as does extending cost-to-evade to
the joint multimodal decision.

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
