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
