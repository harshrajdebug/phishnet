# Abstract

Adversarial robustness for phishing detection is usually posed geometrically, as invariance inside an Lp ball; we argue it is better posed economically. A phishing kit's manipulations are semantic, not norm-bounded, and some evasions defeat the attacker along with the detector: removing the brand token from a URL beats brand matching for free, and removes the cue the victim needs to be deceived. We price 30 lexical and 8 visual manipulations on three axes - money, effort, and victim conversion - and evaluate defences by cost-to-evade. Ablating the conversion axis moves the three deception-bearing features from ranks 3-5 to 27-29 of 30, separating what an attacker can do from what an attacker will do.

Our results are corrective. Reliance alone barely predicts exploitability (r = +0.201), whereas reliance priced by cost predicts it strongly (r = +0.774). The intuitive defence, penalising gradient mass on cheap features, relocates the model as instructed, costs 12.4% of clean F1, and buys less robustness than cost-weighted stochastic dropout, which barely perturbs reliance and yields +457% cost-to-evade for 0.88% F1; at a gentler setting, +0.01% for +265%. Robustness comes from redundancy, not abstinence.

Visually, metric learning over-indexes on colour, leaving an encoder less robust on unseen brands than its untrained initialisation (grayscale -0.242, 13 paired seeds). Semantic augmentation repairs this (+0.069); pricing it adds +0.014 (p = 0.006) while spending nothing on attacks that blind the victim. Validity gates reject two of our own experiments, including a corpus where 37.3% of crops duplicated gallery items.

---

# 7. Conclusion

We set out to test whether pricing an attacker's manipulations changes how a
phishing detector should be defended. It does, though not uniformly, and not
always in the direction we expected.

**What pricing buys.** Exploitability becomes predictable once cost enters the
picture. Reliance on its own correlates at +0.201 with cheapness; reliance
divided by cost correlates at +0.774 with actual evasion success. An audit that
asks only which features a model uses cannot anticipate where it will be evaded.
The conversion axis is what keeps those prices honest. Strip it out and a
cost-aware defence rationally concludes that its scarce capacity belongs on the
one manipulation that ends the campaign.

**What the defence looks like.** The mechanism that works is not the one the
framing suggests. Coercing a model off cheap features succeeds on its own terms
and is unusable, costing 12.4% of clean F1. Removing those same features
*stochastically* leaves reliance almost untouched and buys far more robustness,
because the model builds redundant pathways instead of abandoning a signal it
needs. One operating point raises cost-to-evade by 265% for no measurable
accuracy cost at all. We think "redundancy, not abstinence" is the transferable
lesson here, and note that it only became visible because we measured reliance
directly and found the better mechanism had barely moved it.

**How far it transfers.** In the continuous visual embedding space the same
machinery earns its keep, but on a different scale. Semantic augmentation is
emphatically necessary: without it, metric learning leaves an encoder worse than
its own initialisation on unseen brands. Weighting that augmentation by attacker
cost adds a further significant but modest gain of +0.014 (p = 0.006), roughly a
fifth of what augmentation itself contributes. Pricing is therefore load-bearing
in the lexical setting, where its absence costs 12.4% of clean F1, and a
second-order refinement in the visual one. Cost weighting is also the more
efficient allocation, reaching higher aggregate robustness while spending nothing
on brand-erasing attacks. It gives ground there relative to an untrained encoder,
a trade whose correctness rests on the conversion penalty reflecting real
attacker economics.

**A limit on our own framing.** Attack cost and feature destructiveness are
coupled by construction. A manipulation is cheap in conversion terms when it
leaves the victim's recognition intact, which necessarily leaves signal for a
detector; one that erases the victim's cue erases the detector's evidence too.
Any defence reallocating capacity toward cheap attacks is therefore pushing on an
ordering the task already supplies, which makes cost-weighted robustness on cheap
attacks a weak test. That is why we evaluate on zero-shot transfer, and why we
present the visual result as a contrast rather than a second win.

**Scope and future work.** The visual arm inherits a hard ceiling: 14.27% of the
Phishpedia benchmark yields no detectable logo region, capping any logo-dependent
method at 85.7% recall before an attack is attempted. That bound belongs to the
detector-and-data pair rather than to logo matching in principle, but it is this
paper's clearest argument for why brand-mark matching must sit inside a
multimodal system. The lexical results rest on a small differentiable model over
30 hand-designed features, and the cost table encodes our judgement rather than
measured attacker behaviour, though it survives +/-30% weight perturbation.
Eliciting those prices from observed campaign data is the natural next step, as
is extending cost-to-evade to the joint multimodal decision, where an attacker
pays across modalities at once and the cheapest bundle may exploit whichever one
a defender left unpriced.

**On negative results.** Three findings in this paper correct earlier results of
our own: a fusion mechanism advantage that vanished across seeds, a visual
defence experiment whose baseline could not perform the task, and a cost-weighting
margin that shrank fivefold once evaluation was properly paired, then would have
been wrongly dismissed as null at five seeds. Each was invisible under the
analysis that first produced it. A harness able to reject its authors' own
experiments seems to us a stronger claim to rigour than a table those checks
would have refused.
