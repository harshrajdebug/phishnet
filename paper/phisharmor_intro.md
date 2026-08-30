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
