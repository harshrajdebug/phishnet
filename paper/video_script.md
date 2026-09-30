# PhishNet — video script (~4 min)

Target: 4:00. ~600 words spoken at a normal pace.
Record in five takes, one per section, then cut together.

---

## 0:00 – 0:35 — The problem (why I built this)

**On screen:** a real phishing screenshot next to the genuine brand page,
side by side. No captions yet.

> Here are two login pages. One is a bank. One is a phishing kit. They are
> pixel-identical, because the attacker copied the real one.
>
> That's the problem with visual phishing detection. Every defence gets
> benchmarked on a fixed set of image attacks, and reports a robustness
> number. But the attacker isn't drawing from that fixed set. They're picking
> whatever is cheapest to pull off.
>
> So a defence can look strong on paper and still be trivially cheap to get
> around — because it spent all its capacity on attacks nobody would bother
> to run.

---

## 0:35 – 1:15 — The idea

**On screen:** the cost axes, then the weight bar. Keep it as three labels
and a number, not a busy diagram.

> My question was: what if you weight the training by what each attack
> actually costs the attacker?
>
> I scored every attack on three axes — money, effort, and how much it hurts
> victim conversion. Blurring a logo is cheap. Deleting the brand mark
> entirely is expensive, because now the victim can't tell what they're
> logging into, and conversion drops.
>
> Then I trained the model to spend its robustness budget in proportion to
> one over cost. Cheap attacks get defended hard. Expensive attacks get
> defended less, on purpose.

---

## 1:15 – 2:05 — How it works

**On screen:** the architecture block from the README, built up one branch
at a time. Then a single frame of the attack-budget selection.

> The detector itself reads a page three ways. A character-level CNN over the
> URL. A Siamese ResNet over the screenshot, matched against a brand index.
> And DistilBERT over the page text.
>
> Those three get fused with cross-modal attention, which matters, because the
> useful question isn't "does this look like a bank." A perfect clone looks
> exactly like a bank. The question is "does this look exactly like Brand X
> while being served from a domain that isn't X's" — and that's relational.
> You can only answer it by reading the visual signal against the URL signal.
>
> Training is cost-budgeted TRADES. At each step the inner maximisation only
> considers attack combinations the attacker could actually afford — about
> thirteen hundred of forty-five hundred, at a budget of 0.15.

---

## 2:05 – 3:00 — The result

**On screen:** the seed table. Highlight one row at a time.

> Thirteen seeds, paired — same initialisation, same batch order, same
> eval-time attack draws across every arm. So the only thing that differs is
> the weighting.
>
> Against the cost-weighted attacker, the model beats uniform augmentation by
> 1.4 points, wins eleven of thirteen seeds, p equals 0.006. Against the
> undefended baseline it's 8.3 points and wins all thirteen.
>
> But the number I care about most is this one.
>
> **[pause — put logo_delete on screen]**
>
> On logo deletion, the cost-weighted model is *seventeen points worse* than
> uniform augmentation. That's not a failure. That's the mechanism working.
> Logo deletion is expensive for the attacker, so the model declined to spend
> there and reallocated to the cheap attacks instead. Uniform augmentation
> can't make that choice — it defends everything equally, including the
> attacks nobody runs.
>
> At n=5 this margin looked like nothing. It took thirteen seeds to resolve.

---

## 3:00 – 3:40 — Demo

**On screen:** screen recording. Start the API, open the dashboard, visit a
flagged page with the extension on, show the verdict and the explanation
panel.

> It runs live. The extension scores the page client-side, on what the user
> actually sees — which matters, because modern kits cloak against crawlers
> and serve a clean decoy to anything that looks automated.
>
> Every verdict comes with the URL features that drove it and a Grad-CAM over
> the region the visual branch matched.

---

## 3:40 – 4:00 — Close

**On screen:** back to camera, or the title card.

> The broader claim is that adversarial robustness is an allocation problem,
> not a coverage problem. You don't have the budget to defend everything. So
> the question isn't how much robustness you have — it's where you put it.
>
> Code, paper, and the full thirteen-seed replication are in the repo.

---

## Notes for recording

- **Start the API before the demo take:** `scripts/serve.sh`. It's currently
  stopped, so the extension returns nothing until you do.
- The logo_delete beat at 2:35 is the strongest thirty seconds in the video.
  Slow down there. Let the "seventeen points worse" sit for a full second
  before the explanation.
- Don't read the architecture list as a list. Say each branch while its block
  appears.
- If you need to cut to 3 minutes, drop the fusion-rationale paragraph at
  1:30 first, then compress the demo to a single flagged page.
- Numbers as spoken, all from `results/phishpedia_seeds_stats.json`:
  cost−uniform attacked +0.0144, p=0.0057, 11/13 · cost−baseline attacked
  +0.0831, 13/13 · cost−uniform logo_delete −0.1717, 0/13.
