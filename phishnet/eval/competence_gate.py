"""A mandatory pre-flight check before any defence experiment.

Rationale, learned the expensive way. We ran two defence experiments whose
results were uninterpretable because the *baseline* could not do the task. On the
synthetic corpus the baseline collapsed to chance under any photometric attack;
on real full-page screenshots both baseline and defended sat at 1-4% against a
0.4% chance floor. A defence cannot be shown to improve robustness that is
already at the noise floor, so both runs measured nothing.

The subtler trap was the third check below. On real screenshots our triplet
training *doubled* clean retrieval (0.134 -> 0.299 versus untrained ImageNet
features) while degrading robustness on every single attack (jpeg 0.032 -> 0.013,
logo-occlusion 0.091 -> 0.007). Judging the baseline on clean accuracy alone
would have passed it. A baseline that is worse than doing nothing under attack is
not a valid starting point for measuring a robustness defence.

Run this before spending compute on a defence. It costs one evaluation pass.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    name: str = "COMPETENCE GATE"

    def __str__(self) -> str:
        return (f"{self.name}: {'PASS' if self.passed else 'FAIL'}"
                + "".join(f"\n  - {r}" for r in self.reasons))


def skill(score: float, chance: float) -> float:
    """Performance normalised by the headroom above chance, in [0, 1].

    A raw multiple-of-chance test is not scale invariant: for a balanced binary
    task chance is 0.5, so "10x chance" demands an accuracy of 5.0. This maps any
    task onto a common axis -- 0 means chance, 1 means perfect.
    """
    denom = 1.0 - chance
    return (score - chance) / denom if denom > 0 else 0.0


def competence_gate(clean: float,
                    attacked: float,
                    chance: float,
                    control_clean: float | None = None,
                    control_attacked: float | None = None,
                    min_clean_skill: float = 0.20,
                    min_attacked_skill: float = 0.05) -> GateResult:
    """Decide whether a baseline is competent enough to host a defence experiment.

    clean / attacked   baseline performance, clean and averaged over cheap attacks
    chance             random-guess rate for the task (1 / n_classes, or the
                       majority-class rate for a binary task)
    control_*          the same metrics for an *untrained* reference, e.g. raw
                       ImageNet features. Optional but strongly recommended: it is
                       the only check that catches a baseline which has traded
                       robustness for clean accuracy.
    min_clean_skill    normalised skill required on clean data
    min_attacked_skill normalised skill required under cheap attack; below this
                       there is no robustness signal left for a defence to improve
    """
    reasons, ok = [], True
    cs, as_ = skill(clean, chance), skill(attacked, chance)
    detail = {"clean": clean, "attacked": attacked, "chance": chance,
              "clean_skill": cs, "attacked_skill": as_}

    if cs < min_clean_skill:
        ok = False
        reasons.append(f"clean skill {cs:.3f} < {min_clean_skill:g}; the model cannot "
                       "do the task well enough to host a defence experiment")
    if as_ < min_attacked_skill:
        ok = False
        reasons.append(f"attacked skill {as_:.3f} < {min_attacked_skill:g}; robustness is "
                       "already at the noise floor, so a defence cannot be measured")
    if control_clean is not None and clean <= control_clean:
        ok = False
        reasons.append(f"training does not beat the untrained control on clean "
                       f"({clean:.3f} vs {control_clean:.3f})")
    if control_attacked is not None:
        detail["control_attacked"] = control_attacked
        if attacked < control_attacked:
            ok = False
            reasons.append(f"training is LESS robust than the untrained control "
                           f"({attacked:.3f} vs {control_attacked:.3f}); the baseline has "
                           "overfit to brittle cues and is not a valid starting point")
    if ok:
        reasons.append(f"clean skill {cs:.3f}, attacked skill {as_:.3f} "
                       f"(chance {chance:.4f})")
    return GateResult(ok, reasons, detail)


def corpus_validity_gate(per_attack: dict[str, float],
                         identity_key: str = "identity",
                         identity_destroying: tuple[str, ...] = ("logo_occlude", "logo_delete"),
                         photometric: tuple[str, ...] = ("colour_shift", "brightness",
                                                         "jpeg", "blur", "grayscale")) -> GateResult:
    """Check that a visual corpus encodes brand identity the way real pages do.

    This catches a failure the competence gate cannot see. Our synthetic corpus
    passed every competence check yet was useless, because it encoded brand
    identity almost entirely in colour: deleting the brand wordmark cost only
    0.19 top-1 retrieval while a colour shift cost 0.62. Real brand pages behave
    in the opposite way -- logo-matching systems are built on the premise that
    the wordmark carries the identity.

    The test is an ordering one. Removing the brand mark must hurt at least as
    much as a photometric perturbation. If the ordering is inverted, the model is
    learning a colour detector and any robustness result on that corpus describes
    the renderer rather than the phenomenon.
    """
    ident = per_attack.get(identity_key, 1.0)
    dest = [per_attack[k] for k in identity_destroying if k in per_attack]
    photo = [per_attack[k] for k in photometric if k in per_attack]
    if not dest or not photo:
        return GateResult(False, ["insufficient per-attack coverage to judge corpus validity"],
                          name="CORPUS VALIDITY GATE")
    md, mp = sum(dest) / len(dest), sum(photo) / len(photo)
    detail = {"identity": ident, "mean_identity_destroying": md, "mean_photometric": mp}
    if md > mp:
        return GateResult(False, [
            f"INVERTED attack ordering: removing the brand mark costs less "
            f"({ident:.3f} -> {md:.3f}) than a photometric shift ({ident:.3f} -> {mp:.3f}). "
            "The corpus encodes brand identity in colour, not structure; robustness "
            "results on it describe the renderer, not the phenomenon."], detail,
            name="CORPUS VALIDITY GATE")
    return GateResult(True, [
        f"attack ordering is realistic: brand-mark removal ({md:.3f}) hurts at least "
        f"as much as photometric perturbation ({mp:.3f})"], detail,
        name="CORPUS VALIDITY GATE")


def split_leakage_gate(gallery_fp: Iterable[Hashable],
                       query_fp: Iterable[Hashable],
                       max_dup_frac: float = 0.02) -> GateResult:
    """Check that the evaluation split does not contain its own answers.

    A third failure mode, and one neither gate above can see: both were satisfied
    by a corpus in which 37.3% of query crops were *pixel-identical* to a gallery
    crop. The declared split control was sound -- gallery and query shared no
    phishing-kit family_id, 0 violations across 75 brands -- but family_id dedupes
    the page, not the logo. Distinct kits clone the same official brand mark, so
    different families still yield byte-identical crops. Retrieval then degenerates
    into looking up an identical image, which inflated clean top-1 from 0.668 to
    0.859.

    The lesson generalises past this corpus: a semantic dedup key does not imply
    the *artefact you actually feed the model* is unique. Fingerprint the tensors
    that reach the model, not the records they came from.
    """
    gal = set(gallery_fp)
    q = list(query_fp)
    if not q:
        return GateResult(False, ["empty query set"], name="SPLIT LEAKAGE GATE")
    dup = sum(1 for f in q if f in gal)
    frac = dup / len(q)
    detail = {"n_query": len(q), "n_duplicated": dup, "dup_frac": frac}
    if frac > max_dup_frac:
        return GateResult(False, [
            f"{dup}/{len(q)} query items ({frac:.1%}) are identical to a gallery "
            f"item, above the {max_dup_frac:.1%} budget; retrieval scores on this "
            "split are partly a lookup of the answer, not recognition"],
            detail, name="SPLIT LEAKAGE GATE")
    return GateResult(True, [
        f"{dup}/{len(q)} query items ({frac:.1%}) duplicated in gallery, "
        f"within the {max_dup_frac:.1%} budget"], detail, name="SPLIT LEAKAGE GATE")


if __name__ == "__main__":
    print("--- real-screenshot crucible (what we actually ran) ---")
    print(competence_gate(clean=0.299, attacked=0.017, chance=0.0043,
                          control_clean=0.134, control_attacked=0.032))
    print("\n--- synthetic crucible ---")
    print(competence_gate(clean=0.755, attacked=0.168, chance=0.0625))
    print("\n--- lexical pilot (the arm that was valid) ---")
    print(competence_gate(clean=0.9364, attacked=0.60, chance=0.5))
    print("\n=== corpus validity ===")
    print("synthetic corpus:")
    print(corpus_validity_gate({"identity": 0.755, "logo_occlude": 0.557, "logo_delete": 0.568,
                                "colour_shift": 0.135, "brightness": 0.130, "jpeg": 0.146,
                                "blur": 0.073, "grayscale": 0.167}))
    print("\nreal-screenshot corpus:")
    print(corpus_validity_gate({"identity": 0.299, "logo_occlude": 0.007, "logo_delete": 0.002,
                                "colour_shift": 0.010, "brightness": 0.009, "jpeg": 0.013,
                                "blur": 0.006, "grayscale": 0.043}))
