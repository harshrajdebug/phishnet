"""Fast, dependency-light tests for the load-bearing invariants.

These are the properties that, if silently broken, would invalidate the results
without throwing an error: the leakage-free split, the character encoding round
trip, the fusion missing-modality masking, and the metric behaviour under
imbalance.
"""
import numpy as np
import torch

from phishnet.config import Config
from phishnet.features.url_encoding import encode_url, decode, VOCAB_SIZE
from phishnet.features.lexical import extract, FEATURE_NAMES, registrable_domain
from phishnet.data.datasets import domain_disjoint_split, apply_base_rate, Split
from phishnet.eval.metrics import compute


def test_url_encoding_roundtrip():
    ids = encode_url("http://paypa1-secure.tk/login", max_len=64)
    assert ids.shape == (64,)
    assert ids.max() < VOCAB_SIZE
    assert "paypa1-secure" in decode(ids)


def test_lexical_feature_shape_and_signal():
    v = extract("http://paypal.com.secure-login.attacker.tk/verify")
    assert v.shape == (len(FEATURE_NAMES),)
    i = FEATURE_NAMES.index("brand_outside_domain")
    assert v[i] == 1.0                 # paypal outside the registrable domain
    j = FEATURE_NAMES.index("tld_is_risky")
    assert v[j] == 1.0                 # .tk

    # Regression: a brand substring inside the registrable domain (e.g. "att"
    # inside "attacker") must not cancel the impersonation flag.
    k = FEATURE_NAMES.index("brand_outside_domain")
    assert extract("https://att.com/account")[k] == 0.0        # att is legit here
    assert extract("http://apple-login.attacker.tk/")[k] == 1.0


def test_registrable_domain():
    assert registrable_domain("https://a.b.example.co.uk/x") == "example.co.uk"
    assert registrable_domain("http://sub.attacker.tk/") == "attacker.tk"


def test_domain_disjoint_split_has_no_leakage():
    # Two domains, many URLs each; no domain may appear in two splits.
    urls, labels = [], []
    for d in [f"d{i}.com" for i in range(400)]:
        for k in range(5):
            urls.append(f"http://{d}/p{k}")
            labels.append(1 if int(d[1:-4]) % 2 else 0)
    sp = domain_disjoint_split(urls, np.array(labels), 0.15, 0.15)
    doms = {k: {registrable_domain(u) for u in s.urls} for k, s in sp.items()}
    assert not (doms["train"] & doms["test"])
    assert not (doms["train"] & doms["val"])
    assert not (doms["val"] & doms["test"])


def test_base_rate_downsampling():
    urls = [f"http://d{i}.com/p" for i in range(10000)]
    labels = np.array([1] * 5000 + [0] * 5000)
    sp = Split(urls, labels)
    out = apply_base_rate(sp, 0.02)
    assert abs(out.pos_rate - 0.02) < 0.01


def test_fusion_masks_missing_modalities():
    from phishnet.models.fusion import CrossModalFusion
    f = CrossModalFusion().eval()
    u = torch.randn(3, 256)
    v = torch.randn(3, 256)
    t = torch.randn(3, 256)
    present = torch.tensor([[1, 0, 0], [1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    _, aux = f(u, v, t, present, return_attention=True)
    # A modality marked absent must receive zero gate weight.
    assert aux["gates"][0, 1].item() == 0.0
    assert aux["gates"][0, 2].item() == 0.0
    assert abs(aux["gates"][0].sum().item() - 1.0) < 1e-4


def test_metrics_imbalance_behaviour():
    y = np.r_[np.ones(50), np.zeros(4950)]
    perfect = np.r_[np.ones(50), np.zeros(4950)]
    m = compute(y, perfect * 0.99 + 0.005)
    assert m["recall"] == 1.0
    assert m["fpr"] == 0.0
    # Predicting all-benign scores high accuracy but zero recall -- the trap.
    allben = compute(y, np.zeros(5000) + 0.01)
    assert allben["accuracy"] > 0.98
    assert allben["recall"] == 0.0


def test_url_encoder_forward():
    from phishnet.models.url_cnn import URLClassifier
    m = URLClassifier()
    x = torch.randint(0, 50, (4, 200))
    assert m(x).shape == (4,)


def test_reputation_allowlist_logic():
    """The allowlist must trust a brand's real registrable domain, reject a
    look-alike that only *contains* the brand, and reject shared-hosting domains
    even when they are individually popular."""
    from phishnet.serve.reputation import ReputationAllowlist

    al = ReputationAllowlist({"paypal.com", "microsoftonline.com", "google.com"})
    # Real domain (with subdomain / path) is trusted.
    assert al.is_trusted("https://login.microsoftonline.com/")
    assert al.is_trusted("https://www.paypal.com/signin")
    # Look-alike: registrable domain is the attacker's, not the brand's.
    assert not al.is_trusted("http://paypal.com.verify-account.tk/signin")
    assert not al.is_trusted("http://secure-paypal.tk/login")
    # A domain simply not on the list is not trusted (fail closed to the model).
    assert not al.is_trusted("http://random-site-12345.xyz/")


def test_reputation_excludes_shared_hosting(tmp_path):
    """Building from a ranking must drop free-hosting / dynamic-DNS domains so
    their attacker-controlled subdomains still reach the model."""
    from phishnet.serve.reputation import ReputationAllowlist

    ranking = tmp_path / "rank.txt"
    ranking.write_text("\n".join([
        "https://google.com",
        "https://duckdns.org",        # dynamic DNS -- must be excluded
        "https://netlify.app",        # free hosting -- must be excluded
        "https://000webhostapp.com",  # free hosting -- must be excluded
        "https://paypal.com",
    ]))
    al = ReputationAllowlist.from_tranco(ranking, top_n=100)
    assert al.is_trusted("https://google.com")
    assert al.is_trusted("https://paypal.com/x")
    assert not al.is_trusted("http://apple-id-locked.duckdns.org/unlock")
    assert not al.is_trusted("http://x.netlify.app/login")
    assert not al.is_trusted("http://y.000webhostapp.com/paypal")


def test_competence_gate_rejects_noise_floor_baseline():
    """The real-screenshot crucible must be rejected: its baseline was at the
    noise floor AND less robust than untrained ImageNet features."""
    from phishnet.eval.competence_gate import competence_gate

    r = competence_gate(clean=0.299, attacked=0.017, chance=0.0043,
                        control_clean=0.134, control_attacked=0.032)
    assert not r.passed
    assert any("noise floor" in x for x in r.reasons)
    assert any("LESS robust than the untrained control" in x for x in r.reasons)


def test_competence_gate_accepts_valid_lexical_pilot():
    """The gate must be scale invariant: a balanced binary task has chance 0.5,
    where a naive multiple-of-chance threshold is unsatisfiable."""
    from phishnet.eval.competence_gate import competence_gate

    r = competence_gate(clean=0.9364, attacked=0.60, chance=0.5)
    assert r.passed, r.reasons


def test_corpus_validity_gate_catches_colour_dominated_corpus():
    """Our synthetic corpus passed competence but was invalid: deleting the brand
    mark hurt LESS than a colour shift, which is inverted from real pages."""
    from phishnet.eval.competence_gate import corpus_validity_gate

    synthetic = {"identity": 0.755, "logo_occlude": 0.557, "logo_delete": 0.568,
                 "colour_shift": 0.135, "brightness": 0.130, "jpeg": 0.146,
                 "blur": 0.073, "grayscale": 0.167}
    assert not corpus_validity_gate(synthetic).passed

    real = {"identity": 0.299, "logo_occlude": 0.007, "logo_delete": 0.002,
            "colour_shift": 0.010, "brightness": 0.009, "jpeg": 0.013,
            "blur": 0.006, "grayscale": 0.043}
    assert corpus_validity_gate(real).passed


def test_split_leakage_gate_catches_duplicate_crops():
    """The Phishpedia logo corpus satisfied both other gates while 37.3% of query
    crops were pixel-identical to a gallery crop. family_id dedupes the phishing
    kit, not the cloned brand mark, so a sound-looking split still leaked."""
    from phishnet.eval.competence_gate import split_leakage_gate

    gallery = [f"h{i}" for i in range(100)]
    leaking = [f"h{i}" for i in range(37)] + [f"q{i}" for i in range(63)]
    r = split_leakage_gate(gallery, leaking)
    assert not r.passed
    assert abs(r.detail["dup_frac"] - 0.37) < 1e-9
    assert any("lookup of the answer" in x for x in r.reasons)

    clean = [f"q{i}" for i in range(100)]
    assert split_leakage_gate(gallery, clean).passed


def test_gate_results_are_labelled_by_gate():
    """A corpus-validity result must not print itself as a competence result."""
    from phishnet.eval.competence_gate import (competence_gate,
                                               corpus_validity_gate,
                                               split_leakage_gate)

    assert str(competence_gate(0.9, 0.6, 0.5)).startswith("COMPETENCE GATE:")
    cv = corpus_validity_gate({"identity": 0.7, "logo_occlude": 0.1,
                               "logo_delete": 0.05, "colour_shift": 0.6,
                               "brightness": 0.6, "jpeg": 0.6, "blur": 0.5,
                               "grayscale": 0.4})
    assert str(cv).startswith("CORPUS VALIDITY GATE:")
    assert str(split_leakage_gate(["a"], ["b"])).startswith("SPLIT LEAKAGE GATE:")


def test_phishpedia_crop_attacks_preserve_shape_and_range():
    """Crop attacks are re-defined from the full-page versions: on a crop the mark
    IS the image, so masking a top band would not be brand-mark removal."""
    import numpy as np
    import torch
    from phishnet.eval.phishpedia_crucible import CROP_ATTACKS, apply_crop_attack

    rng = np.random.default_rng(0)
    x = torch.rand(3, 64, 64)
    for a in CROP_ATTACKS:
        y = apply_crop_attack(x, a, rng)
        assert y.shape == x.shape, a
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0, a
    # logo_delete must remove strictly more of the mark than logo_occlude
    occ = apply_crop_attack(x, "logo_occlude", rng)
    dele = apply_crop_attack(x, "logo_delete", rng)
    assert (dele != x).sum() > (occ != x).sum()
