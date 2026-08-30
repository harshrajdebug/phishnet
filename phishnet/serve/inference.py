"""Inference engine: load once, score many, explain on demand.

This is the object the API and the CLI both wrap. It degrades gracefully -- if a
branch checkpoint is missing it runs with the modalities it has, which is also
the deployment reality (a URL always exists; a screenshot may still be
rendering; email context is usually absent on a direct visit).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from phishnet.config import Config, resolve_device
from phishnet.features.url_encoding import encode_url
from phishnet.models.phishnet import PhishNet


class PhishNetEngine:
    def __init__(self, cfg: Config | None = None, ckpt: str | Path | None = None,
                 device: str | None = None):
        self.cfg = cfg or Config()
        self.device = device or resolve_device(self.cfg.train.device)
        self.model = PhishNet(self.cfg, fusion="cross_attention",
                              with_text=True, with_visual=True).to(self.device)
        self._tok = None
        self._brand_index = None
        self._brand_encoder = None
        self._classical = None
        self._reputation = None
        self._user_allow = None
        self.loaded_from = None

        ckpt = ckpt or (Path("results") / "phishnet_full.pt")
        if Path(ckpt).exists():
            state = torch.load(ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["state_dict"], strict=False)
            self.loaded_from = str(ckpt)
        self.model.eval()

        # Operating thresholds chosen on validation at a 1% false-alarm budget.
        # We keep one per modality configuration: a threshold calibrated on the
        # all-modalities-present distribution is far too low for a URL-only scan
        # (the dominant deployment case, run the instant a link is seen), and
        # would flag legitimate login pages. `self.threshold` is the URL-only
        # default; `predict` selects the matching one for the modalities present.
        self._thresholds = self._load_thresholds()
        self.threshold = self._thresholds.get("100", 0.5)

    def _load_thresholds(self) -> dict[str, float]:
        p = Path("results") / "deployment_thresholds.json"
        if p.exists():
            import json
            return {k: float(v) for k, v in json.loads(p.read_text()).items()}
        # Fallback: the single multimodal threshold, used for every config.
        q = Path("results") / "fusion_test_scores.npz"
        thr = float(np.load(q, allow_pickle=True)["threshold"]) if q.exists() else 0.5
        return {"100": thr, "101": thr, "110": thr, "111": thr}

    def _threshold_for(self, has_vis: bool, has_text: bool) -> float:
        key = f"1{int(has_vis)}{int(has_text)}"
        return self._thresholds.get(key, self.threshold)

    @property
    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.cfg.text.model_name)
        return self._tok

    @property
    def reputation(self):
        if self._reputation is None:
            from phishnet.serve.reputation import ReputationAllowlist
            tranco = Path(self.cfg.data.raw_dir) / "benign_urls.txt"
            self._reputation = ReputationAllowlist.from_tranco(tranco, top_n=10_000)
        return self._reputation

    _USER_ALLOW = Path("results") / "user_allowlist.json"

    @property
    def user_allowlist(self) -> set[str]:
        """User-managed trusted registrable domains, always decided as legitimate.

        Reputation covers the global top domains; this covers the sites *this
        user* trusts (a university portal, an intranet) that are too obscure to be
        globally ranked, so the model never blocks them. Persisted to a JSON file.
        """
        if getattr(self, "_user_allow", None) is None:
            import json
            try:
                self._user_allow = set(json.loads(self._USER_ALLOW.read_text()))
            except Exception:
                self._user_allow = set()
        return self._user_allow

    def add_trusted(self, domain: str) -> set[str]:
        from phishnet.features.lexical import registrable_domain
        import json
        d = registrable_domain(domain if "://" in domain else "http://" + domain) or domain.strip().lower()
        s = self.user_allowlist
        s.add(d)
        self._USER_ALLOW.write_text(json.dumps(sorted(s)))
        return s

    def remove_trusted(self, domain: str) -> set[str]:
        import json
        s = self.user_allowlist
        s.discard(domain.strip().lower())
        self._USER_ALLOW.write_text(json.dumps(sorted(s)))
        return s

    @property
    def brand_encoder(self):
        """The standalone visual encoder that built the brand index.

        The fused model carries its own (possibly older) visual encoder for the
        verdict; brand matching must query the index with the *same* encoder that
        produced the index embeddings, or the cosine similarities are meaningless.
        """
        if getattr(self, "_brand_encoder", None) is None:
            p = Path("results") / "visual_encoder_best.pt"
            if p.exists():
                from phishnet.models.visual_siamese import VisualEncoder
                enc = VisualEncoder(embed_dim=self.cfg.visual.embed_dim,
                                    out_dim=self.cfg.fusion.d_model)
                state = torch.load(p, map_location=self.device, weights_only=False)
                enc.load_state_dict(state["encoder_state"])
                self._brand_encoder = enc.to(self.device).eval()
            else:
                self._brand_encoder = self.model.visual_encoder
        return self._brand_encoder

    @property
    def brand_index(self):
        if self._brand_index is None:
            p = Path("results") / "brand_index.pt"
            if p.exists():
                from phishnet.models.visual_siamese import BrandIndex
                self._brand_index = BrandIndex.load(p)
        return self._brand_index

    @property
    def classical(self):
        if self._classical is None:
            p = Path("results") / "classical_baselines.joblib"
            if p.exists():
                import joblib
                self._classical = joblib.load(p)
        return self._classical

    def _prep(self, url: str, text: str | None, image_path: str | None):
        batch = {"url_ids": torch.from_numpy(
            encode_url(url, self.cfg.url.max_len)).unsqueeze(0).to(self.device)}
        has_text, has_vis = False, False
        if text and self.model.text_encoder is not None:
            enc = self.tokenizer(text, truncation=True, padding="max_length",
                                 max_length=self.cfg.text.max_len, return_tensors="pt")
            batch["input_ids"] = enc["input_ids"].to(self.device)
            batch["attention_mask"] = enc["attention_mask"].to(self.device)
            has_text = True
        if image_path and Path(image_path).exists() and self.model.visual_encoder is not None:
            from phishnet.train.train_visual import load_image
            batch["image"] = load_image(Path(image_path), self.cfg.visual.image_size)\
                .unsqueeze(0).to(self.device)
            has_vis = True
        batch["present"] = torch.tensor([[True, has_vis, has_text]], device=self.device)
        return batch, has_text, has_vis

    @torch.no_grad()
    def predict(self, url: str, text: str | None = None,
                image_path: str | None = None, explain: bool = False) -> dict:
        t0 = time.time()

        # Stage 1: reputation short-circuit. A URL genuinely served from a
        # top-ranked registrable domain is decided by reputation, not the model,
        # so a legitimate high-traffic login page is never flagged. Look-alikes
        # do not match here because the registrable domain is the attacker's, not
        # the brand's (see reputation.py).
        from phishnet.features.lexical import registrable_domain
        rd = registrable_domain(url)
        user_trusted = rd in self.user_allowlist
        if user_trusted or (self.reputation is not None and self.reputation.is_trusted(url)):
            src = "user_allowlist" if user_trusted else "reputation_allowlist"
            note = (f"'{rd}' is on your personal trusted list" if user_trusted
                    else f"'{rd}' is a well-established, high-reputation domain")
            out = {
                "url": url,
                "phishing_probability": 0.0,
                "verdict": "legitimate",
                "threshold": 0.5,
                "decided_by": src,
                "modalities_used": {"url": True, "visual": False, "text": False},
                "modality_gates": None,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
            if explain:
                out["explanation"] = {"summary": [note], "url_features": None,
                                      "brand_match": None}
            return out

        # Stage 2: the multimodal model.
        batch, has_text, has_vis = self._prep(url, text, image_path)
        logit, aux = self.model(batch, return_attention=True)
        score = float(torch.sigmoid(logit).item())
        gates = aux["gates"][0].cpu().tolist() if aux["gates"] is not None else None
        threshold = self._threshold_for(has_vis, has_text)
        verdict = "phishing" if score >= threshold else "legitimate"
        decided_by = "model"

        # Stage 3: single-signal safety escalation. Because the multimodal training
        # data was assembled class-consistently (malicious URLs never paired with
        # innocuous text), the fusion can let a benign-looking message veto a
        # clearly-malicious URL -- a false negative that is unacceptable for a
        # security tool. We therefore re-score the URL on its own; if the URL alone
        # is confidently phishing, that verdict stands regardless of what the other
        # modalities say. No strong evidence of phishing is silently overridden.
        url_only_score = score
        if has_text or has_vis:
            url_batch = {"url_ids": batch["url_ids"],
                         "present": torch.tensor([[True, False, False]], device=self.device)}
            url_logit, _ = self.model(url_batch, return_attention=True)
            url_only_score = float(torch.sigmoid(url_logit).item())
            url_thr = self._thresholds.get("100", 0.9)
            if url_only_score >= url_thr and verdict == "legitimate":
                verdict = "phishing"
                decided_by = "url_safety_override"
                score = max(score, url_only_score)

        out = {
            "url": url,
            "phishing_probability": round(score, 4),
            "verdict": verdict,
            "threshold": round(threshold, 4),
            "decided_by": decided_by,
            "url_only_probability": round(url_only_score, 4),
            "modalities_used": {"url": True, "visual": has_vis, "text": has_text},
            "modality_gates": ({"url": round(gates[0], 3), "visual": round(gates[1], 3),
                                "text": round(gates[2], 3)} if gates else None),
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
        if explain:
            out["explanation"] = self.explain(url, text, image_path, score,
                                               threshold if decided_by == "model" else 0.0)
            if decided_by == "url_safety_override":
                out["explanation"]["summary"].insert(
                    0, "Flagged on the URL alone: the link is strongly suspicious "
                       "regardless of the message text")
        return out

    def explain(self, url, text, image_path, score, threshold=None) -> dict:
        exp: dict = {"summary": [], "url_features": None, "brand_match": None}
        is_phish = score >= (threshold if threshold is not None else self.threshold)
        # URL feature-level SHAP via the classical model (human-readable axis).
        if self.classical is not None:
            try:
                from phishnet.explain.url_shap import feature_shap
                feats = feature_shap(self.classical["rf"], self.classical["scaler"],
                                     url, top_k=5)
                exp["url_features"] = feats
                # For a phishing verdict, surface the features pushing *toward*
                # phishing; for a legitimate one, surface those pushing toward
                # benign. Listing phishing cues under a "legitimate" verdict is
                # contradictory and erodes user trust in the panel.
                want = "phishing" if is_phish else "legitimate"
                for f in feats:
                    if f["direction"] == want and abs(f["shap"]) > 0.02:
                        exp["summary"].append(
                            _phrase(f) if is_phish else _benign_phrase(f))
            except Exception as e:
                exp["url_features_error"] = str(e)
        # Visual brand match.
        if image_path and self.brand_index is not None and Path(image_path).exists():
            try:
                from phishnet.train.train_visual import load_image
                img = load_image(Path(image_path), self.cfg.visual.image_size)\
                    .unsqueeze(0).to(self.device)
                z = self.brand_encoder(img)
                top = self.brand_index.query(z.cpu(), k=1)
                if top:
                    name, dom, sim = top[0]
                    exp["brand_match"] = {"brand": name, "legit_domain": dom,
                                          "similarity": round(sim, 3)}
                    from phishnet.features.lexical import registrable_domain
                    if sim > 0.8 and name.lower() not in registrable_domain(url).lower():
                        exp["summary"].append(
                            f"Page visually resembles {name} ({sim:.0%}) but the "
                            f"domain is not {dom}")
            except Exception as e:
                exp["brand_error"] = str(e)
        if not exp["summary"]:
            exp["summary"].append(
                "No single dominant signal; verdict from combined evidence"
                if is_phish else
                "URL structure looks normal; no strong phishing indicators")
        return exp


def _phrase(f: dict) -> str:
    """Map a feature name + value to a human sentence for the explanation panel."""
    name, v = f["feature"], f["value"]
    table = {
        "brand_outside_domain": "A known brand name appears outside the registered domain",
        "is_free_host": "Hosted on a free website/hosting platform",
        "tld_is_risky": "Uses a top-level domain frequently abused for phishing",
        "has_ip_host": "The host is a raw IP address instead of a domain name",
        "has_punycode": "The domain uses punycode (possible homoglyph attack)",
        "num_suspicious_tokens": f"Contains {int(v)} credential-related keyword(s)",
        "num_hyphens": f"Domain contains {int(v)} hyphens (look-alike construction)",
        "hostname_entropy": "Hostname characters look randomly generated",
        "has_at_symbol": "Contains an '@' which can hide the true destination",
        "num_subdomains": f"Unusually deep subdomain chain ({int(v)} levels)",
        # Fallbacks so no raw "feature = value" ever reaches the user.
        "is_https": ("Served over plain HTTP, not HTTPS" if v == 0.0
                     else "Served over HTTPS"),
        "hostname_length": f"Long hostname ({int(v)} characters)",
        "url_length": f"Unusually long URL ({int(v)} characters)",
        "path_length": f"Long URL path ({int(v)} characters)",
        "num_digits": f"Contains {int(v)} digits, common in generated hosts",
        "digit_ratio": "High proportion of digits in the URL",
        "special_ratio": "High proportion of special characters in the URL",
        "num_dots": f"Contains {int(v)} dots (many subdomain separators)",
        "num_slashes": f"Contains {int(v)} path separators",
        "vowel_ratio": "Vowel pattern atypical of real words in the hostname",
        "longest_token_len": f"Contains a very long token ({int(v)} chars)",
        "has_hex_encoding": "URL uses percent/hex encoding to obscure content",
        "num_embedded_urls": "Another URL is embedded inside this one (redirect)",
        "has_double_slash_redirect": "Path contains '//', a redirect/prefix trick",
        "port_explicit": "Connects on a non-standard explicit port",
        "path_depth": f"Deep path structure ({int(v)} levels)",
        "num_params": f"Carries {int(v)} query parameters",
        "query_length": f"Long query string ({int(v)} characters)",
        "brand_in_path": "A brand name appears in the URL path",
        "url_entropy": "Overall URL characters look randomly generated",
    }
    phrase = table.get(name)
    if phrase is not None:
        return phrase
    # Last-resort: humanise the feature name rather than dumping the raw value.
    return name.replace("_", " ").capitalize()


def _benign_phrase(f: dict) -> str:
    """Reassuring counterpart of `_phrase` for features supporting a benign verdict."""
    name = f["feature"]
    table = {
        "is_https": "Served securely over HTTPS",
        "brand_outside_domain": "No brand impersonation detected in the domain",
        "is_free_host": "Hosted on established infrastructure, not a free platform",
        "tld_is_risky": "Uses a mainstream, reputable top-level domain",
        "hostname_entropy": "Hostname reads like real words, not random characters",
        "num_suspicious_tokens": "No credential-related keywords in the URL",
        "has_ip_host": "Uses a proper domain name, not a raw IP address",
        "url_length": "URL length is typical of legitimate pages",
        "num_hyphens": "Clean domain without look-alike hyphenation",
    }
    return table.get(name, "URL structure looks normal")
