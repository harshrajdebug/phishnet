"""An attacker-cost model over the lexical URL feature space.

Each feature is priced on three axes, all in [0,1]:

  c_money       cost to change: domain registration, hosting, certificates
  c_effort      engineering time / whether a kit automates it
  c_conversion  *how much changing it costs the attacker in victim deception*

The third axis is the one the literature omits, and it is what makes some
evasions self-defeating. Removing the brand token from `paypal-secure.tk` beats a
brand-matching detector for free -- but it also removes the cue the victim relies
on to be fooled. An attacker who strips every deceptive signal has evaded the
detector by ceasing to phish.

Scores are grounded in what an attacker actually faces in 2026: TLS is free
(Let's Encrypt), subdomains are free (attacker owns the zone), a fresh cheap-TLD
domain is ~$1-10, and leaving free hosting for reputable infrastructure costs
both money and an abuse-resistant identity.
"""
from __future__ import annotations

import numpy as np

from phishnet.features.lexical import FEATURE_NAMES

# feature -> (c_money, c_effort, c_conversion)
COST_TABLE: dict[str, tuple[float, float, float]] = {
    # --- free string edits: attacker controls the path/query entirely ---
    "url_length":                (0.00, 0.05, 0.05),
    "path_length":               (0.00, 0.05, 0.02),
    "query_length":              (0.00, 0.05, 0.02),
    "num_slashes":               (0.00, 0.05, 0.02),
    "num_params":                (0.00, 0.05, 0.02),
    "path_depth":                (0.00, 0.05, 0.02),
    "num_digits":                (0.00, 0.05, 0.05),
    "digit_ratio":               (0.00, 0.05, 0.05),
    "special_ratio":             (0.00, 0.05, 0.05),
    "url_entropy":               (0.00, 0.10, 0.05),
    "longest_token_len":         (0.00, 0.05, 0.02),
    "has_hex_encoding":          (0.00, 0.10, 0.05),
    "num_embedded_urls":         (0.00, 0.10, 0.05),
    "has_at_symbol":             (0.00, 0.05, 0.05),
    "has_double_slash_redirect": (0.00, 0.05, 0.05),
    "has_punycode":              (0.00, 0.10, 0.10),
    "port_explicit":             (0.00, 0.05, 0.05),
    # --- free, because the attacker owns their own DNS zone ---
    "num_subdomains":            (0.00, 0.10, 0.05),
    "num_dots":                  (0.00, 0.10, 0.05),
    # --- cheap but not free: needs a new registration ---
    "hostname_length":           (0.25, 0.20, 0.05),
    "hostname_entropy":          (0.25, 0.25, 0.10),
    "vowel_ratio":               (0.25, 0.25, 0.10),
    "num_hyphens":               (0.20, 0.15, 0.10),
    "tld_is_risky":              (0.30, 0.15, 0.05),   # ~$1-10 for a better TLD
    "is_https":                  (0.05, 0.20, 0.00),   # free certs; near-zero cost
    # --- expensive: real infrastructure and identity ---
    "is_free_host":              (0.70, 0.45, 0.05),   # leave free hosting entirely
    "has_ip_host":               (0.30, 0.20, 0.15),
    # --- free to remove, but the attacker LOSES the victim (c_conversion) ---
    "brand_outside_domain":      (0.00, 0.05, 0.90),   # drop the brand -> no lure
    "brand_in_path":             (0.00, 0.05, 0.70),
    "num_suspicious_tokens":     (0.00, 0.05, 0.75),   # drop "login"/"verify"
}

DEFAULT_W = np.array([0.45, 0.20, 0.35])   # money, effort, conversion


def cost_vectors() -> np.ndarray:
    """(n_features, 3) matrix aligned to FEATURE_NAMES."""
    return np.array([COST_TABLE.get(f, (0.3, 0.3, 0.3)) for f in FEATURE_NAMES],
                    dtype=np.float64)


def feature_costs(w: np.ndarray | None = None, floor: float = 0.02) -> np.ndarray:
    """Scalar cost per feature. `floor` keeps 1/cost finite for free features."""
    w = DEFAULT_W if w is None else np.asarray(w, dtype=np.float64)
    c = cost_vectors() @ w
    return np.maximum(c, floor)


def perturb_weights(rng: np.random.Generator, pct: float) -> np.ndarray:
    """Randomly perturb the weight vector by +/- pct, for sensitivity analysis."""
    noise = rng.uniform(1 - pct, 1 + pct, size=3)
    w = DEFAULT_W * noise
    return w / w.sum() * DEFAULT_W.sum()


def summary() -> str:
    c = feature_costs()
    order = np.argsort(c)
    lines = ["cheapest to manipulate:"]
    lines += [f"    {FEATURE_NAMES[i]:26s} {c[i]:.3f}" for i in order[:6]]
    lines.append("  most expensive:")
    lines += [f"    {FEATURE_NAMES[i]:26s} {c[i]:.3f}" for i in order[-6:]]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
