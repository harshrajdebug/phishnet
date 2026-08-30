"""Hand-engineered URL features.

These serve two purposes: they are the input to the classical baselines the
neural model is measured against, and they are the human-readable axis for the
SHAP explanations. A user shown "the domain was registered as an IP literal and
the hostname contains 4 hyphens" understands the verdict; a user shown a
convolution activation does not.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse, unquote

import numpy as np

# Brands that phishing kits clone most often, per APWG's brand tracking. Used
# only to test whether a brand token appears *outside* the registrable domain,
# which is the classic look-alike construction.
BRANDS = (
    "paypal", "apple", "microsoft", "office365", "outlook", "netflix", "amazon",
    "google", "facebook", "instagram", "whatsapp", "linkedin", "dhl", "fedex",
    "usps", "chase", "wellsfargo", "hsbc", "barclays", "santander", "coinbase",
    "binance", "metamask", "steam", "roblox", "docusign", "dropbox", "icloud",
    "verizon", "att", "tmobile", "vodafone", "airtel", "jio",
)

SUSPICIOUS_TOKENS = (
    "login", "signin", "secure", "verify", "verification", "account", "update",
    "confirm", "webscr", "banking", "password", "credential", "authenticate",
    "wallet", "recover", "unlock", "suspend", "billing", "invoice", "payment",
)

# TLDs with cheap or free registration and weak abuse handling; consistently
# over-represented in phishing telemetry relative to their share of the DNS.
RISKY_TLDS = (
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "buzz", "click", "link", "work",
    "support", "zip", "mov", "rest", "cfd", "sbs", "icu", "cyou", "bond",
)

# Free hosting and page-publishing platforms that phishing kits deploy onto.
FREE_HOSTS = (
    "000webhostapp", "netlify.app", "vercel.app", "pages.dev", "web.app",
    "firebaseapp", "github.io", "glitch.me", "repl.co", "weebly", "wixsite",
    "blogspot", "herokuapp", "r2.dev", "workers.dev", "azurewebsites",
    "sharepoint.com", "duckdns", "ngrok", "trycloudflare",
)

IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
HEX_IP_RE = re.compile(r"^0x[0-9a-f]+$", re.I)
PUNYCODE_RE = re.compile(r"xn--", re.I)

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_slashes", "num_digits", "num_params",
    "num_subdomains", "digit_ratio", "special_ratio", "url_entropy",
    "hostname_entropy", "has_ip_host", "has_punycode", "has_at_symbol",
    "has_double_slash_redirect", "is_https", "port_explicit",
    "tld_is_risky", "is_free_host", "brand_outside_domain", "brand_in_path",
    "num_suspicious_tokens", "has_hex_encoding", "longest_token_len",
    "vowel_ratio", "num_embedded_urls", "path_depth",
]


def _entropy(s: str) -> float:
    """Shannon entropy over characters; algorithmically generated hostnames
    (DGA-style, or random kit subdomains) sit noticeably higher than English."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _registrable(host: str) -> str:
    """Best-effort eTLD+1 without a network call to the public suffix list.

    Handles the common two-label suffixes (co.uk, com.br) explicitly. This is an
    approximation, but it is applied identically to every sample, so it cannot
    bias one class relative to the other.
    """
    parts = host.split(".")
    if len(parts) < 2:
        return host
    two_label = {"co", "com", "net", "org", "gov", "edu", "ac", "or", "ne"}
    if len(parts) >= 3 and parts[-2] in two_label and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def extract(url: str) -> np.ndarray:
    """Return the fixed-order feature vector described by FEATURE_NAMES."""
    u = url.strip()
    dec = unquote(u)
    try:
        p = urlparse(u if "://" in u else "http://" + u)
    except Exception:
        p = urlparse("http://invalid")

    host = (p.hostname or "").lower()
    path = p.path or ""
    query = p.query or ""
    reg = _registrable(host)
    # The subdomain chain is where look-alike brands get hidden:
    # paypal.com.secure-login.attacker.tk
    sub = host[: -len(reg)].rstrip(".") if reg and host.endswith(reg) else ""
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    digits = sum(c.isdigit() for c in u)
    specials = sum(not c.isalnum() for c in u)
    vowels = sum(c in "aeiou" for c in host)
    tokens = re.split(r"[^a-z0-9]+", host + path.lower())
    tokens = [t for t in tokens if t]

    # Brand matching is token-based, not substring-based. A raw substring test
    # produces silent false negatives -- "att" matches inside "attacker", so a
    # paypal-in-subdomain / attacker.tk-registrable URL would wrongly cancel the
    # impersonation flag. We match a brand only against whole dot/hyphen tokens.
    sub_tokens = set(re.split(r"[^a-z0-9]+", sub))
    reg_tokens = set(re.split(r"[^a-z0-9]+", reg))
    path_tokens = set(re.split(r"[^a-z0-9]+", path.lower()))
    brand_in_sub = any(b in sub_tokens for b in BRANDS)
    brand_in_reg = any(b in reg_tokens for b in BRANDS)
    brand_in_path_tok = any(b in path_tokens for b in BRANDS)

    f = {
        "url_length": len(u),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": u.count("."),
        "num_hyphens": u.count("-"),
        "num_slashes": u.count("/"),
        "num_digits": digits,
        "num_params": query.count("=") if query else 0,
        "num_subdomains": len([x for x in sub.split(".") if x]),
        "digit_ratio": digits / max(len(u), 1),
        "special_ratio": specials / max(len(u), 1),
        "url_entropy": _entropy(u),
        "hostname_entropy": _entropy(host),
        "has_ip_host": float(bool(IP_RE.match(host)) or bool(HEX_IP_RE.match(host))),
        "has_punycode": float(bool(PUNYCODE_RE.search(host))),
        "has_at_symbol": float("@" in u),
        # "//" after the scheme is the open-redirect / credential-prefix trick
        "has_double_slash_redirect": float("//" in u[8:]),
        "is_https": float(p.scheme == "https"),
        "port_explicit": float(p.port is not None),
        "tld_is_risky": float(tld in RISKY_TLDS),
        "is_free_host": float(any(h in host for h in FREE_HOSTS)),
        # A brand name in the subdomain or path but NOT in the registrable
        # domain is the single strongest lexical tell of impersonation.
        "brand_outside_domain": float((brand_in_sub or brand_in_path_tok)
                                      and not brand_in_reg),
        "brand_in_path": float(brand_in_path_tok),
        "num_suspicious_tokens": sum(t in SUSPICIOUS_TOKENS for t in tokens),
        "has_hex_encoding": float("%" in u and bool(re.search(r"%[0-9a-f]{2}", u, re.I))),
        "longest_token_len": max((len(t) for t in tokens), default=0),
        "vowel_ratio": vowels / max(len(host), 1),
        # A second scheme inside the URL means an embedded redirect target
        "num_embedded_urls": max(dec.lower().count("http"), 1) - 1,
        "path_depth": len([x for x in path.split("/") if x]),
    }
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)


def extract_batch(urls: list[str]) -> np.ndarray:
    if not urls:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.stack([extract(u) for u in urls])


def registrable_domain(url: str) -> str:
    """Public helper: the grouping key for leakage-free dataset splits."""
    try:
        p = urlparse(url if "://" in url else "http://" + url)
        return _registrable((p.hostname or "").lower())
    except Exception:
        return ""
