"""Character-level encoding of URLs for the 1D-CNN branch.

Word-level tokenisation fails on this task: phishing hostnames are deliberately
built from non-words ("paypa1-secure-verify", "000webhostapp"), so any fixed
vocabulary is mostly out-of-vocabulary exactly where the signal is. Operating on
characters keeps the model sensitive to the substitutions and padding that
define the attack surface.
"""
from __future__ import annotations

import numpy as np

# Printable ASCII that actually occurs in URLs. Anything outside this -- most
# importantly the Unicode homoglyphs used in IDN attacks -- maps to <UNK>, which
# is itself a strong signal and is preserved rather than silently dropped.
ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "-._~:/?#[]@!$&'()*+,;=%"
    '"<>\\^`{|} '
)

PAD, UNK = 0, 1
STOI = {c: i + 2 for i, c in enumerate(ALPHABET)}
ITOS = {i + 2: c for i, c in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET) + 2


def encode_url(url: str, max_len: int = 200) -> np.ndarray:
    """Map a URL to a fixed-length array of character ids.

    Long URLs are truncated from the *left*, keeping the tail. Attackers pad the
    path with legitimate-looking tokens ("...?https://login.yahoo.com/?.src=ym")
    and the discriminative part is often the final redirect target, so keeping
    the tail beats keeping the head.
    """
    u = url.strip().lower()
    if len(u) > max_len:
        u = u[-max_len:]
    ids = np.full(max_len, PAD, dtype=np.int64)
    for i, ch in enumerate(u):
        ids[i] = STOI.get(ch, UNK)
    return ids


def encode_batch(urls: list[str], max_len: int = 200) -> np.ndarray:
    out = np.full((len(urls), max_len), PAD, dtype=np.int64)
    for i, u in enumerate(urls):
        out[i] = encode_url(u, max_len)
    return out


def decode(ids) -> str:
    """Inverse of `encode_url`, used to render SHAP attributions back over text."""
    return "".join(ITOS.get(int(i), "" if int(i) == PAD else "�") for i in ids)
