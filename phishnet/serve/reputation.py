"""A domain-reputation short-circuit, run before the neural model.

Every production phishing filter (Google Safe Browsing, Microsoft SmartScreen)
pairs its classifier with a reputation allowlist, for a sound reason: a learned
model, however good, will occasionally score a legitimate high-traffic login page
as suspicious because such pages genuinely share surface features with phishing
(the tokens "login"/"signin", long hostnames, deep subdomains). Flagging the real
Microsoft or PayPal login is far more damaging to user trust than missing one
obscure phish, so the reputable-domain case is decided by reputation, not by the
model's raw score.

The allowlist is keyed on the *registrable domain* (eTLD+1), which is the part an
attacker cannot forge. A look-alike such as `paypal.com.verify-account.tk` has
registrable domain `verify-account.tk`, so it never matches `paypal.com`; only a
URL genuinely served from the reputable registrable domain is short-circuited.
The list is drawn from the top of the Tranco research ranking, whose 30-day
averaging makes it robust to the transient spikes that let malicious domains onto
naive popularity lists.
"""
from __future__ import annotations

from pathlib import Path

from phishnet.features.lexical import registrable_domain


class ReputationAllowlist:
    def __init__(self, domains: set[str] | None = None):
        self.domains = domains or set()

    @classmethod
    def from_tranco(cls, path: str | Path, top_n: int = 10_000) -> "ReputationAllowlist":
        """Build from the first `top_n` entries of the benign Tranco URL list,
        with shared-infrastructure domains removed.

        A crucial subtlety: some of the most popular registrable domains are
        *shared platforms* -- free hosting (000webhostapp, netlify.app,
        vercel.app, github.io), dynamic DNS (duckdns.org, ngrok), and CDN edges --
        where the registrable owner does NOT control the subdomain content. These
        rank highly precisely because attackers and everyone else deploy on them.
        Allowlisting `duckdns.org` would wave through `apple-id-locked.duckdns.org`.
        We therefore drop any candidate whose registrable domain matches a known
        shared-hosting pattern; the model still judges those on their merits.
        """
        from phishnet.features.lexical import FREE_HOSTS

        doms: set[str] = set()
        p = Path(path)
        if p.exists():
            for i, line in enumerate(p.read_text(errors="ignore").splitlines()):
                if i >= top_n:
                    break
                d = registrable_domain(line.strip())
                if d and not any(h in d for h in FREE_HOSTS):
                    doms.add(d)
        return cls(doms)

    def is_trusted(self, url: str) -> bool:
        d = registrable_domain(url)
        return bool(d) and d in self.domains

    def __len__(self) -> int:
        return len(self.domains)
