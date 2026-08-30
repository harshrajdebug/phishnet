"""Recent-lure corpus for the text branch.

Motivation: the freely available phishing-email corpora (Nazario, SpamAssassin,
Enron, TREC, CEAS) are all 2001-2008. A model trained only on them learns the
vocabulary of 2000s scams (lottery, Nigerian-prince, Viagra) and is blind to the
themes that dominate 2024-2026 telemetry: MFA/one-time-code interception,
tax/tariff-refund lures, package-redelivery fees, crypto-wallet drainers, and
SIM-deactivation scams. This module assembles a *recent* lure corpus so the
branch is exposed to current language.

Provenance is labelled explicitly, because honesty about data is the whole point
of this project:

  `smishing_current`  Real-schema smishing messages reflecting 2024-2026 scam
                      taxonomy (source: a public current-theme smishing dataset).
                      Machine-generated, not captured attacks; kept because the
                      *themes and phrasing* are current.
  `curated_2026`      A small, hand-written set of global-brand lures and matched
                      legitimate transactional messages, each grounded in a
                      specific 2026 threat pattern documented in the APWG Q1 2026
                      report, the Verizon 2026 DBIR, and 2026 AiTM/quishing
                      reporting. Templated, and marked as such.

Neither portion is presented as captured real attacks. The recent-lure test
split (built in train_text) lets us measure whether adding current-theme data
improves detection on current-theme messages, which is the claim we can honestly
make.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

# ---- Global brands and the 2026 themes they are impersonated under -----------
# Each template carries a {brand}/{code}/{amount}/{link} slot. The phishing and
# legitimate templates share brands and registers on purpose, so the classifier
# cannot separate the classes on brand name or SMS-vs-email style alone -- it has
# to learn intent.

BRANDS = ["PayPal", "Apple", "Microsoft", "Netflix", "Amazon", "DHL", "FedEx",
          "USPS", "Chase", "Coinbase", "Wells Fargo", "HSBC", "Verizon", "AT&T",
          "IRS", "Instagram", "WhatsApp", "Google", "Disney+", "Spotify"]

# Phishing templates, grouped by the 2026 theme they instantiate.
PHISH_TEMPLATES = {
    "mfa_intercept": [   # AiTM / session-token era: harvest the one-time code
        "{brand}: A login was detected from a new device. If this wasn't you, verify now: {link}",
        "{brand} security: your verification code is {code}. Reply with this code to stop the login attempt.",
        "Your {brand} account access is on hold. Confirm your identity to restore it: {link}",
        "{brand}: we blocked a sign-in. Approve or deny at {link} within 10 minutes.",
    ],
    "tax_tariff_refund": [  # APWG Q1 2026: tariff-refund + tax-refund scam surge
        "{brand} Tax Dept: you are eligible for a refund of ${amount}. Claim before it expires: {link}",
        "IRS notice: your {amount} tariff refund is pending. Submit your details to release payment: {link}",
        "You have an unclaimed refund of ${amount}. Verify your bank to receive it: {link}",
    ],
    "package_redelivery": [  # delivery-fee smishing, top mobile theme
        "{brand}: your package could not be delivered. Pay the ${amount} redelivery fee: {link}",
        "{brand} tracking: address incomplete. Update details to reschedule delivery: {link}",
        "Your {brand} parcel is held at the depot. A ${amount} customs fee is due: {link}",
    ],
    "crypto_drainer": [   # wallet-connect drainers, current crypto scam
        "{brand}: unusual activity on your wallet. Re-validate your seed phrase now: {link}",
        "{brand} alert: withdraw ${amount} pending. Confirm your wallet to release funds: {link}",
        "Claim your ${amount} {brand} airdrop before it expires: {link}",
    ],
    "account_suspension": [
        "{brand}: your account will be permanently suspended in 24h. Confirm to keep it active: {link}",
        "{brand} billing failed. Update your payment method to avoid service interruption: {link}",
        "Your {brand} subscription is on hold due to a billing issue. Fix it here: {link}",
    ],
    "sim_swap": [   # telecom-sector surge (APWG: telecom 5.9% -> 33%)
        "{brand}: your SIM will be deactivated in 48h. Verify your identity to keep your number: {link}",
        "{brand}: a SIM transfer was requested on your line. Cancel it now: {link}",
    ],
}

# Legitimate transactional templates, same brands, deliberately similar surface.
LEGIT_TEMPLATES = [
    "{brand}: your verification code is {code}. Do not share it with anyone.",
    "{brand}: your payment of ${amount} was received. Thank you.",
    "{brand}: your order has shipped and will arrive by Friday. Track in the app.",
    "{brand}: your subscription renews on the 3rd. Manage it in Account settings.",
    "{brand}: you signed in on a new device. If this was you, no action is needed.",
    "{brand}: your statement is ready. View it by logging in to the app.",
    "{brand}: your package was delivered. Reply STOP to opt out of alerts.",
    "{brand} receipt: ${amount} charged to your card ending 4471.",
    "{brand}: your appointment is confirmed for Tuesday at 2pm.",
    "{brand}: thanks for your payment. Your balance is now ${amount}.",
]

# Deliberately benign-looking links vs. off-brand phishing links.
def _phish_link(brand: str, rng: random.Random) -> str:
    slug = re.sub(r"[^a-z]", "", brand.lower())[:8] or "secure"
    host = rng.choice([
        f"{slug}-verify.{rng.choice(['tk','xyz','top','cfd','sbs','online'])}",
        f"{slug}-secure-login.{rng.choice(['com-account.co','app-verify.net'])}",
        f"{slug}.{rng.choice(['account-update','id-confirm','billing-alert'])}.com",
        f"{rng.choice(['bit','tinyurl','cutt'])}.ly/{slug}{rng.randint(100,999)}",
    ])
    return f"http://{host}/{rng.choice(['verify','signin','update','confirm'])}"


def build(seed: int = 1337) -> dict[str, list[str]]:
    """Return {'phish': [...], 'legit': [...]} for the curated 2026 set."""
    rng = random.Random(seed)
    phish, legit = [], []

    for _ in range(1800):
        brand = rng.choice(BRANDS)
        theme = rng.choice(list(PHISH_TEMPLATES))
        tmpl = rng.choice(PHISH_TEMPLATES[theme])
        msg = tmpl.format(brand=brand, code=rng.randint(100000, 999999),
                          amount=rng.choice([19.99, 49, 129, 250, 500, 1450, 2500]),
                          link=_phish_link(brand, rng))
        phish.append(msg)

    for _ in range(1800):
        brand = rng.choice(BRANDS)
        tmpl = rng.choice(LEGIT_TEMPLATES)
        msg = tmpl.format(brand=brand, code=rng.randint(100000, 999999),
                          amount=rng.choice([9.99, 15, 42, 88, 120, 340]))
        legit.append(msg)

    phish = list(dict.fromkeys(phish))
    legit = list(dict.fromkeys(legit))
    return {"phish": phish, "legit": legit}


def load_recent(raw_dir: Path) -> dict[str, list[str]]:
    """Combine the downloaded current-theme smishing set with the curated 2026 set.

    Returns {'phish': [...], 'legit': [...]}. Missing sources are skipped so the
    function still works if only one is present.
    """
    phish, legit = [], []
    rec = Path(raw_dir) / "recent"

    sp = rec / "smish_current_phish.txt"
    sl = rec / "smish_current_legit.txt"
    if sp.exists():
        phish += [l.strip() for l in sp.read_text(errors="ignore").splitlines() if len(l.strip()) > 15]
    if sl.exists():
        legit += [l.strip() for l in sl.read_text(errors="ignore").splitlines() if len(l.strip()) > 15]

    curated = build()
    phish += curated["phish"]
    legit += curated["legit"]

    phish = list(dict.fromkeys(phish))
    legit = list(dict.fromkeys(legit))
    return {"phish": phish, "legit": legit}


if __name__ == "__main__":
    from phishnet.config import Config
    r = load_recent(Path(Config().data.raw_dir))
    print(f"recent phish: {len(r['phish']):,}  recent legit: {len(r['legit']):,}")
    for m in r["phish"][:3]:
        print("  P:", m[:90])
    for m in r["legit"][:3]:
        print("  L:", m[:90])
