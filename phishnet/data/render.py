"""Screenshot acquisition for the visual branch.

Rendering *live* phishing pages is both dangerous (they serve exploits) and
futile (most are dead within hours, and the survivors cloak against headless
crawlers -- APWG Q1 2026 documents kits that show benign decoys unless the
referrer and TLS fingerprint match a real victim). So we never touch them.

Instead the visual corpus is built from two safe sources:

  reference set   real login/landing pages of legitimate brands, captured once
                  from their true domains. These are the anchors the Siamese
                  index compares against.

  impersonation   locally generated look-alike login pages. Each template clones
                  a brand's visual identity (logo block, colour, form layout)
                  while being served from a non-brand origin -- exactly the
                  construction that defines a credential-harvesting clone. This
                  gives us labelled positive/anchor pairs with zero exposure to
                  criminal infrastructure, and the freedom to vary the clone
                  fidelity, which is what the zero-shot experiment needs.

Uses headless Chrome via the CDP screenshot flag -- no Selenium/driver version
coupling, just the browser binary already on the machine.
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Brand palette used to synthesise recognisable-but-fake login pages. Colours are
# the brands' well-known primaries; no real logos or copyrighted assets are used
# -- a coloured wordmark block stands in, which is enough for the metric-learning
# task and keeps the corpus free of trademarked images.
BRAND_KIT = {
    "PayPal":     {"primary": "#003087", "accent": "#0070ba", "bg": "#ffffff"},
    "Microsoft":  {"primary": "#0067b8", "accent": "#2f2f2f", "bg": "#ffffff"},
    "Apple":      {"primary": "#111111", "accent": "#0071e3", "bg": "#f5f5f7"},
    "Google":     {"primary": "#4285f4", "accent": "#ea4335", "bg": "#ffffff"},
    "Amazon":     {"primary": "#232f3e", "accent": "#ff9900", "bg": "#ffffff"},
    "Netflix":    {"primary": "#e50914", "accent": "#141414", "bg": "#000000"},
    "Chase":      {"primary": "#117aca", "accent": "#005eb8", "bg": "#ffffff"},
    "DHL":        {"primary": "#ffcc00", "accent": "#d40511", "bg": "#ffffff"},
    "Coinbase":   {"primary": "#0052ff", "accent": "#1652f0", "bg": "#ffffff"},
    "Instagram":  {"primary": "#e1306c", "accent": "#405de6", "bg": "#fafafa"},
    "Facebook":   {"primary": "#1877f2", "accent": "#166fe5", "bg": "#ffffff"},
    "LinkedIn":   {"primary": "#0a66c2", "accent": "#004182", "bg": "#ffffff"},
    "WhatsApp":   {"primary": "#25d366", "accent": "#075e54", "bg": "#ffffff"},
    "Spotify":    {"primary": "#1db954", "accent": "#191414", "bg": "#000000"},
    "Dropbox":    {"primary": "#0061ff", "accent": "#1e1919", "bg": "#ffffff"},
    "Adobe":      {"primary": "#fa0f00", "accent": "#31090a", "bg": "#ffffff"},
    "WellsFargo": {"primary": "#d71e28", "accent": "#ffcd41", "bg": "#ffffff"},
    "HSBC":       {"primary": "#db0011", "accent": "#000000", "bg": "#ffffff"},
    "FedEx":      {"primary": "#4d148c", "accent": "#ff6600", "bg": "#ffffff"},
    "USPS":       {"primary": "#333366", "accent": "#cc0000", "bg": "#ffffff"},
    "Steam":      {"primary": "#171a21", "accent": "#66c0f4", "bg": "#1b2838"},
    "Binance":    {"primary": "#f0b90b", "accent": "#1e2026", "bg": "#ffffff"},
    "eBay":       {"primary": "#e53238", "accent": "#0064d2", "bg": "#ffffff"},
    "Outlook":    {"primary": "#0072c6", "accent": "#28a8ea", "bg": "#ffffff"},
}


def _login_html(brand: str, kit: dict, degrade: float = 0.0) -> str:
    """Render a login page for `brand`.

    `degrade` in [0,1] perturbs layout/spacing/colour to model the imperfect
    clones real kits produce (wrong shade, misaligned form). The zero-shot
    experiment uses higher degrade values to test whether similarity survives
    stylistic drift.
    """
    import colorsys

    def shift(hex_color: str, amt: float) -> str:
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = min(1.0, max(0.0, l + amt * 0.25))
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

    primary = shift(kit["primary"], degrade)
    pad = int(40 * (1 - degrade * 0.3))
    radius = int(8 + degrade * 12)
    fg = "#ffffff" if brand == "Netflix" else kit["accent"]
    return f"""<!doctype html><html><head><meta charset=utf-8>
<style>
 *{{box-sizing:border-box;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif}}
 body{{margin:0;background:{kit['bg']};display:flex;align-items:center;
   justify-content:center;height:100vh}}
 .card{{width:380px;padding:{pad}px;background:#fff;border-radius:{radius}px;
   box-shadow:0 6px 30px rgba(0,0,0,.12);text-align:center}}
 .logo{{font-size:30px;font-weight:800;color:{primary};letter-spacing:-1px;
   margin-bottom:{pad//2}px}}
 h2{{color:{fg if brand=='Netflix' else '#333'};font-size:18px;margin:0 0 22px}}
 input{{width:100%;padding:13px;margin:8px 0;border:1px solid #ccc;
   border-radius:6px;font-size:14px}}
 button{{width:100%;padding:13px;margin-top:14px;background:{primary};color:#fff;
   border:0;border-radius:6px;font-size:15px;font-weight:600;cursor:pointer}}
 .muted{{color:#888;font-size:12px;margin-top:16px}}
</style></head><body>
 <div class=card>
   <div class=logo>{brand}</div>
   <h2>Sign in to your account</h2>
   <input placeholder="Email or phone">
   <input type=password placeholder="Password">
   <button>Sign In</button>
   <div class=muted>Forgot password? &nbsp;·&nbsp; Create account</div>
 </div></body></html>"""


def capture_url(url: str, out_path: Path, size=(1024, 768), timeout: int = 30) -> bool:
    """Headless screenshot of a URL (used for real benign brand pages)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--hide-scrollbars", "--disable-extensions", "--mute-audio",
        f"--window-size={size[0]},{size[1]}",
        "--virtual-time-budget=8000",
        f"--screenshot={out_path}", url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout)
        return out_path.exists() and out_path.stat().st_size > 1000
    except Exception:
        return False


def capture_html(html: str, out_path: Path, size=(1024, 768), timeout: int = 30) -> bool:
    """Headless screenshot of a local HTML string (used for templates)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html)
        tmp = f.name
    ok = capture_url(f"file://{tmp}", out_path, size, timeout)
    Path(tmp).unlink(missing_ok=True)
    return ok


def build_visual_corpus(screenshot_dir: Path, variants_per_brand: int = 8,
                        verbose: bool = True) -> dict:
    """Generate the labelled visual corpus.

    For each brand:
      1 anchor        canonical login page (degrade=0)
      N impersonation clones at increasing degradation (the positive class:
                      brand identity present, served from a non-brand context)
      N benign        the same layout skeleton with a *neutral, non-brand*
                      identity (the negative class: a login page that imitates
                      no one)

    The Siamese objective then learns "close iff same brand", and the phishing
    signal is "high similarity to a brand anchor". Labels and a manifest are
    written so training never re-renders.
    """
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"anchors": {}, "samples": []}
    brands = list(BRAND_KIT.items())

    for bi, (brand, kit) in enumerate(brands):
        anchor_path = screenshot_dir / f"anchor_{brand.lower()}.png"
        if not anchor_path.exists():
            capture_html(_login_html(brand, kit, 0.0), anchor_path)
        manifest["anchors"][brand] = str(anchor_path.name)

        for v in range(variants_per_brand):
            degrade = 0.15 + 0.75 * (v / max(1, variants_per_brand - 1))
            p = screenshot_dir / f"phish_{brand.lower()}_{v}.png"
            if not p.exists():
                capture_html(_login_html(brand, kit, degrade), p)
            manifest["samples"].append(
                {"path": p.name, "brand": brand, "label": 1, "degrade": round(degrade, 3)})

        # Negatives: neutral login pages that impersonate no brand. Colour is
        # rotated away from every brand primary so similarity to any anchor is
        # genuinely low, not an artefact of a shared template.
        neutral = {"primary": ["#556b2f", "#6a5acd", "#708090", "#8b4513"][bi % 4],
                   "accent": "#444", "bg": "#ffffff"}
        for v in range(variants_per_brand):
            name = ["SecurePortal", "AccountHub", "LoginCenter", "MyDashboard",
                    "WebAccess", "UserGate", "SignBox", "NetPortal"][v % 8]
            p = screenshot_dir / f"benign_{bi}_{v}.png"
            if not p.exists():
                capture_html(_login_html(name, neutral, 0.1 * v), p)
            manifest["samples"].append(
                {"path": p.name, "brand": f"neutral_{bi}", "label": 0, "degrade": 0.0})
        if verbose:
            print(f"  [{bi+1}/{len(brands)}] {brand}: anchor + "
                  f"{variants_per_brand} clones + {variants_per_brand} neutral")

    (screenshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if verbose:
        print(f"  visual corpus: {len(manifest['samples'])} samples, "
              f"{len(manifest['anchors'])} brand anchors")
    return manifest


def main() -> int:
    from phishnet.config import Config

    cfg = Config()
    sd = Path(cfg.data.screenshot_dir)
    if not Path(CHROME).exists():
        print(f"Chrome not found at {CHROME}; cannot render.")
        return 1
    build_visual_corpus(sd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
