"""Command-line interface to a trained PhishNet.

    python -m phishnet.cli scan "http://paypa1-secure.tk/login"
    python -m phishnet.cli scan <url> --text "Verify your account now" --image page.png
    python -m phishnet.cli demo        # runs a battery of illustrative URLs
"""
from __future__ import annotations

import argparse
import json
import sys


def _print(result: dict) -> None:
    v = result["verdict"].upper()
    mark = "⚠" if v == "PHISHING" else "✓"
    print(f"\n{mark}  {v}  ({result['phishing_probability']:.1%})   "
          f"[threshold {result['threshold']:.3f}, {result['latency_ms']} ms]")
    print(f"   {result['url']}")
    used = ", ".join(k for k, val in result["modalities_used"].items() if val)
    print(f"   modalities: {used}")
    if result.get("modality_gates"):
        g = result["modality_gates"]
        print(f"   attention:  url={g['url']:.0%} visual={g['visual']:.0%} text={g['text']:.0%}")
    exp = result.get("explanation")
    if exp and exp.get("summary"):
        print("   why:")
        for s in exp["summary"]:
            print(f"     • {s}")


def main() -> int:
    ap = argparse.ArgumentParser(prog="phishnet")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("url")
    s.add_argument("--text", default=None)
    s.add_argument("--image", default=None)
    s.add_argument("--json", action="store_true")
    sub.add_parser("demo")
    args = ap.parse_args()

    from phishnet.serve.inference import PhishNetEngine
    eng = PhishNetEngine()
    if not eng.loaded_from:
        print("WARNING: no trained weights found; running on random init.", file=sys.stderr)

    if args.cmd == "scan":
        r = eng.predict(args.url, text=args.text, image_path=args.image, explain=True)
        print(json.dumps(r, indent=2)) if args.json else _print(r)
    elif args.cmd == "demo":
        cases = [
            ("https://www.paypal.com/signin", None),
            ("http://paypa1-secure-verify.tk/account/login.php", "Your account is suspended. Verify now."),
            ("https://login.microsoftonline.com/", None),
            ("http://0fficeh-verify.000webhostapp.com/office365/login", "Re-authenticate your mailbox"),
            ("https://en.wikipedia.org/wiki/Phishing", None),
            ("http://192.168.-fake.dhl-tracking-update.xyz/track", "Your parcel is on hold, pay customs"),
        ]
        for url, text in cases:
            _print(eng.predict(url, text=text, explain=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
