"""Render paper/paper.md to a print-quality PDF.

Pipeline: markdown-it (with tables + figure images) -> a single self-contained
HTML file with an academic stylesheet -> Chrome headless --print-to-pdf. This
avoids WeasyPrint's native pango/cairo dependency while still producing a
two-column-free, journal-style single-column PDF with real page numbers.
"""
from __future__ import annotations

import argparse
import base64
import html as _html
import re
import subprocess
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "paper.md"
FIGDIR = ROOT / "paper" / "figures"
OUT_HTML = ROOT / "paper" / "paper.html"
OUT_PDF = ROOT / "paper" / "PhishNet_paper.pdf"
def _find_chrome() -> str:
    """Locate a headless-capable Chrome/Chromium across platforms.

    Artifact reviewers typically run Linux, where the macOS bundle path does not
    exist; hardcoding it made the documented build command fail for them.
    Override with CHROME_BIN if the binary lives somewhere unusual.
    """
    import os
    import shutil

    if os.environ.get("CHROME_BIN"):
        return os.environ["CHROME_BIN"]
    for name in ("google-chrome", "google-chrome-stable", "chromium",
                 "chromium-browser", "chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    for path in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/usr/bin/google-chrome", "/usr/bin/chromium",
        "/usr/bin/chromium-browser", "/snap/bin/chromium",
    ):
        if Path(path).exists():
            return path
    raise SystemExit(
        "No Chrome/Chromium found. Install one, or set CHROME_BIN to its path.\n"
        "The HTML is still written, so the PDF step is the only thing blocked."
    )


CHROME = None   # resolved lazily in main(), so importing this module never fails

CSS = """
@page { size: A4; margin: 22mm 20mm; @bottom-center { content: counter(page); } }
* { box-sizing: border-box; }
body { font-family: "Georgia","Times New Roman",serif; font-size: 10.5pt;
       line-height: 1.5; color: #111; max-width: 100%; }
h1 { font-size: 19pt; line-height: 1.25; text-align: center; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 13pt 0 4pt; color: #222; }
h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
table { break-before: avoid; page-break-before: avoid; }
p { orphans: 2; widows: 2; }
p { margin: 0 0 8pt; text-align: justify; }
code { font-family: "SF Mono","Consolas",monospace; font-size: 9pt;
       background: #f4f4f4; padding: 0.5pt 3pt; border-radius: 3px;
       word-break: normal; overflow-wrap: break-word; }
pre { background: #f6f8fa; padding: 8pt; border-radius: 5px; overflow-x: auto; font-size: 8.5pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 3pt 6pt; text-align: center; }
th { background: #eef1f4; font-family: Helvetica, Arial, sans-serif; }
td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; display: block; margin: 8pt auto; }
.byline { text-align: center; font-size: 10pt; line-height: 1.45; margin: 2pt 0 10pt; }
.math { font-family: "Latin Modern Math","Cambria Math","Georgia",serif; font-style: italic; }
.math .up, .math b { font-style: normal; }
.math.block { display: block; text-align: center; margin: 9pt 0 11pt; font-size: 11pt; }
.frac { display: inline-block; vertical-align: middle; text-align: center; margin: 0 2pt; }
.frac .num { display: block; border-bottom: 1px solid #111; padding: 0 3pt; }
.frac .den { display: block; padding: 0 3pt; }
sub, sup { font-size: 72%; font-style: normal; }
figure { margin: 10pt 0; page-break-inside: avoid; }
figcaption { font-size: 8.5pt; color: #444; text-align: center; font-style: italic; }
a { color: #0b5; text-decoration: none; word-break: break-all; }
strong { color: #000; }
blockquote { border-left: 3px solid #ccc; margin: 8pt 0; padding: 2pt 12pt; color: #333; }
h2, h3 { page-break-after: avoid; }
table, figure, pre { page-break-inside: avoid; }
.byline { text-align:center; font-size:10pt; color:#333; margin: 2pt 0 14pt; }
"""


def embed_figures(html: str) -> str:
    """Inline referenced figure PNGs as base64 so the HTML is self-contained."""
    def repl(m):
        alt, src = m.group(1), m.group(2)
        p = (ROOT / "paper" / src) if not src.startswith("/") else Path(src)
        if not p.exists():
            p = FIGDIR / Path(src).name
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            cap = f"<figcaption>{alt}</figcaption>" if alt else ""
            return (f'<figure><img src="data:image/png;base64,{b64}" alt="{alt}"/>'
                    f'{cap}</figure>')
        return m.group(0)
    return re.sub(r'<img alt="([^"]*)" src="([^"]+)"\s*/?>', repl, html)



SYMBOLS = {
    r"\\omega": "\u03c9", r"\\tau": "\u03c4", r"\\lambda": "\u03bb",
    r"\\beta": "\u03b2", r"\\eta": "\u03b7", r"\\ell": "\u2113",
    r"\\times": "\u00d7", r"\\le": "\u2264", r"\\ge": "\u2265",
    r"\\to": "\u2192", r"\\pm": "\u00b1", r"\\in": "\u2208",
    r"\\sum": "\u2211", r"\\partial": "\u2202", r"\\top": "\u22a4",
    r"\\propto": "\u221d", r"\\approx": "\u2248", r"\\cdot": "\u00b7",
    r"\\qquad": "&emsp;&emsp;", r"\\,": "&thinsp;",
    r"\\;": "&thinsp;", r"\\!": "", r"\\%": "%", r"\\&": "&amp;",
}
UPRIGHT = ("min", "max", "exp", "log")


def _tex(src: str) -> str:
    """Render the small LaTeX subset this manuscript uses to inline HTML.

    Deliberately not a general TeX engine: the vocabulary here is 26 commands,
    and a targeted converter keeps the build self-contained (no KaTeX download,
    no network at build time).

    Order matters. \\text{..} and _{..} are expanded *before* \\frac so that by
    the time the fraction rule runs, its arguments contain no nested braces --
    a single-level brace regex cannot match \\frac{p_{\\text{chance}}}{..}.
    """
    t = src
    # 1. brace-taking font commands first (removes their braces)
    t = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r'<span class="up">\1</span>', t)
    t = re.sub(r"\\mathbf\{([^{}]*)\}", r"<b>\1</b>", t)
    # 2. braced sub/superscripts next (removes their braces)
    for _ in range(2):
        t = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", t)
        t = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", t)
    # 3. now \frac sees brace-free arguments
    for _ in range(3):
        t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}",
                   r'<span class="frac"><span class="num">\1</span>'
                   r'<span class="den">\2</span></span>', t)
    # 4. upright operator names, then symbols
    for name in UPRIGHT:
        t = re.sub(r"\\" + name + r"(?![a-zA-Z])", f'<span class="up">{name}</span>', t)
    for k, v in SYMBOLS.items():
        t = re.sub(k + r"(?![a-zA-Z])", v, t)
    # 5. single-token sub/sup last, accepting the unicode we just substituted
    t = re.sub(r"_([^\s{}<])", r"<sub>\1</sub>", t)
    t = re.sub(r"\^([^\s{}<])", r"<sup>\1</sup>", t)
    t = t.replace("\\\\", "<br/>")
    # grouping braces carry no visual meaning once content is rendered
    t = t.replace("{", "").replace("}", "")
    return t


def protect_math(text: str):
    """Pull math out before markdown runs, so `_` is not read as emphasis."""
    store: list[str] = []

    def stash(rendered: str, block: bool) -> str:
        store.append(f'<span class="math{" block" if block else ""}">{rendered}</span>')
        return f"MATHTOKEN{len(store)-1}ZZ"

    text = re.sub(r"\$\$(.+?)\$\$",
                  lambda m: stash(_tex(m.group(1).strip()), True), text, flags=re.S)
    # inline math may wrap across a line break in the source, but must not run
    # past a blank line; forbidding \n outright leaves the opening $ unmatched
    # and it then pairs with a later $, garbling both spans.
    text = re.sub(r"(?<![\\$])\$(?!\$)((?:[^$\n]|\n(?!\s*\n))+?)\$",
                  lambda m: stash(_tex(m.group(1)), False), text)
    return text, store


def restore_math(html_text: str, store: list[str]) -> str:
    for i, frag in enumerate(store):
        html_text = html_text.replace(f"MATHTOKEN{i}ZZ", frag)
    return html_text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(PAPER))
    ap.add_argument("--out-pdf", default=str(OUT_PDF))
    ap.add_argument("--out-html", default=str(OUT_HTML))
    args = ap.parse_args()
    src = Path(args.source).resolve()
    out_pdf = Path(args.out_pdf).resolve()      # as_uri() needs absolute paths
    out_html = Path(args.out_html).resolve()

    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    text = src.read_text()
    text, mstore = protect_math(text)
    body = md.render(text)
    body = restore_math(body, mstore)
    body = embed_figures(body)
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    out_html.write_text(html)
    print(f"wrote {out_html} ({len(html):,} bytes)")

    # Chrome headless print-to-pdf. --no-pdf-header-footer keeps our @page rules.
    cmd = [_find_chrome(), "--headless", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
           "--virtual-time-budget=10000",
           f"--print-to-pdf={out_pdf}", out_html.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out_pdf.exists() and out_pdf.stat().st_size > 0:
        print(f"wrote {out_pdf} ({out_pdf.stat().st_size:,} bytes)")
        return 0
    print("PDF generation failed:\n", r.stderr[-800:])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
