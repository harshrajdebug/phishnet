"""Render any markdown document to a print-quality PDF.

Generalises build_paper.py: takes an input .md and an output .pdf so the same
academic styling serves the paper, proposals, and reports.

    python scripts/build_doc.py <input.md> <output.pdf>
"""
from __future__ import annotations

import base64
import re
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 20mm 18mm; }
* { box-sizing: border-box; }
body { font-family: "Georgia","Times New Roman",serif; font-size: 10.5pt;
       line-height: 1.5; color: #111; }
h1 { font-size: 20pt; line-height: 1.2; text-align: center; margin: 0 0 6pt; }
h2 { font-size: 13.5pt; margin: 20pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 3pt; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; color: #1a1a1a; }
p { margin: 0 0 8pt; text-align: justify; }
code { font-family: "SF Mono","Consolas",monospace; font-size: 8.6pt;
       background: #f4f4f4; padding: 0.5pt 3pt; border-radius: 3px; }
pre { background: #f7f8fa; border: 1px solid #e2e6ea; padding: 9pt; border-radius: 5px;
      overflow-x: auto; font-size: 7.6pt; line-height: 1.35; }
pre code { background: 0; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #eef1f4; font-family: Helvetica, Arial, sans-serif; }
img { max-width: 100%; display: block; margin: 8pt auto; }
a { color: #0b5; text-decoration: none; word-break: break-all; }
strong { color: #000; }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
li { margin: 2pt 0; }
hr { border: 0; border-top: 1px solid #ccc; margin: 14pt 0; }
h1, h2, h3 { page-break-after: avoid; }
table, pre { page-break-inside: avoid; }
"""


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, out_pdf = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = md.render(src.read_text())

    # inline any local images
    def repl(m):
        srcp = (src.parent / m.group(1)).resolve()
        if srcp.exists():
            b64 = base64.b64encode(srcp.read_bytes()).decode()
            return f'<img src="data:image/png;base64,{b64}"'
        return m.group(0)
    body = re.sub(r'<img src="([^"]+)"', repl, body)

    html_path = out_pdf.with_suffix(".html")
    html_path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>")
    cmd = [CHROME, "--headless", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
           "--virtual-time-budget=10000",
           f"--print-to-pdf={out_pdf}", html_path.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if out_pdf.exists() and out_pdf.stat().st_size > 0:
        print(f"wrote {out_pdf} ({out_pdf.stat().st_size:,} bytes)")
        return 0
    print("FAILED:\n", r.stderr[-600:])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
