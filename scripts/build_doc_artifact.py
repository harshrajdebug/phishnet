"""Render a markdown document to a self-contained, theme-aware Artifact page.

Shares the visual identity of the PhishNet paper artifact (Fraunces / Newsreader /
JetBrains Mono, detection-teal accent) so the project's documents read as a family.
The ASCII architecture block is swapped for a native mermaid diagram, which the
Artifact runtime renders without any external library.

    python scripts/build_doc_artifact.py <input.md> <output.html>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt

MERMAID = """flowchart TB
  D["DATA LAYER<br/>Phishpedia 30k · PhishIntention 50k<br/>OpenPhish live · Tranco benign"]
  C["COST MODEL<br/>taxonomy · estimator · conversion"]
  A["ATTACK SUITE<br/>visual · perturbation<br/>html · url · compose"]
  Z["DETECTOR ZOO<br/>Phishpedia · PhishIntention<br/>PhishAgent · PhishArmor"]
  T["DEFENCE TRAINER<br/>cost-weighted AT<br/>reliance penalty · anchored fusion"]
  B["BENCH RUNNER<br/>runner · metrics (CTE) · report"]
  R["RESULTS<br/>CTE curves · robustness frontier · CIs"]
  D --> C
  D --> A
  D --> Z
  C -->|prices| A
  A -->|attacks| Z
  C -->|"P(a) ∝ exp(−cost/τ)"| T
  A --> T
  T -->|trains| B
  Z --> B
  B --> R
"""

CSS = r"""
:root{
  --bg:#f7f9fa; --surface:#ffffff; --panel:#eef4f5;
  --text:#16212b; --muted:#556674; --faint:#8394a0;
  --line:#dbe4e9; --line-strong:#c3d0d8;
  --accent:#0b7285; --accent-2:#0a5c6b; --accent-soft:#dff0f3;
  --serif-display:'Fraunces',Georgia,serif;
  --serif-body:'Newsreader',Georgia,serif;
  --mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0d151d; --surface:#131f29; --panel:#132630;
    --text:#e7eef4; --muted:#9db0bd; --faint:#6c7e8b;
    --line:#243642; --line-strong:#314653;
    --accent:#3fc3dc; --accent-2:#6fd6e8; --accent-soft:#0f2b33;
  }
}
:root[data-theme="dark"]{
  --bg:#0d151d; --surface:#131f29; --panel:#132630;
  --text:#e7eef4; --muted:#9db0bd; --faint:#6c7e8b;
  --line:#243642; --line-strong:#314653;
  --accent:#3fc3dc; --accent-2:#6fd6e8; --accent-soft:#0f2b33;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font-family:var(--serif-body);font-size:17.5px;line-height:1.62}
.wrap{max-width:780px;margin:0 auto;padding:52px 24px 110px}
.eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.22em;text-transform:uppercase;
  color:var(--accent);font-weight:600;display:flex;align-items:center;gap:12px;margin:0 0 18px}
.eyebrow::after{content:"";height:1px;flex:1;background:var(--line)}
h1{font-family:var(--serif-display);font-weight:600;font-size:clamp(28px,5vw,42px);
  line-height:1.08;letter-spacing:-.015em;text-wrap:balance;margin:0 0 18px}
h2{font-family:var(--serif-display);font-weight:600;font-size:24px;letter-spacing:-.01em;
  text-wrap:balance;margin:48px 0 6px;padding-top:16px;border-top:1px solid var(--line)}
h3{font-family:var(--serif-display);font-weight:600;font-size:19px;margin:30px 0 4px}
p{margin:13px 0}
strong{font-weight:700}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid var(--accent-soft)}
a:hover{border-bottom-color:var(--accent)}
hr{border:0;border-top:2px solid var(--text);opacity:.8;margin:26px 0}
.byline{color:var(--muted);font-size:15.5px;line-height:1.5;margin:0 0 4px}
.tablewrap{overflow-x:auto;margin:18px 0;border:1px solid var(--line);border-radius:12px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12.5px}
thead th{background:var(--accent-soft);color:var(--accent-2);font-weight:600;text-align:left;
  padding:11px 14px;border-bottom:1px solid var(--line-strong);white-space:nowrap}
tbody td{padding:10px 14px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:last-child td{border-bottom:0}
tbody tr:nth-child(even) td{background:color-mix(in srgb,var(--panel) 45%,transparent)}
code{font-family:var(--mono);font-size:.85em;background:var(--panel);padding:2px 6px;
  border-radius:5px;color:var(--accent-2);word-break:break-word}
pre{background:var(--surface);border:1px solid var(--line);border-radius:11px;
  padding:16px;overflow-x:auto;font-size:12.5px;line-height:1.45}
pre code{background:0;padding:0;color:var(--text)}
ul,ol{padding-left:22px}
li{margin:5px 0}
.mermaid{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:18px;margin:20px 0;overflow-x:auto;text-align:center}
"""


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    text = src.read_text()

    lines = text.splitlines()
    title = lines[0].lstrip("# ").strip()
    # byline block = lines up to the first '---'
    end = next(i for i, l in enumerate(lines) if l.strip() == "---")
    byline = [l.strip().strip("*") for l in lines[1:end] if l.strip()]
    body_md = text[text.index("## Executive Summary"):]

    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = md.render(body_md)
    body = re.sub(r'(<table>.*?</table>)', r'<div class="tablewrap">\1</div>', body, flags=re.S)

    # swap the ASCII system-overview block for a native mermaid diagram
    def swap(m):
        if "DATA LAYER" in m.group(1):
            return '<pre class="mermaid">' + MERMAID + "</pre>"
        return m.group(0)
    body = re.sub(r'<pre><code>(.*?)</code></pre>', swap, body, flags=re.S)

    head = (
        f"<title>PhishArmor Proposal</title>\n"
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Fraunces:opsz,wght@9..144,500;9..144,600&'
        'family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&'
        'family=JetBrains+Mono:wght@400;600&display=swap">\n'
        f"<style>{CSS}</style>\n")
    mast = ('<div class="eyebrow">Research Proposal</div>'
            f'<h1>{title}</h1>'
            + "".join(f'<p class="byline">{b}</p>' for b in byline)
            + '<hr/>')
    out.write_text(f'{head}<div class="wrap"><article>{mast}{body}</article></div>')
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
