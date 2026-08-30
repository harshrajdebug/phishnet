"""Render paper/paper.md to a self-contained, theme-aware Artifact page.

Editorial-document treatment: a scholarly preprint layout with a proper masthead,
a tinted abstract, mono tabular figures in the results tables, and full light/dark
theming. Figures are embedded as base64 so the page is self-contained. Output is
*content only* (title + font link + style + body markup); the Artifact host wraps
it in the page skeleton at publish time.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "paper.md"
OUT = ROOT / "paper" / "paper_artifact.html"

CSS = r"""
:root{
  --bg:#f7f9fa; --surface:#ffffff; --panel:#eef4f5;
  --text:#16212b; --muted:#556674; --faint:#8394a0;
  --line:#dbe4e9; --line-strong:#c3d0d8;
  --accent:#0b7285; --accent-2:#0a5c6b; --accent-soft:#dff0f3;
  --danger:#b5371f; --danger-soft:#f7e7e2; --good:#1e7d52;
  --serif-display:'Fraunces',Georgia,'Times New Roman',serif;
  --serif-body:'Newsreader',Georgia,serif;
  --mono:'JetBrains Mono',ui-monospace,'SFMono-Regular',Menlo,monospace;
}
:root:not([data-theme="light"]){ }
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0d151d; --surface:#131f29; --panel:#132630;
    --text:#e7eef4; --muted:#9db0bd; --faint:#6c7e8b;
    --line:#243642; --line-strong:#314653;
    --accent:#3fc3dc; --accent-2:#6fd6e8; --accent-soft:#0f2b33;
    --danger:#ff7a68; --danger-soft:#2b1512; --good:#48c98a;
  }
}
:root[data-theme="dark"]{
  --bg:#0d151d; --surface:#131f29; --panel:#132630;
  --text:#e7eef4; --muted:#9db0bd; --faint:#6c7e8b;
  --line:#243642; --line-strong:#314653;
  --accent:#3fc3dc; --accent-2:#6fd6e8; --accent-soft:#0f2b33;
  --danger:#ff7a68; --danger-soft:#2b1512; --good:#48c98a;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--text);
  font-family:var(--serif-body); font-size:18px; line-height:1.62;
  font-optical-sizing:auto;
}
.wrap{max-width:760px; margin:0 auto; padding:56px 26px 120px}
.paper{max-width:100%}

/* ---- masthead ---- */
.eyebrow{
  font-family:var(--mono); font-size:12px; letter-spacing:.22em; text-transform:uppercase;
  color:var(--accent); font-weight:600; margin:0 0 20px; display:flex; gap:12px; align-items:center;
}
.eyebrow::after{content:""; height:1px; flex:1; background:var(--line)}
h1.title{
  font-family:var(--serif-display); font-weight:600; font-optical-sizing:auto;
  font-size:clamp(30px,5.2vw,46px); line-height:1.08; letter-spacing:-.015em;
  text-wrap:balance; margin:0 0 22px;
}
.authors{font-size:17px; font-weight:600; margin:0 0 4px; letter-spacing:.01em}
.affil{color:var(--muted); font-size:15px; margin:0 0 2px}
.mentor{color:var(--muted); font-size:15px; margin:0}
.meta{
  display:flex; flex-wrap:wrap; gap:10px 18px; margin:22px 0 0;
  font-family:var(--mono); font-size:12.5px; color:var(--faint); letter-spacing:.02em;
}
.meta b{color:var(--muted); font-weight:600}
.rule{height:2px; background:var(--text); opacity:.85; margin:34px 0 8px; border:0}

/* ---- body typography ---- */
h2{
  font-family:var(--serif-display); font-weight:600; font-size:26px; letter-spacing:-.01em;
  text-wrap:balance; margin:52px 0 4px; padding-top:16px; border-top:1px solid var(--line);
}
h3{font-family:var(--serif-display); font-weight:600; font-size:20px; margin:32px 0 4px; color:var(--text)}
p{margin:14px 0}
a{color:var(--accent); text-decoration:none; border-bottom:1px solid var(--accent-soft)}
a:hover{border-bottom-color:var(--accent)}
strong{font-weight:700}
em{font-style:italic}
hr{border:0; border-top:1px solid var(--line); margin:36px 0}

/* first section after masthead = Abstract, styled as a panel */
.abstract{
  background:var(--panel); border:1px solid var(--line); border-radius:14px;
  padding:22px 26px; margin:30px 0 8px;
}
.abstract h2{border:0; margin:0 0 6px; padding:0; font-size:15px; letter-spacing:.16em;
  text-transform:uppercase; font-family:var(--mono); color:var(--accent); font-weight:600}
.abstract p{font-size:16.5px; line-height:1.6}
.keywords{font-size:14px; color:var(--muted); margin-top:12px}

/* ---- tables ---- */
.tablewrap{overflow-x:auto; margin:20px 0; border:1px solid var(--line); border-radius:12px; background:var(--surface)}
table{border-collapse:collapse; width:100%; font-family:var(--mono); font-size:13px;
  font-variant-numeric:tabular-nums}
thead th{background:var(--accent-soft); color:var(--accent-2); font-weight:600;
  text-align:right; padding:11px 14px; white-space:nowrap; border-bottom:1px solid var(--line-strong)}
thead th:first-child{text-align:left}
tbody td{padding:10px 14px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap}
tbody td:first-child{text-align:left; font-family:var(--serif-body); font-size:15px}
tbody tr:last-child td{border-bottom:0}
tbody tr:nth-child(even) td{background:color-mix(in srgb, var(--panel) 45%, transparent)}
tbody strong{color:var(--accent-2)}

/* ---- figures ---- */
figure{margin:26px 0; text-align:center}
figure img{max-width:100%; height:auto; border:1px solid var(--line); border-radius:12px; background:var(--surface)}
figcaption{font-size:13.5px; color:var(--muted); font-style:italic; margin-top:10px;
  text-align:left; line-height:1.5}

/* ---- code ---- */
code{font-family:var(--mono); font-size:.86em; background:var(--panel);
  padding:2px 6px; border-radius:5px; color:var(--accent-2); word-break:break-word}
pre{background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:16px; overflow-x:auto} pre code{background:0;padding:0;color:var(--text)}

/* references */
.refs{font-family:var(--mono); font-size:12.5px; line-height:1.7; color:var(--muted)}

/* mobile */
@media (max-width:600px){
  body{font-size:16.5px}
  .wrap{padding:40px 18px 90px}
  h2{font-size:22px}
}
"""


def embed_figures(html: str) -> str:
    def repl(m):
        src, alt = m.group(1), m.group(2)
        p = ROOT / "paper" / src
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            cap = f"<figcaption>{alt}</figcaption>" if alt else ""
            return (f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"/>{cap}</figure>')
        return m.group(0)
    return re.sub(r'<p><img src="([^"]+)" alt="([^"]*)"\s*/?></p>', repl, html)


def wrap_tables(html: str) -> str:
    return re.sub(r'(<table>.*?</table>)', r'<div class="tablewrap">\1</div>', html, flags=re.S)


def main() -> int:
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    text = PAPER.read_text()
    lines = text.splitlines()

    # --- masthead: pull the title block above the first '---' ---
    title = lines[0].lstrip("# ").strip()
    authors = affil = mentor = ""
    for ln in lines[1:8]:
        s = ln.strip().strip("*").strip()
        if not s or s == "---":
            continue
        if not authors and ("," in s or s.replace(" ", "").isalpha()):
            authors = s
        elif "University" in s or "School" in s:
            affil = s
        elif s.lower().startswith("mentor"):
            mentor = s

    # body = everything from '## Abstract' onward
    body_md = text[text.index("## Abstract"):]
    body_html = wrap_tables(embed_figures(md.render(body_md)))

    # mark the abstract block: wrap Abstract heading + following <p> and keywords
    body_html = body_html.replace(
        "<h2>Abstract</h2>", '<div class="abstract"><h2>Abstract</h2>', 1)
    # close the abstract div right before the first numbered section
    body_html = re.sub(r'(<h2>1\. Introduction</h2>)', r'</div>\1', body_html, count=1)
    # style keywords paragraph
    body_html = body_html.replace("<p><strong>Keywords:</strong>",
                                  '<p class="keywords"><strong>Keywords:</strong>')
    # references list styling
    body_html = re.sub(r'(<h2>References</h2>)(.*)$',
                       lambda m: m.group(1) + '<div class="refs">' + m.group(2) + '</div>',
                       body_html, flags=re.S)

    masthead = (
        '<div class="eyebrow">Preprint &middot; 2026</div>'
        f'<h1 class="title">{title}</h1>'
        f'<p class="authors">{authors}</p>'
        f'<p class="affil">{affil}</p>'
        f'<p class="mentor">{mentor}</p>'
        '<div class="meta"><span><b>Field</b> Applied ML / Security</span>'
        '<span><b>Artifact</b> code + models + Chrome extension</span>'
        '<span><b>Status</b> Working system</span></div>'
        '<hr class="rule"/>'
    )

    out = (
        "<title>PhishNet</title>\n"
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&'
        'family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&'
        'family=JetBrains+Mono:wght@400;600&display=swap">\n'
        f"<style>{CSS}</style>\n"
        f'<div class="wrap"><article class="paper">{masthead}{body_html}</article></div>'
    )
    OUT.write_text(out)
    print(f"wrote {OUT} ({len(out):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
