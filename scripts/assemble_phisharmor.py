"""Assemble the PhishArmor manuscript from its section drafts, in order."""
from pathlib import Path

ORDER = [
    ("phisharmor_bookends.md",           "abstract"),   # Abstract only
    ("phisharmor_intro.md",              None),
    ("phisharmor_method.md",             None),
    ("phisharmor_results.md",            None),
    ("phisharmor_related_discussion.md", None),
    ("phisharmor_bookends.md",           "conclusion"), # Conclusion + refs tail
]
P = Path("paper")
# single newlines are collapsed by markdown, which ran the byline together
TITLE = """# PhishArmor: Adversarial Robustness for Phishing Detection as an Economic Problem

<div class="byline">
<b>Harsh Raj, Aryan Kumar, Ayush Prajapati, Ujjawal Jain</b><br/>
School of Computer Science, University of Petroleum and Energy Studies, Dehradun, India<br/>
Mentor: Dr. Swati Rastogi
</div>

---
"""


def part(name: str, which: str | None) -> str:
    t = (P / name).read_text()
    if which == "abstract":
        return t.split("# 7. Conclusion")[0].rstrip()
    if which == "conclusion":
        return "# 7. Conclusion" + t.split("# 7. Conclusion")[1].rstrip()
    return t.rstrip()


def main() -> None:
    # references must trail the conclusion, so lift them out of §5/§6
    body, refs = [], ""
    for name, which in ORDER:
        chunk = part(name, which)
        if "\n# References" in chunk:
            chunk, refs = chunk.split("\n# References", 1)
            refs = "# References" + refs
            chunk = chunk.rstrip().rstrip("-").rstrip()
        body.append(chunk)
    out = TITLE + "\n\n" + "\n\n---\n\n".join(body)
    if refs:
        out += "\n\n---\n\n" + refs.rstrip() + "\n"
    dest = P / "phisharmor_manuscript.md"
    dest.write_text(out)
    words = len(out.split())
    print(f"wrote {dest} ({len(out):,} chars, ~{words:,} words)")
    for line in out.splitlines():
        if line.startswith("# "):
            print("   " + line)


if __name__ == "__main__":
    main()
