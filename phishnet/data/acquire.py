"""Acquisition of the real-world corpora PhishNet trains on.

Four public sources, all fetched live so a run can be re-dated:

  phishing URLs  OpenPhish community feed (current, ~300/day) plus the
                 Phishing.Database ACTIVE list (historical breadth)
  benign URLs    Tranco top-1M research list (rank-stable, resists the
                 manipulation that made Alexa unusable for this task)
  phishing text  Nazario phishing corpus (mbox)
  benign text    SpamAssassin public ham corpus

Nothing here touches a live phishing page. URLs are treated as strings; the
only pages ever rendered are the benign Tranco sites and locally generated
templates (see `render.py`). That keeps the pipeline safe to run unattended
and avoids sending traffic to criminal infrastructure.
"""
from __future__ import annotations

import bz2
import gzip
import io
import mailbox
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) PhishNet-research/1.0"
TIMEOUT = 120

SOURCES = {
    "openphish": "https://openphish.com/feed.txt",
    "phishdb": "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt",
    "tranco": "https://tranco-list.eu/top-1m.csv.zip",
    "nazario": "https://monkey.org/~jose/phishing/phishing3.mbox",
    "spamassassin_ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "spamassassin_ham2": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
}


def _get(url: str, dest: Path, force: bool = False) -> Path:
    """Download `url` to `dest`, skipping if we already have a non-empty copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  cached  {dest.name} ({dest.stat().st_size:,} B)")
        return dest
    print(f"  fetch   {url}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"  saved   {dest.name} ({dest.stat().st_size:,} B)")
    return dest


def fetch_phishing_urls(raw_dir: Path, force: bool = False) -> list[str]:
    """Union of the live feed and the historical ACTIVE list, de-duplicated."""
    urls: list[str] = []

    op = _get(SOURCES["openphish"], raw_dir / "openphish_feed.txt", force)
    live = [l.strip() for l in op.read_text(errors="ignore").splitlines() if l.strip()]
    urls += live
    print(f"  openphish: {len(live):,} live URLs")

    db = _get(SOURCES["phishdb"], raw_dir / "phishing_links_active.txt", force)
    hist = [l.strip() for l in db.read_text(errors="ignore").splitlines()
            if l.strip() and not l.startswith("#")]
    urls += hist
    print(f"  phishdb:   {len(hist):,} historical URLs")

    seen, out = set(), []
    for u in urls:
        if u.startswith(("http://", "https://")) and u not in seen:
            seen.add(u)
            out.append(u)
    print(f"  phishing total (deduped, http/s only): {len(out):,}")
    return out


def fetch_benign_urls(raw_dir: Path, limit: int = 200_000, force: bool = False) -> list[str]:
    """Top-N Tranco domains, materialised as https:// URLs.

    Tranco ranks are an average over 30 days, so a domain that briefly spikes
    cannot buy its way onto the list -- which matters because a poisoned benign
    set silently teaches the model that phishing infrastructure is legitimate.
    """
    z = _get(SOURCES["tranco"], raw_dir / "tranco_top1m.csv.zip", force)
    out: list[str] = []
    with zipfile.ZipFile(z) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8"):
                parts = line.strip().split(",")
                if len(parts) == 2:
                    out.append(f"https://{parts[1]}")
                if len(out) >= limit:
                    break
    print(f"  tranco:    {len(out):,} benign URLs")
    return out


_HDRS = ("subject", "from", "to", "date", "return-path", "received",
         "message-id", "mime-version", "content-type")


def _msg_to_text(msg) -> str:
    """Flatten a message to 'Subject + body', which is what a user actually reads."""
    subject = str(msg.get("Subject", "") or "")
    body_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype in ("text/plain", "text/html"):
                try:
                    payload = part.get_payload(decode=True)
                except Exception:
                    continue
                if payload:
                    body_parts.append(payload.decode("utf-8", errors="ignore"))
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                body_parts.append(payload.decode("utf-8", errors="ignore"))
        except Exception:
            pass
    body = "\n".join(body_parts)
    body = re.sub(r"<script.*?</script>", " ", body, flags=re.S | re.I)
    body = re.sub(r"<style.*?</style>", " ", body, flags=re.S | re.I)
    body = re.sub(r"<[^>]+>", " ", body)          # strip markup, keep the words
    body = re.sub(r"&[a-z]{2,8};", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    return f"{subject}\n{body}".strip()


def fetch_phishing_emails(raw_dir: Path, force: bool = False) -> list[str]:
    p = _get(SOURCES["nazario"], raw_dir / "nazario_phishing3.mbox", force)
    mbox = mailbox.mbox(str(p))
    out = []
    for msg in mbox:
        try:
            t = _msg_to_text(msg)
        except Exception:
            continue
        if len(t) > 40:
            out.append(t)
    print(f"  nazario:   {len(out):,} phishing emails")
    return out


def fetch_benign_emails(raw_dir: Path, force: bool = False) -> list[str]:
    out: list[str] = []
    for key in ("spamassassin_ham", "spamassassin_ham2"):
        p = _get(SOURCES[key], raw_dir / f"{key}.tar.bz2", force)
        with tarfile.open(p, "r:bz2") as tf:
            for m in tf.getmembers():
                if not m.isfile() or m.name.endswith("cmds"):
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                try:
                    msg = BytesParser(policy=policy.default).parsebytes(fh.read())
                    t = _msg_to_text(msg)
                except Exception:
                    continue
                if len(t) > 40:
                    out.append(t)
    print(f"  spamassassin ham: {len(out):,} benign emails")
    return out


def main() -> int:
    from phishnet.config import Config, ensure_dirs

    cfg = Config()
    ensure_dirs(cfg)
    raw = Path(cfg.data.raw_dir)
    force = "--force" in sys.argv

    print("[1/4] phishing URLs")
    ph = fetch_phishing_urls(raw, force)
    print("[2/4] benign URLs")
    bn = fetch_benign_urls(raw, force=force)
    print("[3/4] phishing email text")
    pe = fetch_phishing_emails(raw, force)
    print("[4/4] benign email text")
    be = fetch_benign_emails(raw, force)

    (raw / "phish_urls.txt").write_text("\n".join(ph))
    (raw / "benign_urls.txt").write_text("\n".join(bn))
    (raw / "phish_emails.txt").write_text("\n<<<DOC>>>\n".join(pe))
    (raw / "benign_emails.txt").write_text("\n<<<DOC>>>\n".join(be))

    print("\nacquired:")
    print(f"  phishing URLs   {len(ph):,}")
    print(f"  benign URLs     {len(bn):,}")
    print(f"  phishing emails {len(pe):,}")
    print(f"  benign emails   {len(be):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
