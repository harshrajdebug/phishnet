"""Harvest real benign URLs that have real paths.

Motivation: the standard pairing of a phishing feed with a domain ranking list
(Tranco/Alexa) produces a corpus in which ~99% of phishing samples carry a path
and ~0% of benign samples do. A model trained on that learns "this URL has a
path", reaches near-perfect test accuracy, and collapses in deployment. We
measured exactly this on our own corpus before building this module.

The fix is to give the benign class the same structural shape as the phishing
class: fetch each Tranco homepage once and extract same-origin links. The result
is genuine benign URLs with genuine paths, query strings and depth.

Politeness: one request per domain, hard timeout, no retries, no recursion, and
a User-Agent that identifies the crawler and its purpose.
"""
from __future__ import annotations

import zlib
import random
import re
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
      "PhishNet-academic-crawler/1.0")

HREF_RE = re.compile(rb'href\s*=\s*["\']([^"\']{1,300})["\']', re.I)

# Endpoints that would add path-shaped noise without representing real content.
SKIP_EXT = (".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
            ".woff", ".woff2", ".ttf", ".mp4", ".webm", ".pdf", ".zip", ".xml")

_print_lock = threading.Lock()


def fetch_links(domain_url: str, max_links: int = 40, timeout: int = 8) -> list[str]:
    """Fetch one homepage and return same-origin links that carry a real path."""
    try:
        req = urllib.request.Request(
            domain_url,
            headers={"User-Agent": UA, "Accept": "text/html", "Accept-Encoding": "gzip"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return []
            ctype = r.headers.get("Content-Type", "")
            if "html" not in ctype.lower():
                return []
            body = r.read(400_000)
            if r.headers.get("Content-Encoding") == "gzip":
                # We cap the read, so the gzip stream is usually truncated.
                # decompressobj returns what it has instead of raising, which
                # is exactly what we want -- the <head> and nav links we need
                # are near the start of the document anyway.
                try:
                    body = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(body)
                except Exception:
                    return []
            final = r.geturl()
    except Exception:
        return []

    base_host = (urlparse(final).hostname or "").lower()
    if not base_host:
        return []

    out, seen = [], set()
    for m in HREF_RE.finditer(body):
        href = m.group(1).decode("utf-8", errors="ignore").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            continue
        try:
            absu = urljoin(final, href)
        except Exception:
            continue
        p = urlparse(absu)
        if p.scheme not in ("http", "https"):
            continue
        host = (p.hostname or "").lower()
        # Same registrable origin only: an outbound link could point anywhere,
        # including somewhere we cannot vouch for as benign.
        if host != base_host:
            continue
        path = (p.path or "").strip("/")
        if not path and not p.query:
            continue                                  # homepage again
        if path.lower().endswith(SKIP_EXT):
            continue
        clean = absu.split("#")[0]
        if clean in seen or len(clean) > 400:
            continue
        seen.add(clean)
        out.append(clean)
        if len(out) >= max_links:
            break
    return out


def harvest(domains: list[str], workers: int = 48, max_links: int = 40,
            out_path: Path | None = None, progress_every: int = 250) -> list[str]:
    results: list[str] = []
    done = 0
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_links, d, max_links): d for d in domains}
        for fut in as_completed(futures):
            done += 1
            try:
                links = fut.result()
            except Exception:
                links = []
            if links:
                ok += 1
                results.extend(links)
            if done % progress_every == 0:
                with _print_lock:
                    print(f"  {done:,}/{len(domains):,} domains  "
                          f"{ok:,} responded  {len(results):,} URLs", flush=True)
    # De-duplicate while preserving order, then shuffle so that writing order
    # does not correlate with Tranco rank.
    seen, uniq = set(), []
    for u in results:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    random.Random(1337).shuffle(uniq)
    if out_path:
        out_path.write_text("\n".join(uniq))
        print(f"  wrote {len(uniq):,} benign deep URLs -> {out_path}")
    return uniq


def main() -> int:
    from phishnet.config import Config

    cfg = Config()
    raw = Path(cfg.data.raw_dir)
    n_domains = int(sys.argv[1]) if len(sys.argv) > 1 else 4000

    all_domains = raw.joinpath("benign_urls.txt").read_text().splitlines()
    # Sample across the whole rank range, not just the head: the top 500 sites
    # are atypically large and well-engineered, and a model tuned on them would
    # not generalise to the long tail where most real browsing happens.
    rng = random.Random(1337)
    pool = all_domains[:100_000]
    domains = rng.sample(pool, min(n_domains, len(pool)))

    print(f"harvesting same-origin deep links from {len(domains):,} Tranco domains")
    harvest(domains, out_path=raw / "benign_deep_urls.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
