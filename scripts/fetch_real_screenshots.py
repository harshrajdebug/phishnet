"""Fetch the brand-labelled subset of a real webpage-screenshot corpus.

The synthetic corpus encoded brand identity almost entirely in colour, which made
it useless for testing visual robustness (removing a logo cost 0.19 retrieval;
shifting colour cost 0.62 -- the inverse of reality). This pulls real 1920x1080
screenshots carrying genuine logos, typography and layout, restricted to brands
with at least three captured pages so the Siamese retrieval task is well posed.
"""
from __future__ import annotations

import csv
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DS = "shresthsamyak/phishing-website-screenshots"
BASE = f"https://huggingface.co/datasets/{DS}/resolve/main/"
OUT = Path("data/real_shots")
MIN_PER_BRAND = 3


def fetch(row):
    dest = OUT / row["file_name"]
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + urllib.parse.quote(row["file_name"])
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "phishnet-research/1.0"})
        with urllib.request.urlopen(req, timeout=90) as r, open(dest, "wb") as f:
            f.write(r.read())
        return True
    except Exception:
        dest.unlink(missing_ok=True)
        return False


def main() -> int:
    meta = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/meta.csv")
    rows = list(csv.DictReader(meta.open(encoding="utf-8", errors="ignore")))
    ok = [x for x in rows if x["brand_normalized"] and x["is_blank"] == "false"
          and x["is_error_page"] == "false" and x["file_size_bytes"]]
    by = defaultdict(list)
    for x in ok:
        by[x["brand_normalized"]].append(x)
    sel = [x for b, v in by.items() if len(v) >= MIN_PER_BRAND for x in v]
    brands = {x["brand_normalized"] for x in sel}
    print(f"downloading {len(sel):,} images across {len(brands)} brands")

    OUT.mkdir(parents=True, exist_ok=True)
    done = fail = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(fetch, r): r for r in sel}
        for fu in as_completed(futs):
            if fu.result():
                done += 1
            else:
                fail += 1
            if (done + fail) % 100 == 0:
                print(f"  {done+fail}/{len(sel)}  ok={done} fail={fail}", flush=True)
    # write a manifest the crucible can consume
    man = OUT / "manifest.csv"
    with man.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "brand", "page_type", "label_name"])
        for x in sel:
            if (OUT / x["file_name"]).exists():
                w.writerow([x["file_name"], x["brand_normalized"],
                            x["page_type"], x["label_name"]])
    print(f"done: {done} ok, {fail} failed -> {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
