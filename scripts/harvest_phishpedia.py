"""Harvest a family-disjoint brand corpus from the remote Phishpedia archive.

The archive is 19.3 GB and this machine has ~14 GB free, so nothing is stored
whole: we range-fetch only the members we keep.

Two passes, because the deduplication key lives inside the data:
  A) fetch info.txt (~450 B) for candidate samples -> brand, url, family_id
  B) fetch shot.png + yolo_coords.txt only for the survivors

family_id is a phishing-kit hash. Samples sharing one are near-duplicate pages,
so keeping several would let a model memorise the kit instead of the brand --
the visual analogue of the domain leakage we already control for in the URL arm.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from remote_zip import Entry, RemoteZip  # noqa: E402

OUT = Path("data/phishpedia")
SHOTS = OUT / "shots"

MIN_ARCHIVE_SAMPLES = 20    # brand must have this many samples in the archive
MAX_CANDIDATES = 400        # info.txt fetches per brand (pass A)
MAX_PER_BRAND = 200         # kept samples per brand (pass B)
MIN_PER_BRAND = 12          # brands below this after dedup are dropped


def load_entries() -> list[Entry]:
    with open(OUT / "central_directory.json") as f:
        return [Entry(**e) for e in json.load(f)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pass-a-only", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    SHOTS.mkdir(parents=True, exist_ok=True)

    entries = load_entries()
    by_name = {e.name: e for e in entries}
    dirs = sorted({e.name.split("/")[0] for e in entries if "/" in e.name})

    per_brand: dict[str, list[str]] = defaultdict(list)
    for d in dirs:
        per_brand[d.rsplit("+", 1)[0]].append(d)

    brands = {b: v for b, v in per_brand.items() if len(v) >= MIN_ARCHIVE_SAMPLES}
    print(f"{len(brands)} brands with >={MIN_ARCHIVE_SAMPLES} samples", flush=True)

    candidates: list[str] = []
    for b, v in sorted(brands.items()):
        v = sorted(v)
        rng.shuffle(v)
        candidates += v[:MAX_CANDIDATES]
    print(f"pass A: fetching info.txt for {len(candidates)} candidates", flush=True)

    rz = RemoteZip()
    info: dict[str, dict] = {}
    lock = threading.Lock()

    cache_path = OUT / "info_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        info.update({d: cached[d] for d in candidates if d in cached})
        print(f"reusing {len(info)} cached info records", flush=True)
    candidates = [d for d in candidates if d not in info]

    def fetch_info(d: str):
        e = by_name.get(d + "/info.txt")
        if e is None:
            return
        try:
            raw = rz.read_member(e).decode("utf-8", "replace")
            rec = ast.literal_eval(raw.strip())
        except Exception:
            return
        if not isinstance(rec, dict):
            return
        with lock:
            info[d] = rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_info, d) for d in candidates]
        for i, _ in enumerate(as_completed(futs), 1):
            if i % 500 == 0:
                print(f"  info {i}/{len(candidates)}", flush=True)

    print(f"pass A complete: {len(info)} records", flush=True)
    with open(OUT / "info_cache.json", "w") as f:
        json.dump(info, f)

    # ---- family-disjoint selection -------------------------------------
    fam_of = {d: rec.get("family_id") or f"__nofam__{d}" for d, rec in info.items()}
    by_brand: dict[str, list[str]] = defaultdict(list)
    for d in info:
        by_brand[d.rsplit("+", 1)[0]].append(d)

    selected: list[str] = []
    fam_stats = {}
    for b, ds in sorted(by_brand.items()):
        # true diversity over every candidate, measured before the per-brand cap
        # so it reports kit collapse rather than just re-reporting MAX_PER_BRAND
        n_families = len({fam_of[d] for d in ds})
        seen: set[str] = set()
        keep: list[str] = []
        for d in sorted(ds):
            f = fam_of[d]
            if f in seen:
                continue
            seen.add(f)
            keep.append(d)
            if len(keep) >= MAX_PER_BRAND:
                break
        fam_stats[b] = (len(ds), n_families, len(keep))
        if len(keep) >= MIN_PER_BRAND:
            selected += keep

    kept_brands = sorted({d.rsplit("+", 1)[0] for d in selected})
    print(f"\nafter family dedup: {len(kept_brands)} brands, "
          f"{len(selected)} samples", flush=True)

    collapse = sorted(((n_c, n_f, b) for b, (n_c, n_f, _) in fam_stats.items()),
                      key=lambda t: t[1] / max(t[0], 1))[:12]
    print("\nworst kit-collapse brands (candidates -> unique kit families):", flush=True)
    for n_c, n_f, b in collapse:
        print(f"   {b:<34} {n_c:4d} -> {n_f:3d}  ({100*n_f/max(n_c,1):.0f}%)", flush=True)
    bound = sum(1 for _, n_f, k in fam_stats.values() if n_f < MAX_PER_BRAND)
    print(f"\nbrands where kit diversity (not the cap) binds: {bound}"
          f" of {len(fam_stats)}", flush=True)

    with open(OUT / "selected.json", "w") as f:
        json.dump({"selected": selected, "fam_stats": fam_stats}, f)

    if args.pass_a_only:
        print("\n--pass-a-only set; stopping before screenshot download", flush=True)
        return

    # ---- pass B: screenshots + logo boxes -------------------------------
    print(f"\npass B: fetching shot.png + yolo_coords.txt "
          f"for {len(selected)} samples", flush=True)
    rows: list[dict] = []
    n_done = [0]

    def fetch_sample(d: str):
        se = by_name.get(d + "/shot.png")
        ye = by_name.get(d + "/yolo_coords.txt")
        if se is None:
            return None
        safe = d.replace("/", "_").replace("`", "-")
        dest = SHOTS / f"{safe}.png"
        try:
            if not dest.exists() or dest.stat().st_size == 0:
                dest.write_bytes(rz.read_member(se))
            boxes = []
            if ye is not None:
                for line in rz.read_member(ye).decode("utf-8", "replace").splitlines():
                    if not line.strip():
                        continue
                    part, _, conf = line.partition("\t")
                    try:
                        x1, y1, x2, y2 = ast.literal_eval(part.strip())
                        boxes.append([float(x1), float(y1), float(x2), float(y2),
                                      float(conf or 0.0)])
                    except Exception:
                        continue
        except Exception as exc:
            print(f"  !! {d}: {exc}", flush=True)
            return None

        rec = info[d]
        with lock:
            n_done[0] += 1
            if n_done[0] % 250 == 0:
                print(f"  shot {n_done[0]}/{len(selected)}", flush=True)
        return {
            "dir": d,
            "file": dest.name,
            "brand": d.rsplit("+", 1)[0],
            "family_id": rec.get("family_id", ""),
            "host": rec.get("host", ""),
            "tld": rec.get("tld", ""),
            "sector": rec.get("sector", ""),
            "url": rec.get("url", ""),
            "n_boxes": len(boxes),
            "boxes": json.dumps(boxes),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(fetch_sample, d) for d in selected]):
            r = fut.result()
            if r:
                rows.append(r)

    rows.sort(key=lambda r: (r["brand"], r["dir"]))
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    nb = Counter(r["brand"] for r in rows)
    no_box = sum(1 for r in rows if r["n_boxes"] == 0)
    total = sum(f.stat().st_size for f in SHOTS.glob("*.png"))
    print(f"\nmanifest: {len(rows)} samples, {len(nb)} brands", flush=True)
    print(f"samples with zero logo boxes: {no_box}", flush=True)
    print(f"on disk: {total/1e6:.0f} MB", flush=True)
    print(f"saved -> {OUT/'manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
