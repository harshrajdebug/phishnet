"""Consolidated benchmark: assemble every result table the paper reports.

Reads the per-branch result JSONs produced by the training scripts and prints
(1) the unimodal comparison against classical baselines, (2) the fusion ablation
(cross-attention vs concat vs each unimodal branch), and (3) the base-rate
sensitivity that shows why accuracy is the wrong headline metric.
"""
from __future__ import annotations

import json
from pathlib import Path

from phishnet.eval.metrics import format_table


def load(name: str) -> dict:
    p = Path("results") / name
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> int:
    results = Path("results")
    url = load("url_results.json")
    text = load("text_results.json")
    visual = load("visual_results.json")
    fusion = load("fusion_results.json")

    print("=" * 78)
    print("TABLE 1  Unimodal URL detection (domain-disjoint split)")
    print("=" * 78)
    rows = {k: v for k, v in url.items() if not k.startswith("_") and "@" not in k}
    if rows:
        print(format_table(rows))

    print("\n" + "=" * 78)
    print("TABLE 2  Text branch vs bag-of-words baseline")
    print("=" * 78)
    if text:
        print(format_table(text))

    print("\n" + "=" * 78)
    print("TABLE 3  Visual branch: seen brands vs zero-shot brands")
    print("=" * 78)
    if visual:
        print(format_table(visual, keys=["n", "accuracy", "precision", "recall",
              "f1", "roc_auc", "pr_auc"]))

    print("\n" + "=" * 78)
    print("TABLE 4  Fusion ablation (same frozen encoders, same data)")
    print("=" * 78)
    combined = {}
    if url:
        for k in ("url_cnn",):
            if k in url:
                combined[f"unimodal:{k}"] = url[k]
    if text.get("distilbert"):
        combined["unimodal:text"] = text["distilbert"]
    for k, v in fusion.items():
        combined[f"fusion:{k}"] = v
    if combined:
        print(format_table(combined))

    print("\n" + "=" * 78)
    print("TABLE 5  Base-rate sensitivity of the URL branch")
    print("=" * 78)
    br = {k: v for k, v in url.items() if k == "url_cnn" or "@base_rate" in k}
    if br:
        print(format_table(br, keys=["n", "base_rate", "accuracy", "precision",
              "recall", "f1", "pr_auc", "fpr"]))
        print("\nNote: identical model & threshold. Precision falls as the base "
              "rate\napproaches deployment reality -- the reason we tune on "
              "FPR, not accuracy.")

    summary = {"url": url, "text": text, "visual": visual, "fusion": fusion}
    (results / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfull summary -> {results/'benchmark_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
