"""Generate every figure the paper uses, from the saved score files.

Kept separate from training so figures can be regenerated (restyled, re-binned)
without retraining. All plots read the .npz/.json artefacts in results/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path("results")
FIGDIR = Path("paper/figures")


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.right": False, "axes.spines.top": False,
    })
    FIGDIR.mkdir(parents=True, exist_ok=True)
    return plt


def fig_roc_pr(plt):
    """ROC and PR curves for whichever branches produced score files."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    files = {"URL (1D-CNN)": "url_test_scores.npz",
             "Text (DistilBERT)": "text_test_scores.npz",
             "Fusion (cross-attn)": "fusion_test_scores.npz"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, fn in files.items():
        p = RESULTS / fn
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        y, s = d["labels"], d["scores"]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ax1.plot(fpr, tpr, label=label, lw=2)
        prec, rec, _ = precision_recall_curve(y, s)
        ax2.plot(rec, prec, label=label, lw=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC")
    ax1.set_xlim(0, 0.2)  # deployment-relevant low-FPR region
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
    ax1.legend(); ax2.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_roc_pr.png"); plt.close(fig)
    print("  wrote fig_roc_pr.png")


def fig_fusion_ablation(plt):
    """(a) F1 rises as modalities are added -- the robust multimodal gain.
    (b) cross-attention vs concat across seeds -- indistinguishable (error bars)."""
    abl = _load("ablation_results.json")
    seed = _load("fusion_seed_comparison.json")
    if not abl or not seed:
        return

    # (a) modality ablation F1
    rows = [
        ("URL\nonly", abl.get("url_only")),
        ("URL\n+text", abl.get("url+text")),
        ("URL\n+visual", abl.get("url+visual")),
        ("All\nthree", abl.get("full_fusion")),
    ]
    rows = [(l, r) for l, r in rows if r]
    labels = [r[0] for r in rows]
    x = np.arange(len(labels))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    f1s = [r[1]["f1"] for r in rows]
    colors = ["#8aa0b4"] * (len(rows) - 1) + ["#2ca02c"]
    a1.bar(x, f1s, color=colors)
    for xi, v in zip(x, f1s):
        a1.text(xi, v + 0.0015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=9)
    a1.set(ylabel="F1", title="(a) Adding modalities helps (robust)", ylim=(0.955, 1.002))

    # (b) mechanism comparison with std error bars
    mechs = ["Cross-modal\nattention", "Naive\nconcatenation"]
    means = [seed["cross_attention"]["f1_mean"], seed["concat"]["f1_mean"]]
    stds = [seed["cross_attention"]["f1_std"], seed["concat"]["f1_std"]]
    xb = np.arange(2)
    a2.bar(xb, means, yerr=stds, capsize=8, color=["#1f77b4", "#8aa0b4"],
           error_kw={"elinewidth": 1.5})
    for xi, m, s in zip(xb, means, stds):
        a2.text(xi, m + s + 0.001, f"{m:.4f}\n±{s:.4f}", ha="center", va="bottom", fontsize=8.5)
    a2.set_xticks(xb); a2.set_xticklabels(mechs, fontsize=9)
    n = seed.get("n_runs", 4)
    a2.set(ylabel="F1", title=f"(b) Mechanism: indistinguishable ({n} seeds)",
           ylim=(0.985, 1.001))

    fig.suptitle("Fusion: modalities matter, mechanism does not (identical frozen encoders)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_fusion_ablation.png"); plt.close(fig)
    print("  wrote fig_fusion_ablation.png")


def fig_base_rate(plt):
    """Precision collapse as base rate approaches deployment reality."""
    d = RESULTS / "url_test_scores.npz"
    if not d.exists():
        return
    z = np.load(d, allow_pickle=True)
    y, s, thr = z["labels"], z["scores"], float(z["threshold"])
    from phishnet.eval.metrics import compute
    rates = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    precs, recs = [], []
    rng = np.random.default_rng(0)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    for br in rates:
        n_pos = min(len(pos), int(br * len(neg) / (1 - br)))
        keep = np.concatenate([rng.choice(pos, n_pos, replace=False), neg])
        m = compute(y[keep], s[keep], thr)
        precs.append(m["precision"]); recs.append(m["recall"])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(rates, precs, "o-", label="Precision", lw=2)
    ax.plot(rates, recs, "s-", label="Recall", lw=2)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set(xlabel="Phishing base rate (log)", ylabel="score",
           title="Metric sensitivity to base rate (fixed model & threshold)",
           ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_base_rate.png"); plt.close(fig)
    print("  wrote fig_base_rate.png")


def fig_landscape(plt):
    """2026 threat-landscape context figure (from the cited reports)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    sectors = ["Telecom", "SaaS/\nWebmail", "Social\nMedia", "eCommerce",
               "Financial", "Payment", "Crypto", "Logistics"]
    share = [33, 20, 8, 8, 8, 8, 5, 3]
    ax1.barh(sectors[::-1], share[::-1], color="#1a73e8")
    ax1.set(xlabel="% of attacks", title="Most-targeted sectors, APWG Q1 2026")
    quarters = ["Q2'25", "Q3'25", "Q4'25", "Q1'26"]
    attacks = [1130393, 892494, 853244, 971181]
    ax2.plot(quarters, attacks, "o-", color="#d93025", lw=2)
    ax2.set(ylabel="unique phishing attacks", title="Reported phishing volume (APWG)")
    ax2.ticklabel_format(axis="y", style="plain")
    for q, a in zip(quarters, attacks):
        ax2.annotate(f"{a/1e3:.0f}k", (q, a), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_landscape.png"); plt.close(fig)
    print("  wrote fig_landscape.png")


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> int:
    plt = _setup()
    fig_landscape(plt)
    fig_roc_pr(plt)
    fig_fusion_ablation(plt)
    fig_base_rate(plt)
    print(f"figures -> {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
