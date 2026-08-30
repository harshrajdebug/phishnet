"""Evaluation metrics chosen for the operating regime this system runs in.

Accuracy is close to useless at a 2% base rate -- predicting "benign" always
scores 98%. The metrics that matter to a deployed browser extension are:

  FPR@TPR      how many legitimate pages get flagged when the detector is tuned
               to catch a given fraction of phishing. This is the alert-fatigue
               number and it decides whether users keep the extension enabled.
  TPR@FPR      the mirror: detection achievable under a fixed false-alarm budget.
  PR-AUC       average precision, which unlike ROC-AUC does not flatter a model
               on imbalanced data.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def compute(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    out: dict[str, float] = {
        "n": int(len(y_true)),
        "base_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        fpr, tpr, thr = roc_curve(y_true, y_score)
        for target in (0.90, 0.95, 0.99):
            i = int(np.argmax(tpr >= target))
            out[f"fpr@tpr{int(target*100)}"] = float(fpr[i]) if tpr[i] >= target else float("nan")
        for budget in (0.001, 0.01):
            ok = np.where(fpr <= budget)[0]
            out[f"tpr@fpr{budget}"] = float(tpr[ok].max()) if len(ok) else 0.0
            out[f"thr@fpr{budget}"] = float(thr[ok[int(np.argmax(tpr[ok]))]]) if len(ok) else 1.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    out["fpr"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return out


def threshold_for_fpr(y_true, y_score, budget: float = 0.001) -> float:
    """Pick the operating point on validation data, never on test.

    Tuning the threshold on the test set is a subtle form of leakage that
    reliably buys a point or two of F1 and does not survive deployment.
    """
    fpr, tpr, thr = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_score, float))
    ok = np.where(fpr <= budget)[0]
    if not len(ok):
        return 1.0
    return float(thr[ok[int(np.argmax(tpr[ok]))]])


def format_table(rows: dict[str, dict], keys=None) -> str:
    """Render a comparison table for the console and the paper."""
    keys = keys or ["n", "base_rate", "accuracy", "precision", "recall", "f1",
                    "roc_auc", "pr_auc", "fpr", "fpr@tpr95"]
    width = max(len(k) for k in rows) + 2
    head = "model".ljust(width) + "".join(k.rjust(12) for k in keys)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        cells = []
        for k in keys:
            v = m.get(k, float("nan"))
            cells.append((f"{v:,}" if k == "n" else f"{v:.4f}").rjust(12))
        lines.append(name.ljust(width) + "".join(cells))
    return "\n".join(lines)
