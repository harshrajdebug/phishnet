"""Shared training loop utilities."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def make_scheduler(optimizer, total_steps: int, warmup_frac: float = 0.1):
    """Linear warmup then cosine decay.

    Warmup matters here because the fusion module contains freshly initialised
    attention layers sitting on top of pretrained encoders; without it the first
    few large gradients wash out the pretrained features.
    """
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    def __init__(self, patience: int = 3, mode: str = "max"):
        self.patience, self.mode = patience, mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.bad = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Return True when this is a new best."""
        better = value > self.best if self.mode == "max" else value < self.best
        if better:
            self.best, self.bad = value, 0
            return True
        self.bad += 1
        if self.bad >= self.patience:
            self.should_stop = True
        return False


def pos_weight_for(labels: np.ndarray, device) -> torch.Tensor:
    """Re-weight the positive class to counter imbalance in the training split."""
    pos = float((labels == 1).sum())
    neg = float((labels == 0).sum())
    w = (neg / pos) if pos > 0 else 1.0
    return torch.tensor([w], dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model, loader, device, forward) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    for batch in loader:
        logits, y = forward(model, batch, device)
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(y.float().cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels)


def train_epochs(model, train_loader, val_loader, device, cfg, forward,
                 params=None, lr=None, tag="model", out_dir=None,
                 pos_weight=None, select_metric="pr_auc"):
    """Generic supervised loop shared by every branch.

    `forward(model, batch, device) -> (logits, labels)` is the only thing that
    differs between branches, so the training policy (schedule, clipping, early
    stopping, checkpoint selection) is identical across the ablation. That is
    deliberate: differences in the results table should come from architecture,
    not from one model getting a better-tuned loop.
    """
    from phishnet.eval.metrics import compute

    out_dir = Path(out_dir or "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    params = params if params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr or cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    total = max(1, len(train_loader) * cfg.train.epochs)
    sched = make_scheduler(opt, total, cfg.train.warmup_frac)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    stopper = EarlyStopping(cfg.train.early_stop_patience, mode="max")

    best_path = out_dir / f"{tag}_best.pt"
    history = []
    model.to(device)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for i, batch in enumerate(train_loader):
            logits, y = forward(model, batch, device)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step()
            sched.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
            if i % 100 == 0:
                print(f"    ep{epoch} step {i}/{len(train_loader)} "
                      f"loss={running/max(seen,1):.4f}", flush=True)

        scores, labels = predict(model, val_loader, device, forward)
        m = compute(labels, scores)
        m.update(epoch=epoch, train_loss=running / max(seen, 1),
                 secs=round(time.time() - t0, 1))
        history.append(m)
        print(f"  epoch {epoch}: loss={m['train_loss']:.4f} "
              f"val_pr_auc={m.get('pr_auc', float('nan')):.4f} "
              f"val_roc_auc={m.get('roc_auc', float('nan')):.4f} "
              f"f1={m['f1']:.4f} ({m['secs']}s)", flush=True)

        if stopper.step(m.get(select_metric, 0.0)):
            torch.save({"state_dict": model.state_dict(),
                        "encoder_state": getattr(model, "encoder", model).state_dict(),
                        "epoch": epoch, "val": m}, best_path)
            print(f"    new best ({select_metric}={stopper.best:.4f}) -> {best_path.name}")
        if stopper.should_stop:
            print(f"  early stop at epoch {epoch}")
            break

    (out_dir / f"{tag}_history.json").write_text(json.dumps(history, indent=2))
    if best_path.exists():
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print(f"  restored best epoch {ck['epoch']}")
    return model, history
