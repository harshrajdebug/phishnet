"""Train the fused PhishNet and run the fusion ablation.

The scientific question is narrow and the experiment is built to answer only it:
does cross-modal attention beat naive concatenation when both sit on top of the
*same* trained encoders and see the *same* data? So encoders are warm-started
from their unimodal checkpoints and then frozen, and only the fusion head is
trained. Any difference in the results table is therefore attributable to the
fusion mechanism, not to one model getting more representation-learning.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phishnet.config import Config, ensure_dirs, resolve_device, set_seed
from phishnet.data.multimodal import assemble
from phishnet.eval.metrics import compute, format_table, threshold_for_fpr
from phishnet.features.url_encoding import encode_url
from phishnet.models.phishnet import PhishNet
from phishnet.train.common import EarlyStopping, make_scheduler


class MMDataset(Dataset):
    def __init__(self, samples, tokenizer, screenshot_dir: Path, cfg):
        self.samples = samples
        self.tok = tokenizer
        self.sd = screenshot_dir
        self.cfg = cfg
        self._img_cache: dict[str, torch.Tensor] = {}

    def __len__(self):
        return len(self.samples)

    def _img(self, name):
        from phishnet.train.train_visual import load_image
        if name not in self._img_cache:
            self._img_cache[name] = load_image(self.sd / name, self.cfg.visual.image_size)
        return self._img_cache[name]

    def __getitem__(self, i):
        s = self.samples[i]
        url_ids = torch.from_numpy(encode_url(s.url, self.cfg.url.max_len))
        if s.text:
            enc = self.tok(s.text, truncation=True, padding="max_length",
                           max_length=self.cfg.text.max_len, return_tensors="pt")
            ids, mask = enc["input_ids"][0], enc["attention_mask"][0]
        else:
            ids = torch.zeros(self.cfg.text.max_len, dtype=torch.long)
            mask = torch.zeros(self.cfg.text.max_len, dtype=torch.long)
        img = self._img(s.image) if s.image else torch.zeros(3, self.cfg.visual.image_size,
                                                             self.cfg.visual.image_size)
        present = torch.tensor(s.present, dtype=torch.bool)
        return url_ids, ids, mask, img, present, torch.tensor(float(s.label))


def collate(batch):
    url_ids, ids, mask, img, present, y = zip(*batch)
    return {
        "url_ids": torch.stack(url_ids),
        "input_ids": torch.stack(ids),
        "attention_mask": torch.stack(mask),
        "image": torch.stack(img),
        "present": torch.stack(present),
        "label": torch.stack(y),
    }


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    for batch in loader:
        b = to_device(batch, device)
        logit = model(b)
        scores.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(b["label"].cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels)


def train_one(model, tl, vl, device, cfg, tag, epochs=6):
    """Train fusion head only (encoders frozen)."""
    model.to(device)
    model.freeze_encoders(True)
    params = [p for p in model.fusion.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    total = max(1, len(tl) * epochs)
    sched = make_scheduler(opt, total, cfg.train.warmup_frac)

    # Positive weight from the actual training prevalence.
    ys = np.array([s.label for s in tl.dataset.samples])
    pw = torch.tensor([(ys == 0).sum() / max((ys == 1).sum(), 1)], dtype=torch.float32, device=device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    stopper = EarlyStopping(cfg.train.early_stop_patience, "max")
    best = Path("results") / f"{tag}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        model.freeze_encoders(True)   # keep encoders in eval-ish frozen state
        run, seen = 0.0, 0
        for batch in tl:
            b = to_device(batch, device)
            logit = model(b)
            loss = crit(logit, b["label"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step(); sched.step()
            run += loss.item() * b["label"].size(0); seen += b["label"].size(0)
        s, y = evaluate(model, vl, device)
        m = compute(y, s)
        print(f"  [{tag}] epoch {epoch}: loss={run/seen:.4f} "
              f"val_pr_auc={m['pr_auc']:.4f} val_roc_auc={m['roc_auc']:.4f} f1={m['f1']:.4f}")
        if stopper.step(m["pr_auc"]):
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, best)
        if stopper.should_stop:
            print(f"  early stop at {epoch}")
            break
    if best.exists():
        model.load_state_dict(torch.load(best, map_location=device, weights_only=False)["state_dict"])
    return model


def main() -> int:
    from transformers import AutoTokenizer

    cfg = Config()
    ensure_dirs(cfg)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    out_dir = Path("results")
    print(f"device={device}")

    print("[data] assembling multimodal corpus")
    splits = assemble(cfg, n_per_class=6000)
    tok = AutoTokenizer.from_pretrained(cfg.text.model_name)
    sd = Path(cfg.data.screenshot_dir)

    def loader(key, shuffle):
        ds = MMDataset(splits[key], tok, sd, cfg)
        return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle,
                          collate_fn=collate)

    tl, vl, el = loader("train", True), loader("val", False), loader("test", False)

    ckpts = dict(
        url_ckpt=out_dir / "url_cnn_best.pt" if (out_dir / "url_cnn_best.pt").exists() else None,
        text_ckpt=out_dir / "text_distilbert_best.pt" if (out_dir / "text_distilbert_best.pt").exists() else None,
        visual_ckpt=out_dir / "visual_encoder_best.pt" if (out_dir / "visual_encoder_best.pt").exists() else None,
    )

    results = {}
    for fusion in ("cross_attention", "concat"):
        print(f"\n[train] fusion = {fusion}")
        model = PhishNet(cfg, fusion=fusion, with_text=True, with_visual=True)
        loaded = model.load_branch_weights(**ckpts, map_location=device)
        print(f"  warm-started encoders: {loaded}")
        model = train_one(model, tl, vl, device, cfg, tag=f"fusion_{fusion}")

        s_va, y_va = evaluate(model, vl, device)
        thr = threshold_for_fpr(y_va, s_va, 0.01)
        s_te, y_te = evaluate(model, el, device)
        results[fusion] = compute(y_te, s_te, thr)
        if fusion == "cross_attention":
            torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()},
                       out_dir / "phishnet_full.pt")
            np.savez(out_dir / "fusion_test_scores.npz", scores=s_te, labels=y_te, threshold=thr)

    print("\n" + format_table(results))
    (out_dir / "fusion_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_dir/'fusion_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
