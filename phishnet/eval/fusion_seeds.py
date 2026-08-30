"""Multi-seed comparison of cross-attention vs concatenation fusion.

A single training run showed cross-attention beating concatenation by a large
margin on false-positive rate; a second run reversed it. On a task where both
mechanisms sit near the metric ceiling, a one-run difference is not evidence. This
script trains each mechanism across several seeds and reports mean +/- std plus a
paired per-seed difference, so the paper can state whether the advantage is real
or within noise -- rather than quoting whichever run flattered the hypothesis.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phishnet.config import Config, ensure_dirs, resolve_device, set_seed
from phishnet.models.phishnet import PhishNet
from phishnet.data.multimodal import assemble
from phishnet.train.train_fusion import MMDataset, collate, train_one
from phishnet.eval.metrics import compute, threshold_for_fpr


def run(seeds=(1, 2, 3, 4, 5)) -> dict:
    from transformers import AutoTokenizer
    cfg = Config()
    ensure_dirs(cfg)
    device = resolve_device(cfg.train.device)
    out = Path("results")
    tok = AutoTokenizer.from_pretrained(cfg.text.model_name)
    sd = Path(cfg.data.screenshot_dir)
    ckpts = dict(
        url_ckpt=out / "url_cnn_best.pt",
        text_ckpt=out / "text_distilbert_best.pt",
        visual_ckpt=out / "visual_encoder_best.pt",
    )

    rows = {"cross_attention": [], "concat": []}
    for seed in seeds:
        set_seed(seed)
        splits = assemble(cfg, n_per_class=6000, seed=seed)

        def loader(key, shuffle):
            ds = MMDataset(splits[key], tok, sd, cfg)
            return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle,
                              collate_fn=collate)
        tl, vl, el = loader("train", True), loader("val", False), loader("test", False)

        for fusion in ("cross_attention", "concat"):
            set_seed(seed)  # same init/data order for both mechanisms this seed
            model = PhishNet(cfg, fusion=fusion, with_text=True, with_visual=True)
            model.load_branch_weights(**ckpts, map_location=device)
            model = train_one(model, tl, vl, device, cfg, tag=f"seedcmp_{fusion}_{seed}")

            @torch.no_grad()
            def ev(loader):
                model.eval(); ss=[]; ys=[]
                for b in loader:
                    for k,v in b.items():
                        if torch.is_tensor(v): b[k]=v.to(device)
                    lg,_=model(b, return_attention=True)
                    ss.append(torch.sigmoid(lg).cpu().numpy()); ys.append(b["label"].cpu().numpy())
                return np.concatenate(ss), np.concatenate(ys)

            s_va,y_va = ev(vl); thr = threshold_for_fpr(y_va, s_va, 0.01)
            s_te,y_te = ev(el); m = compute(y_te, s_te, thr)
            rows[fusion].append({"seed": seed, "f1": m["f1"], "fpr": m["fpr"],
                                 "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"]})
            print(f"  seed {seed} {fusion:16s} f1={m['f1']:.4f} fpr={m['fpr']:.4f} pr_auc={m['pr_auc']:.4f}")

    def agg(key, metric):
        xs=[r[metric] for r in rows[key]]
        return st.mean(xs), (st.pstdev(xs) if len(xs)>1 else 0.0)

    summary={}
    for k in rows:
        summary[k]={m: {"mean": agg(k,m)[0], "std": agg(k,m)[1]} for m in ("f1","fpr","pr_auc","roc_auc")}
    # paired per-seed difference (cross - concat) on FPR and F1
    ca={r["seed"]:r for r in rows["cross_attention"]}; co={r["seed"]:r for r in rows["concat"]}
    dfpr=[ca[s]["fpr"]-co[s]["fpr"] for s in ca]; df1=[ca[s]["f1"]-co[s]["f1"] for s in ca]
    summary["paired_diff_cross_minus_concat"]={
        "fpr_mean": st.mean(dfpr), "fpr_std": st.pstdev(dfpr) if len(dfpr)>1 else 0.0,
        "f1_mean": st.mean(df1), "f1_std": st.pstdev(df1) if len(df1)>1 else 0.0}
    summary["raw"]=rows
    return summary


if __name__ == "__main__":
    s=run()
    Path("results/fusion_seed_comparison.json").write_text(json.dumps(s, indent=2))
    print("\n=== SUMMARY (mean +/- std over seeds) ===")
    for k in ("cross_attention","concat"):
        f=s[k]["f1"]; fp=s[k]["fpr"]
        print(f"  {k:16s} F1 {f['mean']:.4f}+/-{f['std']:.4f}   FPR {fp['mean']:.4f}+/-{fp['std']:.4f}")
    d=s["paired_diff_cross_minus_concat"]
    print(f"  paired (cross-concat): dFPR {d['fpr_mean']:+.4f}+/-{d['fpr_std']:.4f}  dF1 {d['f1_mean']:+.4f}+/-{d['f1_std']:.4f}")
    print("saved -> results/fusion_seed_comparison.json")
