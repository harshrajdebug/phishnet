"""Modality ablation on the fused model, all on one held-out test set.

The fusion comparison in `train_fusion` answers "cross-attention vs concat".
This answers the complementary question the paper's central claim needs:
how much does each modality contribute, measured on the *same* samples with the
*same* frozen weights? We do it by overriding the `present` mask at inference,
so the fusion module falls back to its learned missing-modality token for the
ablated inputs -- exactly the code path used when a modality is genuinely absent
in deployment. No retraining, no separate test set, so the numbers are directly
comparable to the full-fusion row.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phishnet.config import Config, resolve_device, ensure_dirs, set_seed
from phishnet.models.phishnet import PhishNet
from phishnet.data.multimodal import assemble
from phishnet.train.train_fusion import MMDataset, collate
from phishnet.eval.metrics import compute, threshold_for_fpr, format_table


@torch.no_grad()
def evaluate_masked(model, loader, device, mask: tuple[int, int, int]):
    """Score the whole loader forcing the modality-presence mask to `mask`.

    A modality is only *revealed* if it is both requested by `mask` and actually
    present in the sample; we never fabricate an input that does not exist.
    """
    model.eval()
    m = torch.tensor(mask, dtype=torch.bool, device=device)
    scores, labels = [], []
    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        real = batch["present"].to(device)          # what each sample truly has
        batch["present"] = real & m.unsqueeze(0)    # AND with the ablation mask
        logit, _ = model(batch, return_attention=True)
        scores.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    s = np.concatenate(scores)
    # Guard: a fully-masked sample can yield a non-finite score. Map it to the
    # benign extreme rather than crashing the metric computation.
    s = np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
    return s, np.concatenate(labels)


def main() -> int:
    from transformers import AutoTokenizer

    cfg = Config()
    ensure_dirs(cfg)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    out_dir = Path("results")
    print(f"device={device}")

    ckpt = out_dir / "phishnet_full.pt"
    if not ckpt.exists():
        print("phishnet_full.pt missing; run train_fusion first")
        return 1

    print("[data] assembling multimodal corpus (same split as training)")
    splits = assemble(cfg, n_per_class=6000)
    tok = AutoTokenizer.from_pretrained(cfg.text.model_name)
    sd = Path(cfg.data.screenshot_dir)

    def loader(key):
        ds = MMDataset(splits[key], tok, sd, cfg)
        return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                          collate_fn=collate)

    vl, el = loader("val"), loader("test")

    model = PhishNet(cfg, fusion="cross_attention", with_text=True, with_visual=True)
    state = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(state)
    model = model.to(device)

    # (url, visual, text). URL is the always-available anchor in deployment (it
    # exists the moment a link is seen, before any page loads), so every
    # configuration keeps it and we measure what text and visual *add* on top.
    # Masks that remove the anchor would leave URL-only samples with no inputs at
    # all, which is not a scenario the system ever faces.
    configs = {
        "url_only":    (1, 0, 0),
        "url+text":    (1, 0, 1),
        "url+visual":  (1, 1, 0),
        "full_fusion": (1, 1, 1),
    }

    results = {}
    thresholds = {}
    for name, mask in configs.items():
        # Fix the operating threshold on validation under the same mask, then
        # report on test -- never peek at the test set to pick a threshold.
        s_va, y_va = evaluate_masked(model, vl, device, mask)
        thr = threshold_for_fpr(y_va, s_va, 0.01)
        thresholds["".join(str(b) for b in mask)] = float(thr)
        s_te, y_te = evaluate_masked(model, el, device, mask)
        results[name] = compute(y_te, s_te, thr)
        print(f"  {name:14s} thr={thr:.3f} f1={results[name]['f1']:.4f} "
              f"pr_auc={results[name]['pr_auc']:.4f} fpr={results[name]['fpr']:.4f}")

    print("\n" + format_table(results))
    (out_dir / "ablation_results.json").write_text(json.dumps(results, indent=2))
    # Per-configuration operating thresholds, keyed by a "url visual text" bit
    # string, so the serving engine can apply the threshold that matches whichever
    # modalities are actually present for a given scan rather than a single
    # threshold calibrated on the all-modalities-present distribution.
    (out_dir / "deployment_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    print(f"saved -> {out_dir/'ablation_results.json'} and deployment_thresholds.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
