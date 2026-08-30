"""Train the Siamese visual encoder and build the brand reference index."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from phishnet.config import Config, ensure_dirs, resolve_device, set_seed
from phishnet.eval.metrics import compute, format_table
from phishnet.models.visual_siamese import BrandIndex, VisualEncoder


def load_image(path: Path, size: int = 224) -> torch.Tensor:
    """Load and ImageNet-normalise a screenshot for a pretrained backbone."""
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (arr - mean) / std


class TripletDataset(Dataset):
    """Emit (anchor, positive, negative) triplets grouped by brand.

    Positive = same brand, different rendering. Negative = a different brand or a
    neutral page. This is what teaches "same brand => close" rather than
    "phishing => close", which is the property that generalises to unseen brands.
    """

    def __init__(self, samples, screenshot_dir: Path, size: int = 224,
                 n_triplets: int = 2000, seed: int = 1337):
        self.dir = screenshot_dir
        self.size = size
        self.by_brand: dict[str, list[str]] = {}
        for s in samples:
            self.by_brand.setdefault(s["brand"], []).append(s["path"])
        self.brands = [b for b, v in self.by_brand.items() if len(v) >= 2]
        rng = random.Random(seed)
        self.triplets = []
        for _ in range(n_triplets):
            pos_brand = rng.choice(self.brands)
            neg_brand = rng.choice([b for b in self.brands if b != pos_brand])
            a, p = rng.sample(self.by_brand[pos_brand], 2)
            n = rng.choice(self.by_brand[neg_brand])
            self.triplets.append((a, p, n))
        self._cache: dict[str, torch.Tensor] = {}

    def _img(self, name: str) -> torch.Tensor:
        if name not in self._cache:
            self._cache[name] = load_image(self.dir / name, self.size)
        return self._cache[name]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, i):
        a, p, n = self.triplets[i]
        return self._img(a), self._img(p), self._img(n)


def evaluate_detection(encoder, index: BrandIndex, samples, screenshot_dir: Path,
                       device, sim_threshold: float = 0.85) -> dict:
    """Score each sample by its nearest-brand similarity and evaluate as a
    detector: high similarity to a known brand anchor == impersonation."""
    encoder.eval()
    scores, labels = [], []
    with torch.no_grad():
        for s in samples:
            img = load_image(screenshot_dir / s["path"]).unsqueeze(0).to(device)
            z = encoder(img)
            top = index.query(z.cpu(), k=1)
            scores.append(top[0][2] if top else 0.0)
            labels.append(s["label"])
    return compute(labels, scores, sim_threshold)


def main() -> int:
    cfg = Config()
    ensure_dirs(cfg)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    sd = Path(cfg.data.screenshot_dir)
    out_dir = Path("results")
    print(f"device={device}")

    man_path = sd / "manifest.json"
    if not man_path.exists():
        from phishnet.data.render import build_visual_corpus
        print("[data] rendering visual corpus")
        build_visual_corpus(sd)
    manifest = json.loads(man_path.read_text())
    # Defensive: drop any manifest entry whose screenshot failed to render, so a
    # partial render never crashes training mid-epoch.
    samples = [s for s in manifest["samples"] if (sd / s["path"]).exists()]
    manifest["anchors"] = {b: a for b, a in manifest["anchors"].items()
                           if (sd / a).exists()}
    dropped = len(manifest["samples"]) - len(samples)
    if dropped:
        print(f"  warning: {dropped} manifest samples missing on disk, skipped")

    # Brand-disjoint split for the zero-shot test: hold out whole brands so the
    # test set measures generalisation to identities never seen in training.
    brands = sorted({s["brand"] for s in samples if s["label"] == 1})
    rng = random.Random(cfg.train.seed)
    rng.shuffle(brands)
    n_holdout = max(2, len(brands) // 3)
    holdout = set(brands[:n_holdout])
    train_samples = [s for s in samples if s["brand"] not in holdout]
    test_samples = [s for s in samples if s["brand"] in holdout
                    or s["brand"].startswith("neutral")]
    print(f"  seen brands: {len(brands)-n_holdout}  zero-shot holdout: {sorted(holdout)}")

    encoder = VisualEncoder(embed_dim=cfg.visual.embed_dim, out_dim=cfg.fusion.d_model,
                            dropout=cfg.visual.dropout, backbone=cfg.visual.backbone,
                            pretrained=cfg.visual.pretrained).to(device)

    ds = TripletDataset(train_samples, sd, cfg.visual.image_size, n_triplets=2400)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(encoder.parameters(), lr=2e-4, weight_decay=1e-4)

    print("[train] Siamese triplet training")
    encoder.train()
    for epoch in range(1, 7):
        tot, dp, dn, nb = 0.0, 0.0, 0.0, 0
        for a, p, n in dl:
            a, p, n = a.to(device), p.to(device), n.to(device)
            za, zp, zn = encoder(a), encoder(p), encoder(n)
            d_pos = 1 - (za * zp).sum(1)
            d_neg = 1 - (za * zn).sum(1)
            loss = F.relu(d_pos - d_neg + 0.3).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); dp += d_pos.mean().item(); dn += d_neg.mean().item(); nb += 1
        print(f"  epoch {epoch}: triplet_loss={tot/nb:.4f} "
              f"d_pos={dp/nb:.3f} d_neg={dn/nb:.3f}")

    # Build the reference index from the anchors of the SEEN brands only; the
    # zero-shot brands are then matched by adding a single anchor each at test
    # time, demonstrating enrolment without retraining.
    index = BrandIndex()
    encoder.eval()
    with torch.no_grad():
        for brand, aname in manifest["anchors"].items():
            img = load_image(sd / aname).unsqueeze(0).to(device)
            index.add(brand, f"{brand.lower()}.com", encoder(img).cpu())
    index.save(out_dir / "brand_index.pt")

    results = {}
    results["visual_seen"] = evaluate_detection(
        encoder, index, [s for s in samples if s["brand"] not in holdout], sd, device)
    results["visual_zeroshot"] = evaluate_detection(encoder, index, test_samples, sd, device)

    print("\n" + format_table(results, keys=["n", "accuracy", "precision",
          "recall", "f1", "roc_auc", "pr_auc"]))
    torch.save({"encoder_state": encoder.state_dict()}, out_dir / "visual_encoder_best.pt")
    (out_dir / "visual_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_dir/'visual_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
