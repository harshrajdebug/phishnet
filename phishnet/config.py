"""Central configuration for PhishNet.

Every path and hyper-parameter the pipeline touches lives here so that a run can
be reproduced from a single object. Values can be overridden from a YAML file
passed to `load_config`, which keeps experiment sweeps out of the source tree.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


@dataclass
class URLConfig:
    """Character-level 1D-CNN over the raw URL string."""
    max_len: int = 200
    embed_dim: int = 64
    # Parallel convolution widths: short filters catch character n-grams such as
    # "-secure-", wide ones catch whole tokens like "webhostapp".
    kernel_sizes: tuple[int, ...] = (3, 4, 5, 6)
    num_filters: int = 128
    dropout: float = 0.3
    out_dim: int = 256


@dataclass
class TextConfig:
    """DistilBERT encoder over the message body."""
    model_name: str = "distilbert-base-uncased"
    max_len: int = 256
    out_dim: int = 256
    dropout: float = 0.2
    freeze_layers: int = 2  # freeze embeddings + first N transformer blocks


@dataclass
class VisualConfig:
    """Siamese CNN over the rendered screenshot."""
    image_size: int = 224
    embed_dim: int = 256
    out_dim: int = 256
    dropout: float = 0.2
    backbone: str = "resnet18"
    pretrained: bool = True


@dataclass
class FusionConfig:
    """Cross-modal attention fusion."""
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    ffn_mult: int = 2
    # Probability of dropping a whole modality during training. Without this the
    # fusion layer learns to depend on a modality that is often missing at
    # inference time (no screenshot yet, no email context on a direct visit).
    modality_dropout: float = 0.15


@dataclass
class TrainConfig:
    seed: int = 1337
    batch_size: int = 64
    epochs: int = 8
    lr: float = 3e-4
    bert_lr: float = 2e-5
    weight_decay: float = 1e-2
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    early_stop_patience: int = 3
    num_workers: int = 0
    device: str = "auto"


@dataclass
class DataConfig:
    raw_dir: Path = DATA / "raw"
    processed_dir: Path = DATA / "processed"
    screenshot_dir: Path = DATA / "screenshots"
    artifact_dir: Path = DATA / "artifacts"
    # Realistic deployment base rate. Public benchmarks are usually balanced,
    # which inflates precision by an order of magnitude relative to a real
    # browsing stream. We evaluate at both.
    eval_base_rate: float = 0.02
    max_phish_urls: int = 60000
    max_benign_urls: int = 60000
    val_frac: float = 0.15
    test_frac: float = 0.15


@dataclass
class Config:
    url: URLConfig = field(default_factory=URLConfig)
    text: TextConfig = field(default_factory=TextConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        def _enc(o):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, tuple):
                return list(o)
            return o
        return json.loads(json.dumps(asdict(self), default=_enc))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def resolve_device(pref: str = "auto") -> str:
    """Pick the best available accelerator.

    Apple silicon exposes MPS, which is roughly 4x faster than CPU for the
    convolutional branches here but still falls back for a few sparse ops.
    """
    import torch

    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    import yaml  # optional dependency; only needed for overrides

    raw = yaml.safe_load(Path(path).read_text()) or {}
    for section, values in raw.items():
        target = getattr(cfg, section, None)
        if target is None:
            raise KeyError(f"unknown config section: {section}")
        for k, v in values.items():
            if not hasattr(target, k):
                raise KeyError(f"unknown key {section}.{k}")
            setattr(target, k, v)
    return cfg


def ensure_dirs(cfg: Config) -> None:
    for p in (cfg.data.raw_dir, cfg.data.processed_dir, cfg.data.screenshot_dir,
              cfg.data.artifact_dir, RESULTS):
        Path(p).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
