"""Character-level 1D-CNN encoder for URLs.

Design follows the CharCNN/URLNet line of work: an embedding over characters,
several parallel convolutions of different widths, and max-over-time pooling.
The parallel widths matter -- a width-3 filter fires on "-ap" style homoglyph
fragments while a width-6 filter fires on whole tokens like "verify" -- and
max-pooling makes the encoder insensitive to *where* in the URL the evidence
sits, which is necessary because attackers move it around.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phishnet.features.url_encoding import VOCAB_SIZE, PAD


class URLEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64, num_filters: int = 128,
                 kernel_sizes: tuple[int, ...] = (3, 4, 5, 6),
                 out_dim: int = 256, dropout: float = 0.3,
                 vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(num_filters * len(kernel_sizes), out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, return_maps: bool = False):
        """x: (B, L) int64 character ids -> (B, out_dim).

        With return_maps=True we also hand back the pre-pool activation maps,
        which the explanation layer turns into a per-character saliency trace.
        """
        e = self.embedding(x).transpose(1, 2)          # (B, E, L)
        feats, maps = [], []
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(e)))                    # (B, F, L')
            if return_maps:
                maps.append(h)
            feats.append(torch.amax(h, dim=2))         # max-over-time
        z = self.dropout(torch.cat(feats, dim=1))
        z = self.norm(F.gelu(self.project(z)))
        return (z, maps) if return_maps else z


class URLClassifier(nn.Module):
    """Standalone URL model: encoder plus a linear head.

    Trained on its own first, then its weights initialise the URL branch of the
    fusion model. Warm-starting each branch separately converges faster and
    keeps the ablation honest -- the unimodal baseline in the results table is
    literally this module, not a crippled version of the fused one.
    """

    def __init__(self, encoder: URLEncoder | None = None, **kw):
        super().__init__()
        self.encoder = encoder or URLEncoder(**kw)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 128), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)   # logits
