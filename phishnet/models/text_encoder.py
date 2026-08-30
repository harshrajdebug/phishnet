"""DistilBERT encoder for message text (email / SMS lure body).

DistilBERT rather than BERT because the extension budget is a few hundred
milliseconds end-to-end: it retains ~97% of BERT's GLUE score at ~60% of the
runtime. The lower layers are frozen -- generic English syntax does not need
re-learning from 6k messages, and freezing them measurably reduces overfitting
on a corpus this size.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased",
                 out_dim: int = 256, dropout: float = 0.2, freeze_layers: int = 2):
        super().__init__()
        from transformers import AutoModel

        # Force the eager attention path. The default SDPA kernel raises
        # NotImplementedError on Apple-silicon MPS when attention dropout is
        # active during training; eager attention supports it and the speed
        # difference is negligible at our sequence length (<=256).
        try:
            self.backbone = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        except (TypeError, ValueError):
            self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.dim if hasattr(self.backbone.config, "dim") \
            else self.backbone.config.hidden_size
        self._freeze(freeze_layers)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(hidden, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def _freeze(self, n: int) -> None:
        if n <= 0:
            return
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        layers = getattr(self.backbone, "transformer", None)
        layers = layers.layer if layers is not None else self.backbone.encoder.layer
        for layer in layers[:n]:
            for p in layer.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                return_tokens: bool = False):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                           # (B, T, H)
        # Mean-pool over real tokens rather than taking [CLS]. Without a
        # next-sentence objective DistilBERT's [CLS] is not a trained summary,
        # and mean-pooling is consistently the stronger sentence representation.
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        z = self.norm(self.project(self.dropout(pooled)))
        return (z, h) if return_tokens else z


class TextClassifier(nn.Module):
    def __init__(self, encoder: TextEncoder | None = None, **kw):
        super().__init__()
        self.encoder = encoder or TextEncoder(**kw)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 128), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask):
        return self.head(self.encoder(input_ids, attention_mask)).squeeze(-1)
