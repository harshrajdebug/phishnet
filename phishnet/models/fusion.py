"""Cross-modal attention fusion.

The claim this module exists to test: naive fusion (concatenate the three
embeddings, feed an MLP) forces each modality to be interpreted in isolation
before the classifier ever sees the others, which is precisely wrong for
phishing. The evidence is *relational*. A screenshot that looks exactly like
Microsoft 365 is not suspicious on its own -- microsoft.com looks like that too.
It becomes suspicious only when read against a URL whose registrable domain is
not Microsoft's. Concatenation cannot represent "this, given that"; attention
can, because the query from one modality selects what to read from another.

Each modality attends to both others, so the URL branch can be modulated by the
visual brand claim and vice versa. Modality dropout during training prevents the
fused model from silently becoming a single-modality model with extra
parameters, which is the usual failure of multimodal architectures on datasets
where one modality happens to be strongest.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ("url", "visual", "text")


class CrossModalBlock(nn.Module):
    """One round of: attend to the other modalities, then a position-wise FFN."""

    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.2,
                 ffn_mult: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * ffn_mult, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None):
        """query: (B, 1, D) -- the attending modality.
        context: (B, M, D) -- the modalities being read.
        key_padding_mask: (B, M) True where a modality is absent.
        """
        attended, weights = self.attn(
            query, context, context,
            key_padding_mask=key_padding_mask, need_weights=True,
        )
        x = self.norm1(query + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, weights


class CrossModalFusion(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.2, ffn_mult: int = 2,
                 modality_dropout: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.modality_dropout = modality_dropout

        # A learned type embedding per modality. Attention is permutation
        # invariant, so without this the model cannot tell which slot it is
        # reading -- "the URL says X" and "the text says X" would be identical.
        self.type_embed = nn.Parameter(torch.randn(len(MODALITIES), d_model) * 0.02)
        # A learned stand-in for an absent modality, so a missing screenshot is
        # represented explicitly rather than as a zero vector the model must
        # guess the meaning of.
        self.missing_token = nn.Parameter(torch.randn(len(MODALITIES), d_model) * 0.02)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({m: CrossModalBlock(d_model, n_heads, dropout, ffn_mult)
                           for m in MODALITIES})
            for _ in range(n_layers)
        ])
        self.n_layers = n_layers

        # Gated pooling: the model weights each modality's final state by a
        # learned, input-dependent confidence rather than averaging blindly.
        self.gate = nn.Sequential(
            nn.Linear(d_model * len(MODALITIES), len(MODALITIES)), nn.Softmax(dim=-1)
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def _apply_modality_dropout(self, present: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout <= 0:
            return present
        drop = torch.rand_like(present.float()) < self.modality_dropout
        kept = present & ~drop
        # Never drop every modality: an all-absent sample carries no signal and
        # only injects label noise.
        empty = kept.sum(dim=1) == 0
        if empty.any():
            kept[empty, 0] = present[empty, 0]
            still = kept.sum(dim=1) == 0
            if still.any():
                kept[still] = present[still]
        return kept

    def forward(self, url_z: torch.Tensor, visual_z: torch.Tensor,
                text_z: torch.Tensor, present: torch.Tensor | None = None,
                return_attention: bool = False):
        """Each *_z is (B, D). `present` is (B, 3) bool over (url, visual, text)."""
        B = url_z.size(0)
        if present is None:
            present = torch.ones(B, len(MODALITIES), dtype=torch.bool, device=url_z.device)
        present = self._apply_modality_dropout(present)

        stack = torch.stack([url_z, visual_z, text_z], dim=1)      # (B, 3, D)
        miss = self.missing_token.unsqueeze(0).expand(B, -1, -1)
        stack = torch.where(present.unsqueeze(-1), stack, miss)
        stack = stack + self.type_embed.unsqueeze(0)

        # Absent modalities are masked out of every attention computation, so
        # they contribute nothing rather than contributing a learned constant.
        pad_mask = ~present

        states = [stack[:, i:i + 1, :] for i in range(len(MODALITIES))]
        attn_log: list[dict[str, torch.Tensor]] = []
        for layer in self.blocks:
            context = torch.cat(states, dim=1) + self.type_embed.unsqueeze(0)
            new_states, layer_attn = [], {}
            for i, m in enumerate(MODALITIES):
                q = states[i]
                out, w = layer[m](q, context, key_padding_mask=pad_mask)
                new_states.append(out)
                layer_attn[m] = w.detach()
            states = new_states
            attn_log.append(layer_attn)

        final = torch.cat(states, dim=1)                            # (B, 3, D)
        flat = final.reshape(B, -1)
        gates = self.gate(flat)                                     # (B, 3)
        gates = gates.masked_fill(pad_mask, 0.0)
        gates = gates / gates.sum(dim=1, keepdim=True).clamp(min=1e-9)
        pooled = self.out_norm((final * gates.unsqueeze(-1)).sum(dim=1))
        logit = self.head(pooled).squeeze(-1)

        if return_attention:
            return logit, {"gates": gates.detach(), "attention": attn_log}
        return logit


class ConcatFusion(nn.Module):
    """Naive baseline: concatenate and classify.

    Parameter count is matched to CrossModalFusion as closely as an MLP allows,
    so the ablation measures the fusion mechanism rather than model capacity.
    """

    def __init__(self, d_model: int = 256, dropout: float = 0.2, hidden: int = 1792):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 3, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 128), nn.GELU(), nn.Linear(128, 1),
        )

    def forward(self, url_z, visual_z, text_z, present=None, return_attention=False):
        if present is not None:
            m = present.unsqueeze(-1).float()
            url_z = url_z * m[:, 0]
            visual_z = visual_z * m[:, 1]
            text_z = text_z * m[:, 2]
        logit = self.net(torch.cat([url_z, visual_z, text_z], dim=1)).squeeze(-1)
        if return_attention:
            return logit, {"gates": None, "attention": None}
        return logit
