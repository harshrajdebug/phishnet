"""The assembled PhishNet model.

Three encoders feed one fusion module. The encoders are interchangeable and the
fusion module is swappable (cross-attention vs concatenation), which is what
makes the ablation in `eval/benchmark.py` a controlled comparison: the same
trained encoders are held fixed and only the fusion head changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from phishnet.models.fusion import ConcatFusion, CrossModalFusion
from phishnet.models.url_cnn import URLEncoder


class PhishNet(nn.Module):
    def __init__(self, cfg, fusion: str = "cross_attention",
                 with_text: bool = True, with_visual: bool = True):
        super().__init__()
        self.cfg = cfg
        d = cfg.fusion.d_model

        self.url_encoder = URLEncoder(
            embed_dim=cfg.url.embed_dim, num_filters=cfg.url.num_filters,
            kernel_sizes=tuple(cfg.url.kernel_sizes), out_dim=d,
            dropout=cfg.url.dropout,
        )

        self.text_encoder = None
        if with_text:
            from phishnet.models.text_encoder import TextEncoder
            self.text_encoder = TextEncoder(
                model_name=cfg.text.model_name, out_dim=d,
                dropout=cfg.text.dropout, freeze_layers=cfg.text.freeze_layers,
            )

        self.visual_encoder = None
        if with_visual:
            from phishnet.models.visual_siamese import VisualEncoder
            self.visual_encoder = VisualEncoder(
                embed_dim=cfg.visual.embed_dim, out_dim=d,
                dropout=cfg.visual.dropout, backbone=cfg.visual.backbone,
                pretrained=cfg.visual.pretrained,
            )

        if fusion == "cross_attention":
            self.fusion = CrossModalFusion(
                d_model=d, n_heads=cfg.fusion.n_heads, n_layers=cfg.fusion.n_layers,
                dropout=cfg.fusion.dropout, ffn_mult=cfg.fusion.ffn_mult,
                modality_dropout=cfg.fusion.modality_dropout,
            )
        elif fusion == "concat":
            self.fusion = ConcatFusion(d_model=d, dropout=cfg.fusion.dropout)
        else:
            raise ValueError(f"unknown fusion: {fusion}")
        self.fusion_kind = fusion
        self.d_model = d

    def encode(self, batch: dict) -> tuple[torch.Tensor, ...]:
        """Run whichever encoders have inputs; substitute zeros elsewhere.

        A zero vector is never fed to the classifier directly -- the fusion
        module replaces absent slots with its learned `missing_token` -- so the
        placeholder here only has to be shape-correct.
        """
        device = next(self.parameters()).device
        B = batch["url_ids"].size(0)
        z = torch.zeros(B, self.d_model, device=device)

        url_z = self.url_encoder(batch["url_ids"])

        if self.text_encoder is not None and batch.get("input_ids") is not None:
            text_z = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        else:
            text_z = z

        if self.visual_encoder is not None and batch.get("image") is not None:
            visual_z = self.visual_encoder(batch["image"])
        else:
            visual_z = z
        return url_z, visual_z, text_z

    def forward(self, batch: dict, return_attention: bool = False):
        url_z, visual_z, text_z = self.encode(batch)
        present = batch.get("present")
        if present is None:
            device = url_z.device
            present = torch.stack([
                torch.ones(url_z.size(0), dtype=torch.bool, device=device),
                torch.full((url_z.size(0),), batch.get("image") is not None,
                           dtype=torch.bool, device=device),
                torch.full((url_z.size(0),), batch.get("input_ids") is not None,
                           dtype=torch.bool, device=device),
            ], dim=1)
        return self.fusion(url_z, visual_z, text_z, present,
                           return_attention=return_attention)

    def load_branch_weights(self, url_ckpt=None, text_ckpt=None, visual_ckpt=None,
                            map_location="cpu") -> list[str]:
        """Warm-start encoders from their unimodal training runs."""
        loaded = []
        pairs = [("url", url_ckpt, self.url_encoder),
                 ("text", text_ckpt, self.text_encoder),
                 ("visual", visual_ckpt, self.visual_encoder)]
        for name, ckpt, module in pairs:
            if ckpt is None or module is None:
                continue
            sd = torch.load(ckpt, map_location=map_location, weights_only=False)
            sd = sd.get("encoder_state", sd.get("state_dict", sd))
            sd = {k.replace("encoder.", "", 1): v for k, v in sd.items()}
            missing, unexpected = module.load_state_dict(sd, strict=False)
            loaded.append(f"{name}(missing={len(missing)},unexpected={len(unexpected)})")
        return loaded

    def freeze_encoders(self, freeze: bool = True) -> None:
        """Fusion-only training: the encoders stay fixed so the ablation
        isolates the fusion mechanism from further representation learning."""
        for enc in (self.url_encoder, self.text_encoder, self.visual_encoder):
            if enc is None:
                continue
            for p in enc.parameters():
                p.requires_grad = not freeze
