"""Siamese visual encoder for webpage screenshots.

The visual question in phishing is not "is this screenshot malicious" -- a
pixel-perfect clone of a bank login page is visually indistinguishable from the
real one. The question is "does this page look like brand X while being served
from a domain that is not X's". So the encoder is trained as a *similarity*
model: an embedding space where two renderings of the same brand are close and
two different brands are far apart. Detection is then a lookup against a
reference set of legitimate brand embeddings.

This framing is what gives zero-shot coverage. Adding a new brand means adding
one reference screenshot, not retraining -- which matters because APWG counted
766 distinct brands targeted in a single quarter.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, out_dim: int = 256,
                 dropout: float = 0.2, backbone: str = "resnet18",
                 pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm

        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(tvm, backbone)(weights=weights)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        # Keep the last residual stage addressable so Grad-CAM has a spatial
        # map to attribute over.
        self.target_layer = self.backbone.layer4
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Sequential(
            nn.Linear(feat_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, out_dim)
        )
        self.out_dim = out_dim
        self._activations = None
        self._gradients = None

    def _hook(self):
        """Attach forward/backward hooks used only during explanation."""
        def fwd(_m, _i, o):
            self._activations = o
            if o.requires_grad:
                o.register_hook(lambda g: setattr(self, "_gradients", g))
        return self.target_layer.register_forward_hook(fwd)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        z = self.project(self.dropout(self.backbone(x)))
        # L2-normalise so that cosine similarity is a plain dot product and the
        # margin in the triplet loss has a consistent geometric meaning.
        return F.normalize(z, p=2, dim=1) if normalize else z


class SiameseVisual(nn.Module):
    """Wraps the encoder with the two training objectives it needs.

    Triplet loss shapes the metric space; a small binary head on the *pair*
    (embedding, reference-similarity) turns a similarity into a phishing
    decision, because high similarity is only suspicious when the domain does
    not match the brand.
    """

    def __init__(self, encoder: VisualEncoder | None = None, **kw):
        super().__init__()
        self.encoder = encoder or VisualEncoder(**kw)
        self.out_dim = self.encoder.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def triplet_loss(self, anchor, positive, negative, margin: float = 0.3):
        a, p, n = self(anchor), self(positive), self(negative)
        # Cosine distance in [0, 2]; margin 0.3 chosen so that a same-brand pair
        # must be meaningfully closer than a cross-brand pair, not merely tied.
        d_pos = 1 - (a * p).sum(1)
        d_neg = 1 - (a * n).sum(1)
        return F.relu(d_pos - d_neg + margin).mean(), d_pos.mean(), d_neg.mean()


class BrandIndex:
    """Reference embeddings for legitimate brand pages.

    A screenshot is 'impersonating' when its nearest reference exceeds a
    similarity threshold. Kept deliberately simple (exact cosine over a few
    hundred vectors) -- at this size an approximate index would add dependency
    weight and latency for no measurable gain.
    """

    def __init__(self):
        self.names: list[str] = []
        self.domains: list[str] = []
        self.vectors: torch.Tensor | None = None

    def add(self, name: str, domain: str, vec: torch.Tensor) -> None:
        v = F.normalize(vec.detach().float().reshape(1, -1), p=2, dim=1)
        self.names.append(name)
        self.domains.append(domain)
        self.vectors = v if self.vectors is None else torch.cat([self.vectors, v], 0)

    def query(self, vec: torch.Tensor, k: int = 1):
        """Return the top-k (brand, legitimate_domain, cosine_similarity)."""
        if self.vectors is None or not self.names:
            return []
        v = F.normalize(vec.detach().float().reshape(1, -1), p=2, dim=1)
        sims = (self.vectors @ v.T).squeeze(1)
        k = min(k, len(self.names))
        top = torch.topk(sims, k)
        return [(self.names[i], self.domains[i], float(top.values[j]))
                for j, i in enumerate(top.indices.tolist())]

    def save(self, path) -> None:
        torch.save({"names": self.names, "domains": self.domains,
                    "vectors": self.vectors}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        idx = cls()
        idx.names, idx.domains, idx.vectors = d["names"], d["domains"], d["vectors"]
        return idx
