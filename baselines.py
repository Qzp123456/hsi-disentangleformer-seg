"""
baselines.py — comparison models for HSI semantic segmentation

All models share the same interface as DisentangleFormerSeg:
    input  (B, n_bands, H, W)  ->  output (B, num_classes, H, W)

Implemented:
  - ViTSeg          : plain Vision Transformer encoder + seg head (Dosovitskiy et al., 2021)
  - SpectralFormerSeg: spectral-sequence transformer with cross-layer adaptive
                       fusion, adapted for dense prediction (Hong et al., 2022)
  - CNN3DSeg        : 3D-CNN spectral-spatial baseline (classic HSI approach)
"""
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════
# Shared lightweight segmentation head
# ══════════════════════════════════════════════════════════
class SegHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(dim, num_classes, 1)

    def forward(self, x):
        return self.classifier(x + self.refine(x))


# ══════════════════════════════════════════════════════════
# 1. ViT — plain Vision Transformer, spatial tokens only
# ══════════════════════════════════════════════════════════
class ViTSeg(nn.Module):
    """Standard ViT encoder: each spatial position is a token, joint
    spatial-channel attention (i.e. NO disentanglement)."""
    def __init__(self, n_bands, num_classes, patch=32,
                 dim=64, depth=2, heads=4):
        super().__init__()
        num_tokens = patch * patch
        self.embed = nn.Sequential(
            nn.Conv2d(n_bands, dim, 3, padding=1),
            nn.BatchNorm2d(dim), nn.GELU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = SegHead(dim, num_classes)

    def forward(self, x):
        B, _, H, W = x.shape
        z = self.embed(x).flatten(2).transpose(1, 2)   # (B, N, dim)
        z = self.encoder(z + self.pos)
        z = self.norm(z).transpose(1, 2).reshape(B, -1, H, W)
        return self.head(z)


# ══════════════════════════════════════════════════════════
# 2. SpectralFormer — groupwise spectral embedding + CAF
# ══════════════════════════════════════════════════════════
class CAF(nn.Module):
    """Cross-layer Adaptive Fusion: gated fusion of the current layer
    output with the previous layer's representation."""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, cur, prev):
        g = self.gate(torch.cat([cur, prev], dim=-1))
        return g * cur + (1 - g) * prev


class SpectralFormerSeg(nn.Module):
    """Adapted from SpectralFormer (Hong et al., TGRS 2022).
    Key ideas retained: (i) groupwise spectral embedding (GSE) that mixes
    neighbouring bands rather than treating each band independently,
    (ii) cross-layer adaptive fusion (CAF) between transformer layers.
    Adapted here to dense prediction by attaching a segmentation head."""
    def __init__(self, n_bands, num_classes, patch=32,
                 dim=64, depth=2, heads=4, group=3):
        super().__init__()
        num_tokens = patch * patch
        # groupwise spectral embedding: 1D conv over the band axis
        self.gse = nn.Sequential(
            nn.Conv2d(n_bands, dim, 1),
            nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, group, padding=group // 2, groups=dim),
            nn.BatchNorm2d(dim), nn.GELU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.cafs = nn.ModuleList([CAF(dim) for _ in range(depth - 1)])
        self.norm = nn.LayerNorm(dim)
        self.head = SegHead(dim, num_classes)

    def forward(self, x):
        B, _, H, W = x.shape
        z = self.gse(x).flatten(2).transpose(1, 2)
        z = z + self.pos
        prev = z
        for i, layer in enumerate(self.layers):
            cur = layer(z)
            if i > 0:
                cur = self.cafs[i - 1](cur, prev)
            prev, z = z, cur
        z = self.norm(z).transpose(1, 2).reshape(B, -1, H, W)
        return self.head(z)


# ══════════════════════════════════════════════════════════
# 3. 3D-CNN — classic spectral-spatial convolution baseline
# ══════════════════════════════════════════════════════════
class CNN3DSeg(nn.Module):
    """3D convolutions over (band, height, width) to jointly model
    spectral and spatial context — the pre-transformer standard for HSI."""
    def __init__(self, n_bands, num_classes, patch=32, dim=64, depth=2, heads=4):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(n_bands, 32, 1), nn.BatchNorm2d(32), nn.GELU())
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(8), nn.GELU(),
            nn.Conv3d(8, 16, (5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(16), nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(16 * 32, dim, 1), nn.BatchNorm2d(dim), nn.GELU())
        self.head = SegHead(dim, num_classes)

    def forward(self, x):
        z = self.reduce(x)                    # (B, 32, H, W)
        z = z.unsqueeze(1)                    # (B, 1, 32, H, W)
        z = self.conv3d(z)                    # (B, 16, 32, H, W)
        B, C, D, H, W = z.shape
        z = z.reshape(B, C * D, H, W)
        return self.head(self.proj(z))


MODELS = {
    "vit": ViTSeg,
    "spectralformer": SpectralFormerSeg,
    "cnn3d": CNN3DSeg,
}
