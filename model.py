"""
model.py — DisentangleFormer with WINDOW-BASED attention (corrected)

Key fix vs. the previous version:
  The paper builds on Swin Transformer: ST/CT attention operates inside
  local windows of M x M tokens, NOT over the whole feature map.
  Previously the whole 32x32 map was flattened, making the CT path's
  d_model = 1024 and blowing the parameter count up to ~17M, which
  overfits badly on 48 training tiles.

  Now:  window_size M (default 8) -> CT d_model = 64, params ~1M.

Modules (paper Section 4):
  Eq.(3)  ST path  : TransformerEncoder over spatial tokens  (N=M*M, C)
  Eq.(4)  CT path  : TransformerEncoder over channel tokens  (C, N)
  Eq.(5-8) STE     : depthwise conv + channel gating + residual
  Eq.(9-11) MS-FFN : multi-branch depthwise conv, k in {1,3,5,7}

Plus our proposed gated CMA (bidirectional cross-stream attention).
"""
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
# window helpers (Swin-style)
# ──────────────────────────────────────────────────────────────
def window_partition(x, M):
    """(B, C, H, W) -> (B*nW, M*M, C)"""
    B, C, H, W = x.shape
    x = x.view(B, C, H // M, M, W // M, M)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()      # B, nH, nW, M, M, C
    return x.view(-1, M * M, C)


def window_reverse(win, M, H, W, B):
    """(B*nW, M*M, C) -> (B, C, H, W)"""
    C = win.shape[-1]
    x = win.view(B, H // M, W // M, M, M, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()      # B, C, nH, M, nW, M
    return x.view(B, C, H, W)


def valid_heads(d, h):
    """largest divisor of d that is <= h"""
    while d % h != 0:
        h -= 1
    return h


# ──────────────────────────────────────────────────────────────
# Squeezed Token Enhancer — Eq.(7)
# ──────────────────────────────────────────────────────────────
class STE(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        r = max(channels // reduction, 4)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.dw(x)
        return x + out * self.gate(out)


# ──────────────────────────────────────────────────────────────
# Multi-Scale FFN — Eq.(9)-(11)
# ──────────────────────────────────────────────────────────────
class MSFFN(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden = dim * expansion
        assert hidden % 4 == 0
        c = hidden // 4
        self.proj_in = nn.Conv2d(dim, hidden, 1)
        self.branches = nn.ModuleList([
            nn.Conv2d(c, c, k, padding=k // 2, groups=c) for k in (1, 3, 5, 7)])
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(hidden)
        self.proj_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        z = self.act(self.proj_in(x))
        ms = torch.cat([b(c) for b, c in
                        zip(self.branches, torch.chunk(z, 4, dim=1))], dim=1)
        z = self.act(self.norm(z + ms))
        return self.proj_out(z)


# ──────────────────────────────────────────────────────────────
# Gated CMA — OUR CONTRIBUTION
# operates inside windows, gamma zero-init for stability
# ──────────────────────────────────────────────────────────────
class CMAFusion(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        h = valid_heads(dim, heads)
        self.attn_s2c = nn.MultiheadAttention(dim, h, batch_first=True)
        self.attn_c2s = nn.MultiheadAttention(dim, h, batch_first=True)
        self.norm_s = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)
        self.gamma_s = nn.Parameter(torch.zeros(1))
        self.gamma_c = nn.Parameter(torch.zeros(1))

    def forward(self, r_s, r_c):
        z_s, _ = self.attn_s2c(r_s, r_c, r_c)
        z_c, _ = self.attn_c2s(r_c, r_s, r_s)
        return (self.norm_s(r_s + self.gamma_s * z_s),
                self.norm_c(r_c + self.gamma_c * z_c))


# ──────────────────────────────────────────────────────────────
# DisentangleFormer Block (window-based)
# ──────────────────────────────────────────────────────────────
class DisentangleFormerBlock(nn.Module):
    def __init__(self, dim, window_size=8, heads=4, use_cma=False, shift=False):
        super().__init__()
        self.M = window_size
        self.shift = shift
        self.use_cma = use_cma
        N = window_size * window_size          # tokens per window

        def enc(d, h):
            return nn.TransformerEncoderLayer(
                d_model=d, nhead=valid_heads(d, h), dim_feedforward=d * 2,
                batch_first=True, norm_first=True)

        self.norm_s = nn.LayerNorm(dim)
        self.st_encoder = nn.TransformerEncoder(enc(dim, heads), 1)

        self.norm_c = nn.LayerNorm(N)
        self.ct_encoder = nn.TransformerEncoder(enc(N, heads), 1)

        if use_cma:
            self.cma = CMAFusion(dim, heads)

        self.ste = STE(dim * 2)
        self.proj = nn.Conv2d(dim * 2, dim, 1)
        self.msffn = MSFFN(dim)

    def forward(self, x):                      # (B, C, H, W)
        B, C, H, W = x.shape
        M = self.M
        shortcut = x

        # cyclic shift for cross-window connectivity (Swin)
        if self.shift:
            x = torch.roll(x, shifts=(-M // 2, -M // 2), dims=(2, 3))

        win = window_partition(x, M)           # (B*nW, N, C)

        r_s = self.st_encoder(self.norm_s(win))                       # Eq.(3)
        r_c = self.ct_encoder(self.norm_c(win.transpose(1, 2)))       # Eq.(4)
        r_c = r_c.transpose(1, 2)

        if self.use_cma:
            r_s, r_c = self.cma(r_s, r_c)

        # Eq.(5)-(6) reshape windows back to 2D and concat
        m_s = window_reverse(r_s, M, H, W, B)
        m_c = window_reverse(r_c, M, H, W, B)
        m = torch.cat([m_s, m_c], dim=1)       # (B, 2C, H, W)

        y = self.proj(self.ste(m))             # Eq.(7)-(8)

        if self.shift:                          # undo shift
            y = torch.roll(y, shifts=(M // 2, M // 2), dims=(2, 3))

        y = shortcut + y                        # residual
        return y + self.msffn(y)                # MS-FFN residual


# ──────────────────────────────────────────────────────────────
# Classification model (patch-based)
# ──────────────────────────────────────────────────────────────
class DisentangleFormer(nn.Module):
    def __init__(self, n_bands, num_classes, patch_size=7,
                 dim=64, depth=2, heads=4, use_cma=False, window_size=None):
        super().__init__()
        M = window_size or patch_size          # for small patches use full patch
        while patch_size % M != 0:
            M -= 1
        self.embed = nn.Sequential(
            nn.Conv2d(n_bands, dim, 1), nn.BatchNorm2d(dim), nn.GELU())
        self.blocks = nn.ModuleList([
            DisentangleFormerBlock(dim, M, heads, use_cma, shift=(i % 2 == 1))
            for i in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        z = self.embed(x)
        for blk in self.blocks:
            z = blk(z)
        return self.head(z.mean(dim=(2, 3)))
