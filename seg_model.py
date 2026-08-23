"""
seg_model.py — DisentangleFormer for semantic segmentation (window-based)
"""
import torch
import torch.nn as nn

from model import DisentangleFormerBlock


class SegDecoder(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim), nn.GELU(),
        )
        self.classifier = nn.Conv2d(dim, num_classes, 1)

    def forward(self, x):
        return self.classifier(x + self.refine(x))


class DisentangleFormerSeg(nn.Module):
    def __init__(self, n_bands, num_classes, patch=32,
                 dim=64, depth=2, heads=4, use_cma=False, window_size=8):
        super().__init__()
        M = window_size
        while patch % M != 0:
            M -= 1
        self.window_size = M

        self.embed = nn.Sequential(
            nn.Conv2d(n_bands, dim, 3, padding=1),
            nn.BatchNorm2d(dim), nn.GELU())

        self.blocks = nn.ModuleList([
            DisentangleFormerBlock(dim, M, heads, use_cma, shift=(i % 2 == 1))
            for i in range(depth)])

        self.decoder = SegDecoder(dim, num_classes)

    def forward(self, x):                      # (B, n_bands, H, W)
        z = self.embed(x)
        for blk in self.blocks:
            z = blk(z)
        return self.decoder(z)                 # (B, num_classes, H, W)
