"""
eval_perclass.py — per-class IoU under the spatial (no-leakage) protocol

Loads seg_*_r60_s{seed}_sp_best.pth for every method, rebuilds the model and
the spatial test split (left 60% train / right 40% test), then prints per-class
IoU (mean over seeds) so the thesis can show WHICH classes degrade when the
leaky random protocol is replaced by an honest spatial one.

Usage:
  python eval_perclass.py                      # all 5 methods, seeds 42/123/2024
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from seg_dataset import make_seg_splits
from seg_model import DisentangleFormerSeg
from baselines import MODELS

IGNORE = 255
CLASS_NAMES = {
    "indian_pines": ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
                     'Grass-trees', 'Grass-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill',
                     'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods', 'Buildings',
                     'Stone-Towers'],
    "pavia": ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal',
              'Bare Soil', 'Bitumen', 'Bricks', 'Shadows'],
}
METHODS = ["cnn3d", "vit", "spectralformer", "disentangle", "ours_cma"]


def build_model(name, n_bands, n_cls, patch, dim, depth, heads, use_cma):
    if name in ("disentangle", "ours_cma"):
        return DisentangleFormerSeg(n_bands, n_cls, patch,
                                    dim=dim, depth=depth, heads=heads, use_cma=use_cma)
    cls = MODELS[name]
    return cls(n_bands, n_cls, patch, dim=dim, depth=depth, heads=heads)


@torch.no_grad()
def eval_iou(model, loader, n_cls, device):
    model.eval()
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu().numpy().ravel()
        gt = y.numpy().ravel()
        m = gt != IGNORE
        pred, gt = pred[m], gt[m]
        conf += np.bincount(n_cls * gt + pred,
                            minlength=n_cls ** 2).reshape(n_cls, n_cls)
    tp = np.diag(conf)
    return tp / (conf.sum(0) + conf.sum(1) - tp).clip(min=1)   # nan = no GT pixels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="/mnt/scratch/znzs0468/results")
    ap.add_argument("--data_root", default="/mnt/scratch/znzs0468/data")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--tag", default="sp", choices=["sp", "ch"],
                    help="checkpoint tag suffix: sp=spatial(lr), ch=checker")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ds in ["indian_pines", "pavia"]:
        names = CLASS_NAMES[ds]
        print("\n" + "=" * 100)
        proto = "checkerboard" if args.tag == "ch" else "spatial (lr)"
        print(f"[{ds}] {proto} protocol — per-class IoU (mean over seeds {args.seeds})")
        print("=" * 100)

        # iou[method][seed] -> (n_cls,) array
        ious = {}
        for mname in METHODS:
            per_seed = []
            for seed in args.seeds:
                path = f"{args.results_dir}/seg_{ds}_{mname}_r60_s{seed}_{args.tag}_best.pth"
                if not os.path.exists(path):
                    print(f"[MISSING] {path}")
                    break
                ck = torch.load(path, map_location="cpu", weights_only=False)
                a = ck.get("args", {}) or {}
                use_cma = bool(a.get("use_cma", mname == "ours_cma"))
                _, test_set, n_cls, n_bands = make_seg_splits(
                    ds, args.data_root, a.get("patch", 32), a.get("stride", 16),
                    a.get("train_ratio", 0.6), int(a.get("seed", seed)),
                    a.get("split", "spatial"))
                model = build_model(mname, n_bands, n_cls, a.get("patch", 32),
                                    a.get("dim", 64), a.get("depth", 2),
                                    a.get("heads", 4), use_cma).to(device)
                model.load_state_dict(ck["model"])
                loader = DataLoader(test_set, batch_size=16, shuffle=False)
                per_seed.append(eval_iou(model, loader, n_cls, device))
            if len(per_seed) == len(args.seeds):
                ious[mname] = np.stack(per_seed)   # (seeds, n_cls)

        hdr = f"{'class':<16}" + "".join(f"{m:>16}" for m in METHODS if m in ious)
        print(hdr)
        print("-" * len(hdr))
        for c in range(len(names)):
            row = f"{names[c]:<16}"
            for mname in METHODS:
                if mname not in ious:
                    continue
                v = ious[mname][:, c] * 100
                if np.isnan(v).any():
                    row += f"{'--':>16}"
                else:
                    row += f"{v.mean():>16.1f}"
            print(row)
        print(f"{'mIoU':<16}" +
              "".join(f"{np.nanmean(ious[m]) * 100:>16.1f}" for m in METHODS if m in ious))


if __name__ == "__main__":
    main()
