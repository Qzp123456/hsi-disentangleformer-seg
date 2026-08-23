"""
analyze_split_classes.py — per-class labelled-pixel counts in train vs test
region for different spatial splits, to diagnose missing classes.

Splits:
  lr      : left train_ratio of the width = train, rest = test
  checker : block x block cells, (r+c)%2==0 = train, odd = test

Usage:
  python analyze_split_classes.py --dataset indian_pines --split lr
  python analyze_split_classes.py --dataset indian_pines --split checker --block 64
  python analyze_split_classes.py --dataset pavia --split checker --block 64
"""
import argparse

import numpy as np
import scipy.io as sio

from dataset import DATASET_INFO

CLASS_NAMES = {
    "indian_pines": ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
                     'Grass-trees', 'Grass-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill',
                     'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods', 'Buildings',
                     'Stone-Towers'],
    "pavia": ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal',
              'Bare Soil', 'Bitumen', 'Bricks', 'Shadows'],
}


def region_masks(gt_shape, split, train_ratio, block, off=(0, 0)):
    H, W = gt_shape
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    if split == "lr":
        cut = int(train_ratio * W)
        train = cols < cut
        test = cols >= cut
    elif split == "checker":
        br = (rows - off[0]) // block          # numpy floor division
        bc = (cols - off[1]) // block
        parity = (br + bc) % 2
        train = parity == 0
        test = parity == 1
    else:
        raise ValueError(f"unknown split: {split}")
    return np.broadcast_to(train, gt_shape).copy(), np.broadcast_to(test, gt_shape).copy()


def class_counts(gt, train_m, test_m):
    """per-class (train_px, test_px) lists + summary dict"""
    n_cls = int(gt.max())
    rows = []
    missing_tr, missing_te = [], []
    n_both = 0
    for c in range(n_cls):
        m = (gt == c + 1)
        tr = int((m & train_m).sum())
        te = int((m & test_m).sum())
        rows.append((c + 1, tr, te))
        if tr + te == 0:
            continue
        if tr == 0:
            missing_te.append(c + 1)     # absent from train -> TEST-ONLY
        elif te == 0:
            missing_tr.append(c + 1)     # absent from test -> TRAIN-ONLY
        else:
            n_both += 1
    return rows, {"both": n_both, "train_only": missing_tr, "test_only": missing_te}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["indian_pines", "pavia"])
    ap.add_argument("--split", required=True, choices=["lr", "checker"])
    ap.add_argument("--train_ratio", type=float, default=0.6)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--off_r", type=int, default=0)
    ap.add_argument("--off_c", type=int, default=0)
    ap.add_argument("--scan", action="store_true",
                    help="scan all grid offsets (0 / block//2) and print a "
                         "compact summary line per variant")
    ap.add_argument("--data_root", default="/mnt/scratch/znzs0468/data")
    args = ap.parse_args()

    info = DATASET_INFO[args.dataset]
    gt = sio.loadmat(f"{args.data_root}/{info['gt_file']}")[info["gt_key"]].astype(np.int64)
    names = CLASS_NAMES[args.dataset]

    if args.scan:
        print(f"[{args.dataset}] split={args.split} block={args.block} — offset scan (step 8)")
        best = None
        for off_r in range(0, args.block, 8):
            for off_c in range(0, args.block, 8):
                train_m, test_m = region_masks(
                    gt.shape, args.split, args.train_ratio, args.block, (off_r, off_c))
                rows, s = class_counts(gt, train_m, test_m)
                tpx = int(((gt > 0) & train_m).sum())
                # priority: 0 test-only, then max both, then min train-only, then balance
                key = (len(s["test_only"]), -s["both"],
                       len(s["train_only"]), abs(tpx - int(((gt > 0) & test_m).sum())))
                if best is None or key < best[0]:
                    best = (key, (off_r, off_c), s)
                if len(s["test_only"]) <= 1:
                    print(f"  off=({off_r:>3},{off_c:>3}): both={s['both']:>2}/{len(names)} "
                          f"train-only={len(s['train_only'])} test-only={len(s['test_only'])} "
                          f"train_px={tpx}")
        if best is None:
            print("  (no variant)")
            return
        _, off, s = best
        print(f"\nBEST: off=({off[0]},{off[1]}) -> both={s['both']}/{len(names)}, "
              f"train-only={[names[c-1] for c in s['train_only']] or '-'}, "
              f"test-only={[names[c-1] for c in s['test_only']] or '-'}")
        return

    train_m, test_m = region_masks(
        gt.shape, args.split, args.train_ratio, args.block, (args.off_r, args.off_c))

    n_cls = len(names)
    print(f"[{args.dataset}] split={args.split} block={args.block} "
          f"off=({args.off_r},{args.off_c}) train_ratio={args.train_ratio}")
    print(f"{'class':<16}{'train px':>10}{'test px':>10}{'train%':>8}  status")
    missing_tr, missing_te, n_both = [], [], 0
    for c in range(n_cls):
        m = (gt == c + 1)
        tr = int((m & train_m).sum())
        te = int((m & test_m).sum())
        tot = tr + te
        if tot == 0:
            continue
        if tr == 0:
            status = "TEST-ONLY  <--"
            missing_te.append(names[c])
        elif te == 0:
            status = "TRAIN-ONLY <--"
            missing_tr.append(names[c])
        else:
            status = f"both (train {100 * tr / tot:.0f}%)"
            n_both += 1
        print(f"{names[c]:<16}{tr:>10}{te:>10}{100 * tr / tot:>7.1f}%  {status}")
    print(f"\nsummary: {n_both}/{n_cls} classes in both splits; "
          f"train-only={missing_tr or '-'}; test-only={missing_te or '-'}")
    print(f"labelled px: train={int(((gt > 0) & train_m).sum())} "
          f"test={int(((gt > 0) & test_m).sum())}")


if __name__ == "__main__":
    main()
