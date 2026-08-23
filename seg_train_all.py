"""
seg_train_all.py — unified training for all segmentation models

Usage:
  python seg_train_all.py --dataset indian_pines --model vit
  python seg_train_all.py --dataset indian_pines --model spectralformer
  python seg_train_all.py --dataset indian_pines --model cnn3d
  python seg_train_all.py --dataset indian_pines --model disentangle
  python seg_train_all.py --dataset indian_pines --model disentangle --use_cma   # ours
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seg_dataset import make_seg_splits, HSISegDataset
from seg_model import DisentangleFormerSeg
from baselines import MODELS

IGNORE = HSISegDataset.IGNORE


@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu().numpy().ravel()
        gt = y.numpy().ravel()
        m = gt != IGNORE
        pred, gt = pred[m], gt[m]
        conf += np.bincount(num_classes * gt + pred,
                            minlength=num_classes ** 2).reshape(num_classes, num_classes)
    tp = np.diag(conf)
    oa = tp.sum() / conf.sum().clip(min=1)
    iou = tp / (conf.sum(0) + conf.sum(1) - tp).clip(min=1)
    acc = tp / conf.sum(1).clip(min=1)
    return oa, np.nanmean(iou), np.nanmean(acc), iou


def build_model(name, n_bands, n_cls, args):
    if name == "disentangle":
        return DisentangleFormerSeg(n_bands, n_cls, args.patch,
                                    dim=args.dim, depth=args.depth,
                                    heads=args.heads, use_cma=args.use_cma)
    cls = MODELS[name]
    return cls(n_bands, n_cls, args.patch,
               dim=args.dim, depth=args.depth, heads=args.heads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="indian_pines",
                    choices=["indian_pines", "pavia", "houston"])
    ap.add_argument("--model", default="disentangle",
                    choices=["disentangle", "vit", "spectralformer", "cnn3d"])
    ap.add_argument("--data_root", default="/mnt/scratch/znzs0468/data")
    ap.add_argument("--out_dir", default="/mnt/scratch/znzs0468/results")
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--train_ratio", type=float, default=0.6)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--use_cma", action="store_true")
    ap.add_argument("--split", default="random",
                    choices=["random", "spatial", "checker"],
                    help="spatial = left/right region split; "
                         "checker = checkerboard cells (offsets per dataset in "
                         "seg_dataset.CHECKER_OFFSETS); both leak-free")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mname = args.model
    if mname == "disentangle":
        mname = "ours_cma" if args.use_cma else "disentangle"
    sp = {"spatial": "_sp", "checker": "_ch"}.get(args.split, "")
    tag = f"seg_{args.dataset}_{mname}_r{int(args.train_ratio*100)}_s{args.seed}{sp}"
    os.makedirs(args.out_dir, exist_ok=True)

    train_set, test_set, n_cls, n_bands = make_seg_splits(
        args.dataset, args.data_root, args.patch, args.stride,
        args.train_ratio, args.seed, args.split)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(args.model, n_bands, n_cls, args).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{tag}] train={len(train_set)} test={len(test_set)} "
          f"bands={n_bands} classes={n_cls} params={n_params:.2f}M", flush=True)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_miou, ckpt = 0.0, f"{args.out_dir}/{tag}_best.pth"
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
        scheduler.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            oa, miou, macc, _ = evaluate(model, test_loader, n_cls, device)
            print(f"epoch {epoch:3d} | loss {total/len(train_set):.4f} "
                  f"| OA {oa:.4f} mIoU {miou:.4f} mAcc {macc:.4f}", flush=True)
            if miou > best_miou:
                best_miou = miou
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "oa": oa, "miou": miou, "macc": macc,
                            "params_M": n_params, "args": vars(args)}, ckpt)

    ck = torch.load(ckpt, weights_only=False)
    model.load_state_dict(ck["model"])
    oa, miou, macc, iou = evaluate(model, test_loader, n_cls, device)
    print("=" * 60, flush=True)
    print(f"FINAL [{tag}]  OA={oa:.4f}  mIoU={miou:.4f}  mAcc={macc:.4f}  "
          f"params={n_params:.2f}M  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
