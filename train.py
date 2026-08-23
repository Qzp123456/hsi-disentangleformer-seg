"""
train.py — training / evaluation for DisentangleFormer on HSI benchmarks

Usage:
  python train.py --dataset indian_pines --epochs 100
  python train.py --dataset pavia --epochs 100 --use_cma      # our method
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from dataset import make_splits
from model import DisentangleFormer


def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.append(out.argmax(1).cpu().numpy())
            gts.append(y.numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    oa = (preds == gts).mean()
    cm = confusion_matrix(gts, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    aa = per_class_acc.mean()
    kappa = cohen_kappa_score(gts, preds)
    return oa, aa, kappa, per_class_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="indian_pines",
                    choices=["indian_pines", "pavia"])
    ap.add_argument("--data_root", default="/mnt/scratch/znzs0468/data")
    ap.add_argument("--out_dir", default="/mnt/scratch/znzs0468/results")
    ap.add_argument("--patch_size", type=int, default=7)
    ap.add_argument("--train_ratio", type=float, default=0.1)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--use_cma", action="store_true",
                    help="enable our CMA cross-stream fusion")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{args.dataset}_{'cma' if args.use_cma else 'base'}_s{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)

    # ── data ─────────────────────────────────────────────
    train_set, test_set, num_classes, n_bands = make_splits(
        args.dataset, args.data_root, args.patch_size,
        args.train_ratio, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256,
                             shuffle=False, num_workers=2, pin_memory=True)
    print(f"[{tag}] train={len(train_set)} test={len(test_set)} "
          f"bands={n_bands} classes={num_classes}", flush=True)

    # ── model ────────────────────────────────────────────
    model = DisentangleFormer(
        n_bands, num_classes, args.patch_size,
        dim=args.dim, depth=args.depth, heads=args.heads,
        use_cma=args.use_cma).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.2f} M", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ── train loop ───────────────────────────────────────
    best_oa = 0.0
    ckpt_path = f"{args.out_dir}/{tag}_best.pth"
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        scheduler.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            oa, aa, kappa, _ = evaluate(model, test_loader, device)
            print(f"epoch {epoch:3d} | loss {total_loss/len(train_set):.4f} "
                  f"| OA {oa:.4f} AA {aa:.4f} Kappa {kappa:.4f}", flush=True)
            if oa > best_oa:
                best_oa = oa
                torch.save({"epoch": epoch,
                            "model": model.state_dict(),
                            "oa": oa, "aa": aa, "kappa": kappa,
                            "args": vars(args)}, ckpt_path)

    # ── final report ─────────────────────────────────────
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    oa, aa, kappa, per_class = evaluate(model, test_loader, device)
    mins = (time.time() - t0) / 60
    print("=" * 55, flush=True)
    print(f"FINAL [{tag}]  OA={oa:.4f}  AA={aa:.4f}  Kappa={kappa:.4f}  "
          f"({mins:.1f} min)", flush=True)
    print("Per-class accuracy:", flush=True)
    for i, acc in enumerate(per_class):
        print(f"  class {i:2d}: {acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
