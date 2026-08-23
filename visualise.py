"""
visualise.py — qualitative comparison of segmentation results

Produces a figure with 4 panels per dataset:
    False-colour RGB | Ground Truth | Baseline prediction | Ours (CMA) prediction
plus a difference map highlighting where the two models disagree with the GT.

Usage:
  python visualise.py --dataset indian_pines
  python visualise.py --dataset pavia
"""
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from seg_dataset import load_and_normalise
from seg_model import DisentangleFormerSeg

CLASS_NAMES = {
    "indian_pines": ['Alfalfa','Corn-notill','Corn-mintill','Corn','Grass-pasture',
                     'Grass-trees','Grass-mowed','Hay-windrowed','Oats','Soybean-notill',
                     'Soybean-mintill','Soybean-clean','Wheat','Woods','Buildings','Stone-Towers'],
    "pavia": ['Asphalt','Meadows','Gravel','Trees','Painted metal',
              'Bare Soil','Bitumen','Bricks','Shadows'],
}
RGB_BANDS = {"indian_pines": (29, 19, 9), "pavia": (55, 35, 15)}


def make_rgb(data, bands):
    img = data[:, :, list(bands)].astype(float)
    for i in range(3):
        lo, hi = np.percentile(img[:, :, i], 2), np.percentile(img[:, :, i], 98)
        img[:, :, i] = np.clip((img[:, :, i] - lo) / (hi - lo + 1e-8), 0, 1)
    return img


@torch.no_grad()
def predict_full_scene(model, data, device, patch=32, stride=16):
    """Sliding-window inference over the whole scene, averaging overlaps."""
    H, W, C = data.shape
    model.eval()
    # logits accumulator
    n_cls = model.decoder.classifier.out_channels
    acc = np.zeros((n_cls, H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    positions = []
    for i in range(0, max(H - patch, 0) + 1, stride):
        for j in range(0, max(W - patch, 0) + 1, stride):
            positions.append((i, j))
    if (H - patch) % stride != 0:
        positions += [(H - patch, j) for j in range(0, max(W - patch, 0) + 1, stride)]
    if (W - patch) % stride != 0:
        positions += [(i, W - patch) for i in range(0, max(H - patch, 0) + 1, stride)]

    for i, j in positions:
        tile = data[i:i + patch, j:j + patch, :]
        x = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(device)
        logits = model(x)[0].cpu().numpy()      # (n_cls, patch, patch)
        acc[:, i:i + patch, j:j + patch] += logits
        cnt[i:i + patch, j:j + patch] += 1

    cnt = np.maximum(cnt, 1)
    return (acc / cnt).argmax(0)                # (H, W) predicted class ids


def load_model(ckpt_path, n_bands, n_cls, use_cma, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    model = DisentangleFormerSeg(n_bands, n_cls, a["patch"],
                                 dim=a["dim"], depth=a["depth"],
                                 heads=a["heads"], use_cma=use_cma).to(device)
    model.load_state_dict(ck["model"])
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="indian_pines", choices=["indian_pines", "pavia"])
    ap.add_argument("--data_root", default="/mnt/scratch/znzs0468/data")
    ap.add_argument("--ckpt_dir", default="/mnt/scratch/znzs0468/results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/mnt/scratch/znzs0468/results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, gt, n_cls = load_and_normalise(args.dataset, args.data_root)
    n_bands = data.shape[-1]
    names = CLASS_NAMES[args.dataset]

    base_ck = f"{args.ckpt_dir}/seg_{args.dataset}_disentangle_s{args.seed}_best.pth"
    cma_ck = f"{args.ckpt_dir}/seg_{args.dataset}_ours_cma_s{args.seed}_best.pth"
    for p in (base_ck, cma_ck):
        if not os.path.exists(p):
            raise SystemExit(f"checkpoint not found: {p}")

    m_base, ck_b = load_model(base_ck, n_bands, n_cls, False, device)
    m_cma, ck_c = load_model(cma_ck, n_bands, n_cls, True, device)

    print("running full-scene inference ...", flush=True)
    pred_base = predict_full_scene(m_base, data, device)
    pred_cma = predict_full_scene(m_cma, data, device)

    # mask out unlabelled background for display
    mask = gt > 0
    gt0 = gt - 1                                  # 0-indexed
    disp_base = np.where(mask, pred_base, -1)
    disp_cma = np.where(mask, pred_cma, -1)
    disp_gt = np.where(mask, gt0, -1)

    # colour map: index -1 -> white background
    cmap_colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_cls]
    cmap = ListedColormap(np.vstack([[1, 1, 1, 1], cmap_colors]))

    def show(ax, arr, title):
        ax.imshow(arr + 1, cmap=cmap, vmin=0, vmax=n_cls, interpolation="nearest")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2))
    title = "Indian Pines" if args.dataset == "indian_pines" else "Pavia University"
    fig.suptitle(f"{title} — Qualitative Segmentation Comparison (seed {args.seed})",
                 fontsize=15, fontweight="bold", y=1.02)

    axes[0].imshow(make_rgb(data, RGB_BANDS[args.dataset]))
    axes[0].set_title("False-colour RGB", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    show(axes[1], disp_gt, "Ground Truth")
    show(axes[2], disp_base, f"Baseline (static STE)\nmIoU {ck_b['miou']*100:.1f}%")
    show(axes[3], disp_cma, f"Ours (Gated CMA)\nmIoU {ck_c['miou']*100:.1f}%")

    # error map: red = baseline wrong & CMA right, blue = opposite, grey = both wrong
    err = np.zeros((*gt.shape, 3), dtype=float) + 1.0
    b_ok = (pred_base == gt0) & mask
    c_ok = (pred_cma == gt0) & mask
    err[mask & c_ok & ~b_ok] = [0.13, 0.70, 0.55]      # CMA fixes  (teal)
    err[mask & b_ok & ~c_ok] = [0.85, 0.30, 0.25]      # CMA breaks (red)
    err[mask & ~b_ok & ~c_ok] = [0.75, 0.75, 0.75]     # both wrong (grey)
    axes[4].imshow(err, interpolation="nearest")
    n_fix = int((mask & c_ok & ~b_ok).sum())
    n_brk = int((mask & b_ok & ~c_ok).sum())
    axes[4].set_title(f"Difference map\nCMA fixes {n_fix} px / breaks {n_brk} px",
                      fontsize=11, fontweight="bold")
    axes[4].axis("off")
    axes[4].legend(handles=[
        mpatches.Patch(color=(0.13, 0.70, 0.55), label="CMA correct, baseline wrong"),
        mpatches.Patch(color=(0.85, 0.30, 0.25), label="baseline correct, CMA wrong"),
        mpatches.Patch(color=(0.75, 0.75, 0.75), label="both wrong"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.28), fontsize=8, frameon=False)

    plt.tight_layout()
    out_png = f"{args.out}/qualitative_{args.dataset}_s{args.seed}.png"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print("saved:", out_png, flush=True)

    # separate legend figure (class colours)
    fig2, ax2 = plt.subplots(figsize=(10, 1.2 + 0.22 * n_cls))
    ax2.axis("off")
    handles = [mpatches.Patch(color=cmap_colors[i], label=names[i]) for i in range(n_cls)]
    ax2.legend(handles=handles, ncol=4, loc="center", fontsize=9, frameon=False)
    out_leg = f"{args.out}/legend_{args.dataset}.png"
    plt.savefig(out_leg, dpi=160, bbox_inches="tight")
    print("saved:", out_leg, flush=True)

    print(f"\nSummary: CMA corrects {n_fix} pixels that the baseline got wrong, "
          f"and breaks {n_brk} that the baseline got right "
          f"(net {n_fix - n_brk:+d}).", flush=True)


if __name__ == "__main__":
    main()
