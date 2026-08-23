"""
seg_dataset.py — Semantic segmentation dataset for HSI

Difference from patch classification:
  - classification: one 7x7 patch  -> one label   (the centre pixel)
  - segmentation:   one big patch   -> a label MAP (every pixel labelled)

We tile the whole scene into overlapping large patches (e.g. 32x32),
each patch keeps its full label map. Pixels with label 0 (unlabelled)
are ignored in the loss via ignore_index.
"""
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from dataset import DATASET_INFO   # reuse dataset registry


def load_and_normalise(name, data_root):
    info = DATASET_INFO[name]
    data = sio.loadmat(f"{data_root}/{info['data_file']}")[info["data_key"]].astype(np.float32)
    gt = sio.loadmat(f"{data_root}/{info['gt_file']}")[info["gt_key"]].astype(np.int64)
    H, W, C = data.shape
    data = StandardScaler().fit_transform(data.reshape(-1, C)).reshape(H, W, C)
    return data, gt, info["num_classes"]


def tile_scene(data, gt, patch=32, stride=16):
    """Cut the scene into overlapping (patch x patch) tiles.

    Returns (tiles_x, tiles_y, origins) where origins[k] = (i, j) is the
    top-left corner of tile k, used by spatial (non-leaky) train/test splits."""
    H, W, C = data.shape
    tiles_x, tiles_y, origins = [], [], []
    for i in range(0, max(H - patch, 0) + 1, stride):
        for j in range(0, max(W - patch, 0) + 1, stride):
            tiles_x.append(data[i:i + patch, j:j + patch, :])
            tiles_y.append(gt[i:i + patch, j:j + patch])
            origins.append((i, j))
    # ensure the bottom / right edges are covered
    if (H - patch) % stride != 0:
        for j in range(0, max(W - patch, 0) + 1, stride):
            tiles_x.append(data[H - patch:H, j:j + patch, :])
            tiles_y.append(gt[H - patch:H, j:j + patch])
            origins.append((H - patch, j))
    if (W - patch) % stride != 0:
        for i in range(0, max(H - patch, 0) + 1, stride):
            tiles_x.append(data[i:i + patch, W - patch:W, :])
            tiles_y.append(gt[i:i + patch, W - patch:W])
            origins.append((i, W - patch))
    return np.stack(tiles_x), np.stack(tiles_y), np.array(origins)


class HSISegDataset(Dataset):
    """label - 1 so classes become 0..K-1; unlabelled (was 0) -> ignore_index."""
    IGNORE = 255

    def __init__(self, tiles_x, tiles_y, augment=False):
        # (N,P,P,C) -> (N,C,P,P)
        self.x = torch.from_numpy(tiles_x).permute(0, 3, 1, 2).contiguous().float()
        y = tiles_y.copy()
        y = y - 1                       # 0-index labelled classes
        y[y < 0] = self.IGNORE          # unlabelled -> ignore
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.augment:
            if torch.rand(1) < 0.5:                      # h-flip
                x, y = torch.flip(x, [2]), torch.flip(y, [1])
            if torch.rand(1) < 0.5:                      # v-flip
                x, y = torch.flip(x, [1]), torch.flip(y, [0])
            k = int(torch.randint(0, 4, (1,)))           # rot90
            if k:
                x, y = torch.rot90(x, k, [1, 2]), torch.rot90(y, k, [0, 1])
        return x, y


# Checkerboard cell config, chosen with analyze_split_classes.py --scan
# (priority: 0 test-only classes, then max classes in both splits, then balance):
#   indian_pines: block 56, off (8, 16) — 0 test-only, 13/16 classes in both,
#                 train-only = Alfalfa/Grass-mowed/Stone-Towers (all < 100 px);
#                 stride 8 because 56 % 16 != 0 -> 16px tiles would leave
#                 uncovered gaps inside cells
#   pavia:        block 64, off (0, 32)  — 9/9 in both, balanced train/test px
CHECKER_CONFIG = {
    "indian_pines": {"block": 56, "off": (8, 16), "stride": 8},
    "pavia": {"block": 64, "off": (0, 32), "stride": 16},
}


def make_seg_splits(name, data_root, patch=32, stride=16,
                    train_ratio=0.6, seed=42, split_mode="random",
                    checker_off=None):
    """split_mode="random": permute tiles (old protocol — overlapping tiles
    leak across train/test).

    split_mode="spatial": left train_ratio of the scene width is train, the
    rest is test; tiles straddling the vertical boundary are dropped so no
    pixel is seen in both splits (no leakage).

    split_mode="checker": checkerboard cells (size/offset/stride per dataset
    in CHECKER_CONFIG) alternate train/test. Tiles must lie fully inside a
    single cell — boundary tiles are dropped — so train and test tiles never
    share a pixel, but classes that spread over several cells appear in both
    splits."""
    data, gt, num_classes = load_and_normalise(name, data_root)

    if split_mode == "checker":
        cfg = CHECKER_CONFIG.get(name, {"block": 2 * patch, "off": (0, 0),
                                        "stride": stride})
        block = cfg["block"]
        off = tuple(checker_off) if checker_off is not None else cfg["off"]
        sub = cfg["stride"]
        H, W = gt.shape
        xs, ys, tr, te = [], [], [], []

        def consider(i, j):
            xs.append(data[i:i + patch, j:j + patch, :])
            ys.append(gt[i:i + patch, j:j + patch])
            if (i - off[0]) % block + patch <= block and \
               (j - off[1]) % block + patch <= block:
                bi, bj = (i - off[0]) // block, (j - off[1]) // block
                (tr if (bi + bj) % 2 == 0 else te).append(len(xs) - 1)

        for i in range(0, max(H - patch, 0) + 1, sub):
            for j in range(0, max(W - patch, 0) + 1, sub):
                consider(i, j)
        if (H - patch) % sub != 0:              # bottom edge rows
            for j in range(0, max(W - patch, 0) + 1, sub):
                consider(H - patch, j)
        if (W - patch) % sub != 0:              # right edge columns
            for i in range(0, max(H - patch, 0) + 1, sub):
                consider(i, W - patch)

        tiles_x, tiles_y = np.stack(xs), np.stack(ys)
        tr, te = np.array(tr), np.array(te)
        print(f"[checker split] block={block}px off={off} stride={sub}, "
              f"train={len(tr)} test={len(te)} tiles, "
              f"{len(xs) - len(tr) - len(te)} boundary tiles dropped", flush=True)
    else:
        tiles_x, tiles_y, origins = tile_scene(data, gt, patch, stride)
        if split_mode == "spatial":
            W = gt.shape[1]
            cut = train_ratio * W
            tr = np.array([k for k, (i, j) in enumerate(origins) if j + patch <= cut])
            te = np.array([k for k, (i, j) in enumerate(origins) if j >= cut])
            print(f"[spatial split] train={len(tr)} test={len(te)} tiles, "
                  f"{len(origins) - len(tr) - len(te)} straddling tiles dropped "
                  f"(cut={cut:.1f}px)", flush=True)
        elif split_mode == "random":
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(tiles_x))
            n_tr = int(len(idx) * train_ratio)
            tr, te = idx[:n_tr], idx[n_tr:]
        else:
            raise ValueError(f"unknown split_mode: {split_mode}")
    if len(tr) == 0 or len(te) == 0:
        raise ValueError(f"split_mode={split_mode} left an empty split "
                         f"(train={len(tr)}, test={len(te)}), check train_ratio")

    n_bands = data.shape[-1]
    return (HSISegDataset(tiles_x[tr], tiles_y[tr], augment=True),
            HSISegDataset(tiles_x[te], tiles_y[te], augment=False),
            num_classes, n_bands)
