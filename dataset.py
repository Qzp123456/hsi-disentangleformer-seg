"""
dataset.py — HSI patch dataset for Indian Pines / Pavia University
"""
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


DATASET_INFO = {
    "indian_pines": {
        "data_file": "Indian_pines_corrected.mat",
        "data_key": "indian_pines_corrected",
        "gt_file": "Indian_pines_gt.mat",
        "gt_key": "indian_pines_gt",
        "num_classes": 16,
    },
    "pavia": {
        "data_file": "PaviaU.mat",
        "data_key": "paviaU",
        "gt_file": "PaviaU_gt.mat",
        "gt_key": "paviaU_gt",
        "num_classes": 9,
    },
    "houston": {
        "data_file": "Houston.mat",
        "data_key": "Houston",
        "gt_file": "Houston_gt.mat",
        "gt_key": "Houston_gt",
        "num_classes": 15,
    },
}


def load_hsi(name, data_root):
    info = DATASET_INFO[name]
    data = sio.loadmat(f"{data_root}/{info['data_file']}")[info["data_key"]].astype(np.float32)
    gt = sio.loadmat(f"{data_root}/{info['gt_file']}")[info["gt_key"]].astype(np.int64)
    return data, gt, info["num_classes"]


def build_patches(data, gt, patch_size=7):
    """StandardScaler per band -> reflect padding -> extract P×P patches
    around every labelled pixel."""
    H, W, C = data.shape
    pad = patch_size // 2

    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, C)).reshape(H, W, C)
    data = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    coords = np.argwhere(gt > 0)               # labelled pixels only
    patches = np.empty((len(coords), patch_size, patch_size, C), dtype=np.float32)
    labels = np.empty(len(coords), dtype=np.int64)
    for k, (i, j) in enumerate(coords):
        patches[k] = data[i:i + patch_size, j:j + patch_size, :]
        labels[k] = gt[i, j] - 1                # 0-indexed
    return patches, labels, coords


class HSIPatchDataset(Dataset):
    def __init__(self, patches, labels):
        # (N, P, P, C) -> (N, C, P, P) for PyTorch conv layout
        self.x = torch.from_numpy(patches).permute(0, 3, 1, 2).contiguous()
        self.y = torch.from_numpy(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_splits(name, data_root, patch_size=7, train_ratio=0.1, seed=42):
    data, gt, num_classes = load_hsi(name, data_root)
    patches, labels, _ = build_patches(data, gt, patch_size)
    x_tr, x_te, y_tr, y_te = train_test_split(
        patches, labels, train_size=train_ratio,
        random_state=seed, stratify=labels)
    n_bands = patches.shape[-1]
    return (HSIPatchDataset(x_tr, y_tr),
            HSIPatchDataset(x_te, y_te),
            num_classes, n_bands)
