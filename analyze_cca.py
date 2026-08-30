"""
analyze_cca.py — disentanglement analysis of the ST and CT streams via CCA

For each checkpoint, hook the st_encoder / ct_encoder / (optional) cma modules of
the DisentangleFormerBlock, run a forward pass over the whole test set, and collect
paired token-level features:

  stage "pre"  : r_s, r_c output by the ST/CT encoders respectively (after Eq.3/Eq.4,
                 before fusion; for the baseline (disentangle, no CMA) model these are
                 the final r_s/r_c that go into the concatenation)
  stage "post" : ours_cma model only — r_s, r_c after the CMAFusion

r_s / r_c are paired token-for-token at the same token positions within one forward
pass and one batch, and CCA is used to measure the canonical correlation between the
two streams (0~1; larger = more entangled, smaller = more disentangled).

Comparisons:
  1) baseline "pre" (=final) vs ours_cma "post" — across models (different trained weights)
  2) ours_cma "pre" vs ours_cma "post"          — within a single forward pass, separated
     only by the CMA operator, directly measuring the causal effect of the CMA step on
     correlation while controlling for the trained-weight difference

If the "post" correlation is systematically higher than "pre" / higher than baseline,
this indicates CMA re-couples two streams that are by design meant to be independent,
contradicting the disentanglement principle and explaining why CMA's mIoU gain is limited.
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import torch
from sklearn.cross_decomposition import CCA

from seg_dataset import make_seg_splits
from seg_model import DisentangleFormerSeg


# ──────────────────────────────────────────────────────────────
# CCA scoring
# ──────────────────────────────────────────────────────────────
def cca_score(X, Y, n_components=8, max_samples=8000, seed=0):
    n = X.shape[0]
    if n > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_samples, replace=False)
        X, Y = X[idx], Y[idx]
    k = min(n_components, X.shape[1], Y.shape[1])
    cca = CCA(n_components=k, max_iter=1000)
    Xc, Yc = cca.fit_transform(X, Y)
    corrs = []
    for i in range(k):
        c = np.corrcoef(Xc[:, i], Yc[:, i])[0, 1]
        corrs.append(abs(c) if np.isfinite(c) else 0.0)
    return float(np.mean(corrs)), float(np.max(corrs))


def cca_null_score(X, Y, n_components=8, max_samples=8000, seed=0):
    """Break the token-pairing (shuffle Y independently of X) and refit CCA.
    With finite n and d=64, CCA is upward-biased even under H0 (no true
    relationship) because it actively searches for the best-correlated
    projection. This gives the noise floor to subtract off, so the CMA
    pre/post delta isn't mistaken for a fitting artefact."""
    rng = np.random.RandomState(seed + 999)
    perm = rng.permutation(Y.shape[0])
    return cca_score(X, Y[perm], n_components, max_samples, seed)


# ──────────────────────────────────────────────────────────────
# forward-hook feature collector (pairs r_s / r_c token-for-token)
# ──────────────────────────────────────────────────────────────
class FeatureCollector:
    def __init__(self, model, use_cma, per_batch_samples=300, seed=0):
        self.use_cma = use_cma
        self.per_batch_samples = per_batch_samples
        self.rng = np.random.RandomState(seed)
        self.data = defaultdict(lambda: {"s": [], "c": []})   # (layer, stage) -> lists
        self._cur_idx = {}                                    # layer -> idx array (shared s/c/post within one batch)
        self.handles = []

        for li, blk in enumerate(model.blocks):
            self.handles.append(blk.st_encoder.register_forward_hook(self._st_hook(li)))
            self.handles.append(blk.ct_encoder.register_forward_hook(self._ct_hook(li)))
            if use_cma:
                self.handles.append(blk.cma.register_forward_hook(self._cma_hook(li)))

    def _idx_for(self, li, n):
        k = min(self.per_batch_samples, n)
        idx = self.rng.choice(n, k, replace=False)
        self._cur_idx[li] = idx
        return idx

    def _st_hook(self, li):
        def hook(module, inp, out):
            flat = out.detach().reshape(-1, out.shape[-1]).cpu().numpy()
            idx = self._idx_for(li, flat.shape[0])
            self.data[(li, "pre")]["s"].append(flat[idx])
        return hook

    def _ct_hook(self, li):
        def hook(module, inp, out):
            out = out.transpose(1, 2)                          # (B*nW,C,N) -> (B*nW,N,C), same token order as r_s
            flat = out.detach().reshape(-1, out.shape[-1]).cpu().numpy()
            idx = self._cur_idx[li]                             # reuse SAME indices as st_hook this batch -> paired tokens
            self.data[(li, "pre")]["c"].append(flat[idx])
        return hook

    def _cma_hook(self, li):
        def hook(module, inp, out):
            z_s, z_c = out
            idx = self._cur_idx[li]                             # token order unchanged by attention -> same idx still valid
            fs = z_s.detach().reshape(-1, z_s.shape[-1]).cpu().numpy()
            fc = z_c.detach().reshape(-1, z_c.shape[-1]).cpu().numpy()
            self.data[(li, "post")]["s"].append(fs[idx])
            self.data[(li, "post")]["c"].append(fc[idx])
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    def finalize(self):
        out = {}
        for key, d in self.data.items():
            if d["s"] and d["c"]:
                out[key] = (np.concatenate(d["s"], 0), np.concatenate(d["c"], 0))
        return out


# ──────────────────────────────────────────────────────────────
def parse_tag(fname):
    """seg_indian_pines_disentangle_s42_best.pth -> (dataset, model, seed)
       seg_pavia_ours_cma_s42_best.pth -> (dataset, model, seed)"""
    base = os.path.basename(fname)
    m = re.match(r"seg_(indian_pines|pavia)_(disentangle|ours_cma)_s(\d+)_best\.pth$", base)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)


def run_one(path, dataset, data_root, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    targs = ck.get("args", {}) or {}
    use_cma = bool(targs.get("use_cma", "ours_cma" in os.path.basename(path)))
    patch = targs.get("patch", 32)
    stride = targs.get("stride", 16)
    train_ratio = targs.get("train_ratio", 0.6)
    seed = int(targs.get("seed", 42))
    dim = targs.get("dim", 64)
    depth = targs.get("depth", 2)
    heads = targs.get("heads", 4)

    train_set, test_set, n_cls, n_bands = make_seg_splits(
        dataset, data_root, patch, stride, train_ratio, seed)

    model = DisentangleFormerSeg(n_bands, n_cls, patch=patch,
                                 dim=dim, depth=depth,
                                 heads=heads, use_cma=use_cma)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    collector = FeatureCollector(model, use_cma, per_batch_samples=4000, seed=seed)
    loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device))
    collector.remove()
    feats = collector.finalize()

    results = {}
    for (li, stage), (X, Y) in feats.items():
        mean_corr, max_corr = cca_score(X, Y, seed=seed)
        null_mean, _ = cca_null_score(X, Y, seed=seed)
        results[(li, stage)] = (mean_corr, max_corr, null_mean, X.shape[0])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="/mnt/scratch/znzs0468/results")
    ap.add_argument("--data_root", default="/mnt/scratch/znzs0468/data")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    pattern = os.path.join(args.results_dir, "seg_*_s*_best.pth")
    paths = sorted(glob.glob(pattern))
    paths = [p for p in paths if re.match(
        r"seg_(indian_pines|pavia)_(disentangle|ours_cma)_s\d+_best\.pth$", os.path.basename(p))]

    print(f"Found {len(paths)} checkpoints (disentangle baseline + ours_cma, main-experiment seed set)\n")

    # agg[dataset][model][(layer,stage)] -> list of (mean_corr, max_corr)
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for i, p in enumerate(paths):
        dataset, mname, seed = parse_tag(p)
        if dataset is None:
            continue
        try:
            res = run_one(p, dataset, args.data_root, device)
        except Exception as e:
            print(f"[SKIP] {os.path.basename(p)}: {e}")
            continue
        for (li, stage), (mean_c, max_c, null_m, n) in res.items():
            agg[dataset][mname][(li, stage)].append((mean_c, max_c, null_m))
        print(f"[{i+1}/{len(paths)}] {os.path.basename(p)} done "
              f"({', '.join(f'{li}-{st}:{mc:.3f}' for (li, st), (mc, *_) in res.items())})")

    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== Summary: mean canonical corr (mean±std over seeds) ===")
    print("=" * 70)
    for dataset in sorted(agg.keys()):
        print(f"\n[{dataset}]")
        for mname in sorted(agg[dataset].keys()):
            print(f"  -- {mname} --")
            for key in sorted(agg[dataset][mname].keys()):
                vals = [m for m, _, _ in agg[dataset][mname][key]]
                nulls = [nm for _, _, nm in agg[dataset][mname][key]]
                mean = np.mean(vals)
                std = np.std(vals)
                li, stage = key
                print(f"    layer {li} [{stage}]: {mean:.4f} ± {std:.4f}  "
                      f"null(H0)={np.mean(nulls):.4f}  (n={len(vals)})")

    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== Comparison 1: baseline(pre=final) vs ours_cma(post) — across models ===")
    print("=" * 70)
    for dataset in sorted(agg.keys()):
        base = agg[dataset].get("disentangle", {})
        cma = agg[dataset].get("ours_cma", {})
        n_layers = max((li for (li, _) in list(base.keys()) + list(cma.keys())), default=-1) + 1
        print(f"\n[{dataset}]")
        for li in range(n_layers):
            b_vals = [m for m, _, _ in base.get((li, "pre"), [])]
            b_nulls = [nm for _, _, nm in base.get((li, "pre"), [])]
            c_vals = [m for m, _, _ in cma.get((li, "post"), [])]
            c_nulls = [nm for _, _, nm in cma.get((li, "post"), [])]
            if not b_vals or not c_vals:
                continue
            bm, cm = np.mean(b_vals), np.mean(c_vals)
            bn, cn = np.mean(b_nulls), np.mean(c_nulls)
            delta = cm - bm
            print(f"  layer {li}: baseline={bm:.4f} (null={bn:.4f})  "
                  f"ours_cma(post)={cm:.4f} (null={cn:.4f})  "
                  f"delta={delta:+.4f}{'  <-- CMA increased correlation (more entangled)' if delta > 0.02 else ''}")

    print("\n" + "=" * 70)
    print("=== Comparison 2: within the ours_cma model, pre(before fusion) vs post(after fusion) — same forward pass, isolating the causal effect of the CMA operator itself ===")
    print("=" * 70)
    for dataset in sorted(agg.keys()):
        cma = agg[dataset].get("ours_cma", {})
        n_layers = max((li for (li, _) in cma.keys()), default=-1) + 1
        print(f"\n[{dataset}]")
        for li in range(n_layers):
            pre_vals = [m for m, _, _ in cma.get((li, "pre"), [])]
            pre_nulls = [nm for _, _, nm in cma.get((li, "pre"), [])]
            post_vals = [m for m, _, _ in cma.get((li, "post"), [])]
            post_nulls = [nm for _, _, nm in cma.get((li, "post"), [])]
            if not pre_vals or not post_vals:
                continue
            pm, qm = np.mean(pre_vals), np.mean(post_vals)
            pn, qn = np.mean(pre_nulls), np.mean(post_nulls)
            delta = qm - pm
            print(f"  layer {li}: pre={pm:.4f} (null={pn:.4f})  post={qm:.4f} (null={qn:.4f})  "
                  f"delta={delta:+.4f}")
            if delta > (qn - pn) + 0.02:
                print("    <-- post-pre increase exceeds the null-hypothesis (shuffled) refit noise"
                      " floor; the CMA operator itself does re-couple the two streams")
            else:
                print("    <-- post-pre increase is on the same scale as the null-hypothesis refit"
                      " noise; the increase can be explained by fitting bias and is insufficient to"
                      " prove that CMA breaks disentanglement")


if __name__ == "__main__":
    main()
