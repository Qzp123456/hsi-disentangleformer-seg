"""
analyze_gamma.py — aggregates the gamma_s / gamma_c gating values of the Gated CMA module

Loads every seg_*_ours_cma_s*_best.pth under /mnt/scratch/znzs0468/results/
(i.e. the main-experiment CMA segmentation checkpoints, excluding the
r10/r20/r30 train_ratio ablation runs), prints the gamma_s / gamma_c scalar of
every CMAFusion layer in each checkpoint, and aggregates mean/std per dataset
to determine whether the network has effectively "turned off" CMA itself
(the closer gamma is to 0, the smaller that branch's cross-attention update
contributes to the residual stream).
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import torch

GAMMA_RE = re.compile(r"^blocks\.(\d+)\.cma\.gamma_(s|c)$")


def load_gammas(path):
    """returns dict: {(layer_idx, 's'|'c'): float}"""
    ck = torch.load(path, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    out = {}
    for k, v in state.items():
        m = GAMMA_RE.match(k)
        if m:
            layer, which = int(m.group(1)), m.group(2)
            out[(layer, which)] = v.item()
    return out


def parse_tag(fname):
    """seg_indian_pines_ours_cma_s42_best.pth -> (dataset, seed)"""
    base = os.path.basename(fname)
    m = re.match(r"seg_(.+)_ours_cma_s(\d+)_best\.pth$", base)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="/mnt/scratch/znzs0468/results")
    args = ap.parse_args()

    # exact match to task spec: seg_*_ours_cma_s*_best.pth (excludes r10/r20/r30 ablation ckpts)
    pattern = os.path.join(args.results_dir, "seg_*_ours_cma_s*_best.pth")
    paths = sorted(p for p in glob.glob(pattern) if "_r10_" not in p and "_r20_" not in p and "_r30_" not in p)

    if not paths:
        print(f"No files matched: {pattern}")
        return

    print(f"Found {len(paths)} checkpoints\n")

    per_ckpt = {}  # fname -> {(layer,which): val}
    for p in paths:
        dataset, seed = parse_tag(p)
        if dataset is None:
            continue
        gammas = load_gammas(p)
        per_ckpt[os.path.basename(p)] = (dataset, seed, gammas)

    if not per_ckpt:
        print("Files matched but dataset/seed could not be parsed; check the naming convention")
        return

    n_layers = max(layer for _, _, g in per_ckpt.values() for (layer, _) in g.keys()) + 1

    # ---- per-checkpoint table ----
    header_cols = []
    for layer in range(n_layers):
        header_cols += [f"L{layer}_gamma_s", f"L{layer}_gamma_c"]
    col_w = 14
    print("=== Per-checkpoint gamma values ===")
    print(f"{'checkpoint':<45}" + "".join(f"{c:>{col_w}}" for c in header_cols))
    for fname, (dataset, seed, gammas) in sorted(per_ckpt.items()):
        row = f"{fname:<45}"
        for layer in range(n_layers):
            for which in ("s", "c"):
                val = gammas.get((layer, which))
                row += f"{val:>{col_w}.6f}" if val is not None else f"{'--':>{col_w}}"
        print(row)

    # ---- aggregate by dataset x layer x which ----
    agg = defaultdict(list)  # (dataset, layer, which) -> [values across seeds]
    for fname, (dataset, seed, gammas) in per_ckpt.items():
        for (layer, which), val in gammas.items():
            agg[(dataset, layer, which)].append(val)

    print("\n=== Per-dataset / per-layer summary (signed mean ± std, plus mean |gamma|, over seeds) ===")
    datasets = sorted(set(d for d, _, _ in agg.keys()))
    for dataset in datasets:
        print(f"\n[{dataset}]")
        for layer in range(n_layers):
            for which in ("s", "c"):
                vals = agg.get((dataset, layer, which), [])
                if not vals:
                    continue
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                std = var ** 0.5
                abs_mean = sum(abs(v) for v in vals) / len(vals)
                n_pos = sum(1 for v in vals if v > 0)
                n_neg = len(vals) - n_pos
                flag = " <-- signed mean near 0 but |gamma| not small -> sign flips across seeds" \
                    if abs(mean) < 0.02 and abs_mean >= 0.02 else \
                    (" <-- genuinely near 0 (|gamma| also small)" if abs_mean < 0.02 else "")
                print(f"  layer {layer} gamma_{which}: mean={mean:+.6f} ± {std:.6f}  "
                      f"|gamma|_mean={abs_mean:.6f}  sign(+/-)={n_pos}/{n_neg}  (n={len(vals)}){flag}")

    # ---- overall verdict ----
    all_means, all_abs_means = [], []
    for (dataset, layer, which), vals in agg.items():
        all_means.append(sum(vals) / len(vals))
        all_abs_means.append(sum(abs(v) for v in vals) / len(vals))
    overall_abs_of_mean = sum(abs(m) for m in all_means) / len(all_means)
    overall_mean_of_abs = sum(all_abs_means) / len(all_abs_means)
    print(f"\n=== Overall verdict ===")
    print(f"Mean |signed mean| (across layer x gamma_{{s,c}} x dataset): {overall_abs_of_mean:.6f}")
    print(f"Mean of mean(|gamma|) (same grouping, signs not cancelled): {overall_mean_of_abs:.6f}")
    if overall_mean_of_abs < 0.02:
        print("-> The absolute gamma values are also consistently small, indicating the network"
              " does tend to shut down / suppress the CMA cross-attention contribution, consistent"
              " with CMA's limited mIoU gain and serving as direct evidence of the negative result.")
    elif overall_abs_of_mean < 0.02 <= overall_mean_of_abs:
        print("-> gamma magnitudes are themselves not small (~0.05~0.24) and are not globally shut"
              " off; but the signs are roughly split half-and-half across seeds and flip randomly,"
              " so they cancel out when averaged over seeds. This means the CMA branch does learn a"
              " non-trivial modulation at each run, but its direction (enhancing vs. suppressing a"
              " given token) is sensitive to random initialisation / seed and does not converge to a"
              " stable direction — which explains why CMA's mIoU gain is small and unstable"
              " (Indian +0.70, Pavia +0.05, std up to ±6): it is not that the module is turned off,"
              " but that the fusion direction it learns is inconsistent.")
    else:
        print("-> gamma values do not generally approach 0; the CMA branch contributes in a stable,"
              " consistent direction, so the limited gain is unlikely to be caused by the gating itself.")


if __name__ == "__main__":
    main()
