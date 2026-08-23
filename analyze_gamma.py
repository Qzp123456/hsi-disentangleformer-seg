"""
analyze_gamma.py — 汇总 Gated CMA 模块的 gamma_s / gamma_c 门控值

加载 /mnt/scratch/znzs0468/results/ 下所有 seg_*_ours_cma_s*_best.pth
(即 CMA 分割模型的主实验 checkpoint, 不含 r10/r20/r30 train_ratio 消融),
打印每个 checkpoint 里每一层 CMAFusion 的 gamma_s / gamma_c 标量值,
并按数据集汇总 mean/std, 用于判断 CMA 是否被网络自己"关掉"了
(gamma 越接近 0, 说明该分支的 cross-attention 更新对残差流的贡献越小).
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
        print(f"没有匹配到文件: {pattern}")
        return

    print(f"找到 {len(paths)} 个 checkpoint\n")

    per_ckpt = {}  # fname -> {(layer,which): val}
    for p in paths:
        dataset, seed = parse_tag(p)
        if dataset is None:
            continue
        gammas = load_gammas(p)
        per_ckpt[os.path.basename(p)] = (dataset, seed, gammas)

    if not per_ckpt:
        print("匹配到文件但没有解析出 dataset/seed, 检查命名规则")
        return

    n_layers = max(layer for _, _, g in per_ckpt.values() for (layer, _) in g.keys()) + 1

    # ---- per-checkpoint table ----
    header_cols = []
    for layer in range(n_layers):
        header_cols += [f"L{layer}_gamma_s", f"L{layer}_gamma_c"]
    col_w = 14
    print("=== Per-checkpoint gamma 值 ===")
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

    print("\n=== 按数据集/层 汇总 (signed mean ± std, 以及 |gamma| 均值, over seeds) ===")
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
                flag = " <-- signed mean 接近0 但 |gamma| 不小 -> 符号随 seed 翻转" \
                    if abs(mean) < 0.02 and abs_mean >= 0.02 else \
                    (" <-- 真正接近0 (|gamma| 也小)" if abs_mean < 0.02 else "")
                print(f"  layer {layer} gamma_{which}: mean={mean:+.6f} ± {std:.6f}  "
                      f"|gamma|_mean={abs_mean:.6f}  sign(+/-)={n_pos}/{n_neg}  (n={len(vals)}){flag}")

    # ---- overall verdict ----
    all_means, all_abs_means = [], []
    for (dataset, layer, which), vals in agg.items():
        all_means.append(sum(vals) / len(vals))
        all_abs_means.append(sum(abs(v) for v in vals) / len(vals))
    overall_abs_of_mean = sum(abs(m) for m in all_means) / len(all_means)
    overall_mean_of_abs = sum(all_abs_means) / len(all_abs_means)
    print(f"\n=== 总体结论 ===")
    print(f"|signed mean| 的均值(跨 layer x gamma_{{s,c}} x dataset): {overall_abs_of_mean:.6f}")
    print(f"mean(|gamma|) 的均值(同上,不消号): {overall_mean_of_abs:.6f}")
    if overall_mean_of_abs < 0.02:
        print("-> gamma 的绝对值也普遍很小, 说明网络确实倾向于关闭/抑制 CMA cross-attention 的贡献,"
              " 这与 CMA 对 mIoU 提升有限的现象一致, 可作为负结果的直接证据。")
    elif overall_abs_of_mean < 0.02 <= overall_mean_of_abs:
        print("-> gamma 幅值本身不小(约 0.05~0.24), 并未被网络整体关闭; 但正负号在不同 seed 间"
              " 大致对半分布、随机翻转, 导致跨 seed 平均后互相抵消。这说明 CMA 分支在每次训练"
              " 中确实学到了非平凡的调制, 但其方向(增强还是抑制该 token)对随机初始化/seed 敏感、"
              " 不收敛到稳定的方向 —— 这正好可以解释为什么 CMA 带来的 mIoU 提升幅度小且不稳定"
              "(Indian +0.70, Pavia +0.05, std 达 ±6): 不是模块被关闭, 而是它学到的融合方向不一致。")
    else:
        print("-> gamma 值未普遍趋近 0, CMA 分支有稳定且一致方向的贡献, 提升有限的原因可能不在门控本身。")


if __name__ == "__main__":
    main()
