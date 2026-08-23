"""
summarise.py — parse all FINAL lines from logs and print a benchmark table
(also emits LaTeX for direct use in the thesis)

Usage:
  python summarise.py --logs logs
"""
import argparse
import glob
import re
from collections import defaultdict
import numpy as np

PATTERN = re.compile(
    r"FINAL \[seg_(?P<ds>indian_pines|pavia|houston)_(?P<model>[a-z0-9_]+?)"
    r"(?:_r(?P<ratio>\d+))?_s(?P<seed>\d+)\]\s+"
    r"OA=(?P<oa>[\d.]+)\s+mIoU=(?P<miou>[\d.]+)\s+mAcc=(?P<macc>[\d.]+)"
    r"(?:\s+params=(?P<params>[\d.]+)M)?"
)

# spatial-protocol runs end their tag with _sp so they do NOT match PATTERN
PATTERN_SP = re.compile(
    r"FINAL \[seg_(?P<ds>indian_pines|pavia|houston)_(?P<model>[a-z0-9_]+)_r\d+_s(?P<seed>\d+)_sp\]\s+"
    r"OA=(?P<oa>[\d.]+)\s+mIoU=(?P<miou>[\d.]+)\s+mAcc=(?P<macc>[\d.]+)"
    r"(?:\s+params=(?P<params>[\d.]+)M)?"
)

# checkerboard-protocol runs: tag suffix _ch
PATTERN_CH = re.compile(
    r"FINAL \[seg_(?P<ds>indian_pines|pavia|houston)_(?P<model>[a-z0-9_]+)_r\d+_s(?P<seed>\d+)_ch\]\s+"
    r"OA=(?P<oa>[\d.]+)\s+mIoU=(?P<miou>[\d.]+)\s+mAcc=(?P<macc>[\d.]+)"
    r"(?:\s+params=(?P<params>[\d.]+)M)?"
)

DISPLAY = {
    "cnn3d": "3D-CNN",
    "vit": "ViT",
    "spectralformer": "SpectralFormer",
    "disentangle": "DisentangleFormer",
    "base": "DisentangleFormer",
    "ours_cma": "Ours (Gated CMA)",
    "cma": "Ours (Gated CMA)",
}
ORDER = ["3D-CNN", "ViT", "SpectralFormer", "DisentangleFormer", "Ours (Gated CMA)"]
DS_NAME = {"indian_pines": "Indian Pines", "pavia": "Pavia University",
           "houston": "Houston 2013"}


def collect(logs, pattern, seeds=None, ratios=None):
    """results[dataset][model][metric] = list of values; params dict.
    seeds: optional whitelist of seed strings, for paired protocol comparison.
    ratios: optional whitelist for the _r{ratio} tag; lines WITH a ratio not
    in the set are dropped (lines without a ratio tag are always kept)."""
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    params = {}
    for f in glob.glob(f"{logs}/*.out"):
        for line in open(f, errors="ignore"):
            m = pattern.search(line)
            if not m:
                continue
            d = m.groupdict()
            if seeds is not None and d["seed"] not in seeds:
                continue
            if ratios is not None and d.get("ratio") is not None \
                    and d["ratio"] not in ratios:
                continue
            model = DISPLAY.get(d["model"], d["model"])
            ds = d["ds"]
            for k in ("oa", "miou", "macc"):
                res[ds][model][k].append(float(d[k]) * 100)
            if d.get("params"):
                params[model] = float(d["params"])
    return res, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--spatial", action="store_true",
                    help="parse spatial-protocol runs (tag suffix _sp) instead")
    ap.add_argument("--checker", action="store_true",
                    help="parse checkerboard-protocol runs (tag suffix _ch) instead")
    ap.add_argument("--gap", action="store_true",
                    help="parse ALL protocols and print random vs spatial vs checker mIoU")
    args = ap.parse_args()

    if args.gap:
        # same 3 benchmark seeds on all sides -> paired comparison
        seeds = {"42", "123", "2024"}
        res_r, _ = collect(args.logs, PATTERN, seeds=seeds, ratios={"60"})
        res_s, _ = collect(args.logs, PATTERN_SP, seeds=seeds)
        res_c, _ = collect(args.logs, PATTERN_CH, seeds=seeds)
        print("=== Protocol comparison: random (leaky) vs spatial lr vs checker ===")
        print("    (paired on benchmark seeds 42/123/2024, mIoU mean over seeds)\n")
        for ds in ["indian_pines", "pavia", "houston"]:
            print(f"\n[{DS_NAME.get(ds, ds)}]")
            print(f"{'Method':<22}{'random':<10}{'spatial lr':<12}{'checker':<12}"
                  f"{'lr gap':<10}{'chk gap':<10}")
            for model in ORDER:
                if model not in res_r.get(ds, {}):
                    continue
                r = np.array(res_r[ds][model]["miou"])
                s = res_s.get(ds, {}).get(model, {}).get("miou")
                c = res_c.get(ds, {}).get(model, {}).get("miou")
                s_m, c_m = np.mean(s) if s else None, np.mean(c) if c else None
                s_str = f"{s_m:<12.2f}" if s_m is not None else f"{'-':<12}"
                c_str = f"{c_m:<12.2f}" if c_m is not None else f"{'-':<12}"
                g1 = f"{r.mean()-s_m:<10.2f}" if s_m is not None else f"{'-':<10}"
                g2 = f"{r.mean()-c_m:<10.2f}" if c_m is not None else f"{'-':<10}"
                print(f"{model:<22}{r.mean():<10.2f}{s_str}{c_str}{g1}{g2}")
        return

    res, params = collect(args.logs, PATTERN_CH if args.checker
                          else PATTERN_SP if args.spatial else PATTERN,
                          ratios={"60"})
    if args.checker:
        print("*** checkerboard split protocol (no overlap leakage) ***\n")
    elif args.spatial:
        print("*** spatial split protocol (no overlap leakage) ***\n")

    if not res:
        print("No FINAL lines found. Check --logs path.")
        return

    for ds in ["indian_pines", "pavia", "houston"]:
        if ds not in res:
            continue
        print("\n" + "=" * 74)
        print(f"  {DS_NAME.get(ds, ds)}")
        print("=" * 74)
        print(f"{'Method':<22}{'OA':<16}{'mIoU':<16}{'mAcc':<16}{'#seeds'}")
        print("-" * 74)
        for model in ORDER:
            if model not in res[ds]:
                continue
            r = res[ds][model]
            cells = []
            for k in ("oa", "miou", "macc"):
                v = np.array(r[k])
                cells.append(f"{v.mean():.2f}±{v.std():.2f}")
            n = len(r["oa"])
            print(f"{model:<22}{cells[0]:<16}{cells[1]:<16}{cells[2]:<16}{n}")

    # ── LaTeX ─────────────────────────────────────────────
    print("\n\n% ---------- LaTeX table ----------")
    print(r"\begin{table}[t]\centering")
    print(r"\caption{Semantic segmentation results (mean$\pm$std over seeds). "
          r"Best in \textbf{bold}.}")
    print(r"\begin{tabular}{l" + "ccc" * len(res) + "}")
    print(r"\toprule")
    hdr = "Method"
    for ds in res:
        hdr += r" & \multicolumn{3}{c}{" + DS_NAME.get(ds, ds) + "}"
    print(hdr + r" \\")
    print("".join([r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i)
                   for i in range(len(res))]))
    print(" " + " & OA & mIoU & mAcc" * len(res) + r" \\")
    print(r"\midrule")

    best = {}
    for ds in res:
        for k in ("oa", "miou", "macc"):
            best[(ds, k)] = max(np.mean(res[ds][m][k])
                                for m in res[ds] if res[ds][m][k])

    for model in ORDER:
        if not any(model in res[ds] for ds in res):
            continue
        row = model
        for ds in res:
            if model not in res[ds]:
                row += " & - & - & -"
                continue
            for k in ("oa", "miou", "macc"):
                v = np.array(res[ds][model][k])
                s = f"{v.mean():.2f}$\\pm${v.std():.2f}"
                if abs(v.mean() - best[(ds, k)]) < 1e-9:
                    s = r"\textbf{" + s + "}"
                row += " & " + s
        print(row + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
