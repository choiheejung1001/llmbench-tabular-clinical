#!/usr/bin/env python3
"""Forest plot of paired differences (LLM - ML baseline) per LLM and training seed (Figs. 3-5, Supplementary Figs. S7-S8), from the JSON
files written by scripts/evaluate.py (one per seed, the same fixed ML baseline as --reference in every file).

  CUDA_VISIBLE_DEVICES="" python figures/make_forest_plots.py --reference LightGBM --metric AUROC AUPRC --out Fig4.png \
      --eval 3407=eval_seed3407.json --eval 1=eval_seed1.json --eval 2=eval_seed2.json
Metrics: AUROC, AUPRC, ECE, Brier (classification); MAE, within20 (warfarin). Filled markers: 95% CI excludes 0.
Repeat the call (or merge the JSON files) for each field-name condition.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="append", required=True, help="SEED=evaluate.py JSON")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--metric", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ev = {s: json.load(open(p)) for s, p in (e.split("=", 1) for e in a.eval)}
    llms = [m for m in next(iter(ev.values()))["models"] if m != a.reference]
    fig, axes = plt.subplots(1, len(a.metric), figsize=(4.2 * len(a.metric), 0.9 + 0.9 * len(llms)), squeeze=False, sharey=True)
    for ax, k in zip(axes[0], a.metric):
        for i, m in enumerate(llms):
            for j, s in enumerate(ev):
                d = ev[s]["models"][m]["delta_vs_" + a.reference][k]; y = i + 0.25 * (j - (len(ev) - 1) / 2)
                excl = d["lo"] > 0 or d["hi"] < 0
                ax.plot([d["lo"], d["hi"]], [y, y], color="0.3", lw=1.2); ax.plot(d["est"], y, "os^"[j % 3], color="0.2", mfc="0.2" if excl else "white", ms=5)
        ax.axvline(0, color="0.5", lw=0.8, ls="--"); ax.set_yticks(range(len(llms))); ax.set_yticklabels(llms); ax.invert_yaxis()
        ax.set_title(f"{k}: LLM - {a.reference}", fontsize=9)
    fig.tight_layout(); fig.savefig(a.out, dpi=350, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
