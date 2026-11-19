#!/usr/bin/env python3
"""Decision curves (Fig. 6), reliability diagrams (Supplementary Figs. S2-S3) and confusion matrices (Supplementary Figs. S4-S6) of one
classification task, drawn from prediction CSVs (format of scripts/evaluate.py). The first --model is the ML baseline.

  CUDA_VISIBLE_DEVICES="" python figures/make_final_figures.py --task mace --out-dir figs \
      --model LightGBM=ml_test.csv,ml_valid.csv --model Qwen2.5-7B=llm_test.csv,llm_valid.csv --dca-range 0.10 0.30 --dca-steps 0.10 0.20 0.30

* Decision curves: net benefit with pointwise 95% bands (simple bootstrap, B = 1000, seed 2026, same resamples for all models); the
  threshold range is exploratory (no clinical-range claim); the faint curve is the part outside --dca-range.
* Reliability: 10 equal-width bins (event frequency vs mean predicted probability) and a table of patients per bin; bins with n < 5 are
  not plotted and shown as "<5".
* Confusion matrices at each model's threshold chosen on the validation split (Youden J; validation CSV required), applied unchanged to test.
Also writes the plotted aggregates next to each PNG (350 dpi).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from llmbench import metrics as M  # noqa: E402
from llmbench.serialize import task_spec  # noqa: E402

DPI = 350
PAL = ["black", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728", "#8c564b"]


def read(path, pos):
    d = pd.read_csv(path)
    return (d.y_true.astype(str) == pos).astype(int).values, d[f"prob_{pos}"].astype(float).values


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["mace", "mortality"])
    ap.add_argument("--model", action="append", required=True, help="NAME=TEST_CSV,VALID_CSV (first = ML baseline)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dca-range", nargs=2, type=float, required=True)
    ap.add_argument("--dca-steps", nargs="*", type=float, default=[])
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    T = task_spec(a.task); pos, neg = T["labels"]; names, test, valid = [], {}, {}
    for s in a.model:
        n, paths = s.split("=", 1); tp, vp = paths.split(","); names.append(n); test[n] = read(tp, pos); valid[n] = read(vp, pos)
    y = test[names[0]][0]; col = dict(zip(names, PAL * 3))
    for n in names:
        assert np.array_equal(test[n][0], y), "test rows differ between models (paired evaluation needs the same order)"

    # ---- decision curves
    pts = np.round(np.arange(0.01, 0.995, 0.01), 2); odds = pts / (1 - pts); n_ = len(y); W = M.count_matrix(M.boot_indices(n_))
    prev = y.mean(); ta = prev - (1 - prev) * odds; a0, b0 = a.dca_range; inr = (pts >= a0 - 1e-9) & (pts <= b0 + 1e-9); rows = []
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.axvspan(a0, b0, color="0.85", alpha=0.6, lw=0, label="Threshold range (exploratory)")
    for s in a.dca_steps:
        ax.axvline(s, color="0.5", lw=0.9, ls=":")
    ax.plot(pts, ta, "--", color="0.45", lw=1.3, label="Treat all"); ax.axhline(0, color="0.2", lw=1.0, label="Treat none")
    for n in names:
        nb, nbb = M.net_benefit(y, test[n][1], pts, W); lo, hi = np.percentile(nbb, [2.5, 97.5], axis=0)
        ax.fill_between(pts, lo, hi, color=col[n], alpha=0.10, lw=0); ax.plot(pts, nb, color=col[n], lw=1.8, alpha=0.35)
        ax.plot(pts[inr], nb[inr], color=col[n], lw=2.0, label=n)
        rows += [dict(model=n, threshold=t, net_benefit=nb[j], nb_lo=lo[j], nb_hi=hi[j], treat_all=ta[j]) for j, t in enumerate(pts)]
    ax.set_xlim(0, float(np.max(test[names[0]][1])) * 1.05 if a.task == "mace" else 1.0); ax.set_ylim(min(-0.05, -0.5 * prev), prev * 1.15)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit"); ax.grid(alpha=0.3); ax.legend(frameon=True, framealpha=0.9)
    fig.savefig(os.path.join(a.out_dir, f"DCA_{a.task}.png"), dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, f"DCA_{a.task}_data.csv"), index=False, float_format="%.6f")

    # ---- reliability
    edges = np.linspace(0, 1, 11); rel = []
    for n in names:
        b, _ = M.event_parts(y, test[n][1])
        for k in range(10):
            s = b == k; nk = int(s.sum())
            if nk:
                ok = nk >= 5
                rel.append(dict(model=n, bin=k, n=nk, mean_pred=test[n][1][s].mean() if ok else np.nan, observed=y[s].mean() if ok else np.nan, suppressed_n_lt_5=not ok))
    R = pd.DataFrame(rel); lim = min(1.0, float(max(test[n][1].max() for n in names)) * 1.1)
    fig = plt.figure(figsize=(6.6, 8.0)); gs = fig.add_gridspec(2, 1, height_ratios=[5.2, 1.6], hspace=0.28); ax = fig.add_subplot(gs[0])
    ax.plot([0, 1], [0, 1], "--", color="brown", lw=1, label="Perfectly calibrated")
    for n in names:
        r = R[(R.model == n) & ~R.suppressed_n_lt_5]; ax.plot(r.mean_pred, r.observed, "o-", lw=1.6, ms=4.5, color=col[n], label=n)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=9.5)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    bins = sorted(R.bin.unique()); tx = fig.add_subplot(gs[1]); tx.axis("off")
    def ncell(n, k):
        q = R[(R.model == n) & (R.bin == k)]
        return "–" if q.empty else "<5" if q.suppressed_n_lt_5.iloc[0] else str(int(q.n.iloc[0]))
    cell = [[ncell(n, k) for k in bins] for n in names]
    tb = tx.table(cellText=cell, rowLabels=names, colLabels=[f"{edges[k]:.1f}–{edges[k + 1]:.1f}" for k in bins], loc="upper center", bbox=[0.17, 0, 0.83, 1])
    tb.auto_set_font_size(False); tb.set_fontsize(7.5)
    fig.savefig(os.path.join(a.out_dir, f"reliability_{a.task}.png"), dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)
    R.to_csv(os.path.join(a.out_dir, f"reliability_{a.task}_data.csv"), index=False, float_format="%.6f")

    # ---- confusion matrices at the validation Youden threshold
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.4), squeeze=False); rows = []
    for ax, n in zip(axes[0], names):
        thr = M.choose_thresholds(*valid[n])["youden"]; pr = test[n][1] >= thr
        c = dict(TN=int((~pr & (y == 0)).sum()), FP=int((pr & (y == 0)).sum()), FN=int((~pr & (y == 1)).sum()), TP=int((pr & (y == 1)).sum()))
        sns.heatmap([[c["TN"], c["FP"]], [c["FN"], c["TP"]]], annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=[neg, pos], yticklabels=[neg, pos], ax=ax)
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title(f"{n}\n(validation Youden threshold {thr:.3f})", fontsize=11)
        rows.append(dict(model=n, threshold=thr, **c))
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, f"confusion_{a.task}.png"), dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, f"confusion_{a.task}_data.csv"), index=False)
    print("wrote figures to", a.out_dir)


if __name__ == "__main__":
    main()
