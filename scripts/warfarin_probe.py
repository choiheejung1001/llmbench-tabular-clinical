"""Warfarin regression-head analysis, two-stage linear probe (supplementary).

Stage 1 (GPU): hidden-state features of the fine-tuned generative adapter (same model and seed), one record at a time:
  python scripts/infer_llm.py --task warfarin --model <m> --adapter <adapter> --jsonl <prefix>_<split>.jsonl --mode features --out feat_<split>.npy
  for split in train, valid, test.
Stage 2 (CPU, this script): features standardised with the training mean / SD (SD 0 -> 1); target = discharge dose standardised with the
training mean / SD (ddof 0); sklearn Ridge (intercept) for alpha in logspace(-2, 6, 33); alpha chosen by validation MAE (mg; ties -> larger
alpha); fit on the training split only; test predicted once. Prediction = mean + SD x output (continuous mg).

  CUDA_VISIBLE_DEVICES="" python scripts/warfarin_probe.py --features-prefix FEAT/feat --jsonl-dir DATA/warfarin/jsonl --out-prefix RUNS/probe/qwen2.5-3B_seed3407
Outputs: <out-prefix>_{valid,test}.csv (y_true, y_pred) and <out-prefix>_fit.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from llmbench.serialize import load_jsonl  # noqa: E402

ALPHAS = np.logspace(-2, 6, 33)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-prefix", required=True, help="<prefix>_{train,valid,test}.npy")
    ap.add_argument("--jsonl-dir", required=True, help="folder with warfarin_reg_{train,valid,test}.jsonl (targets)")
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    S = ("train", "valid", "test")
    Y = {s: np.array([float(r["output"]) for r in load_jsonl(os.path.join(a.jsonl_dir, f"warfarin_reg_{s}.jsonl"))]) for s in S}
    X = {s: np.load(f"{a.features_prefix}_{s}.npy").astype(np.float64) for s in S}
    assert all(X[s].shape[0] == len(Y[s]) and np.isfinite(X[s]).all() for s in S)
    mu, sd = float(Y["train"].mean()), float(Y["train"].std())
    xm = X["train"].mean(0); xs = X["train"].std(0); xs[xs == 0] = 1.0
    Z = {s: (X[s] - xm) / xs for s in S}; zt = (Y["train"] - mu) / sd
    curve = np.array([np.mean(np.abs(mu + sd * Ridge(alpha=al, fit_intercept=True).fit(Z["train"], zt).predict(Z["valid"]) - Y["valid"]))
                      for al in ALPHAS])
    best = int(np.flatnonzero(curve == curve.min())[-1]); al = float(ALPHAS[best])
    r = Ridge(alpha=al, fit_intercept=True).fit(Z["train"], zt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
    for s in ("valid", "test"):
        pd.DataFrame({"y_true": Y[s], "y_pred": mu + sd * r.predict(Z[s])}).to_csv(f"{a.out_prefix}_{s}.csv", index=False)
    fit = dict(feature_dim=int(X["train"].shape[1]), alpha_grid="logspace(-2, 6, 33)", alpha_selected=al,
               alpha_at_grid_edge=bool(best in (0, len(ALPHAS) - 1)), valid_MAE_selected=float(curve[best]),
               selection="validation MAE (mg); test not used")
    json.dump(fit, open(f"{a.out_prefix}_fit.json", "w"), indent=1)
    print(json.dumps(fit))


if __name__ == "__main__":
    main()
