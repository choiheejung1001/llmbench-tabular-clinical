"""Extra comparator rows of the main tables, computed from the split CSVs (CPU, no tuning, fitted on the training split only).

  warfarin   Physician         physician's initial dose (column initial_dose), as recorded
             Constant          training-split median of final_dose for every patient
  mace       LogReg            unpenalised logistic regression (lbfgs, max_iter 10000) after mean imputation and standardisation
  mortality  LogReg            same; the categorical columns are target-encoded first
(the original-study models are fitted by scripts/ml_baselines.py)
Inputs are read with scripts/ml_baselines.py load() (same cleaning and columns as the primary ML baseline).
Output: <out-dir>/pred_<task>_<model>_{valid,test}.csv in the format of scripts/evaluate.py, e.g.
  CUDA_VISIBLE_DEVICES="" python scripts/extra_baselines.py --task warfarin --split-dir DATA/warfarin --out-dir RUNS/extra/warfarin
  CUDA_VISIBLE_DEVICES="" python scripts/evaluate.py --task warfarin \
      --model Physician=RUNS/extra/warfarin/pred_warfarin_Physician_test.csv --model Constant=RUNS/extra/warfarin/pred_warfarin_Constant_test.csv \
      --out RUNS/eval_extra_warfarin.json
The IWPC clinical-algorithm row is not produced here: it needs an enzyme-inducer variable that is not in the split tables (README).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, ".."))); sys.path.insert(0, HERE)
from llmbench.serialize import task_spec  # noqa: E402
from ml_baselines import load, write_preds  # noqa: E402

SPLITS = ("valid", "test")
UNROUNDED = ("Physician", "Constant")


def warfarin_models(X, Y, split_dir):
    init = {s: pd.to_numeric(pd.read_csv(os.path.join(split_dir, f"split_{s}.csv"), dtype=str)["initial_dose"], errors="coerce").values
            for s in SPLITS}
    med = float(np.median(Y["train"]))
    P = {"Physician": init, "Constant": {s: np.full(len(Y[s]), med) for s in SPLITS}}
    return P


def logreg(task, X, Y):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if task == "mortality":
        import category_encoders as ce
        cat = [c for c in X["train"].columns if c not in task_spec(task)["ml_numeric_columns"]]
        enc = ce.TargetEncoder(cols=cat, drop_invariant=False).fit(X["train"], pd.Series(Y["train"]))
        X = {s: (enc.transform(X[s], pd.Series(Y[s])) if s == "train" else enc.transform(X[s])) for s in X}
    imp = SimpleImputer(strategy="mean").fit(X["train"]); sc = StandardScaler().fit(imp.transform(X["train"]))
    Z = {s: sc.transform(imp.transform(X[s])) for s in X}
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000).fit(Z["train"], Y["train"])
    return {"LogReg": {s: lr.predict_proba(Z[s])[:, 1] for s in SPLITS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["warfarin", "mace", "mortality"])
    ap.add_argument("--split-dir", required=True, help="folder with split_{train,valid,test}.csv")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    X, Y = load(a.task, a.split_dir)
    P = warfarin_models(X, Y, a.split_dir) if a.task == "warfarin" else logreg(a.task, X, Y)
    os.makedirs(a.out_dir, exist_ok=True)
    for m, ps in P.items():
        for s in SPLITS:
            path = os.path.join(a.out_dir, f"pred_{a.task}_{m}_{s}.csv")
            if m in UNROUNDED:
                pd.DataFrame({"y_true": Y[s], "y_pred": ps[s]}).to_csv(path, index=False)
            else:
                write_preds(a.task, ps[s], Y[s], path)
        print("wrote", os.path.join(a.out_dir, f"pred_{a.task}_{m}_{{valid,test}}.csv"))


if __name__ == "__main__":
    main()
