"""Evaluate prediction CSVs of several models on the same test patients (paired), with optional validation predictions.

Classification (--task mace|mortality), each model given as NAME=TEST_CSV[,VALID_CSV]:
  * AUROC, AUPRC, ECE (confidence), Brier, calibration slope / intercept, with 95% CI (simple bootstrap, B=1000, seed 2026);
  * paired differences vs --reference (AUROC, AUPRC, ECE, Brier) with CI and bootstrap p;
  * if validation CSVs are given: Platt and temperature recalibration fitted on validation and applied to test (metrics after
    recalibration), and operating thresholds chosen on validation (Youden, sensitivity >= 0.80, F1-max, plus 0.5) applied to test;
  * decision curves: range summary over --dca-range (step 0.01) and net benefit at --dca-steps.
Regression (--task warfarin): MAE and % within +-20% with CI, paired differences vs --reference.

  python scripts/evaluate.py --task mortality --reference CatBoost \
     --model CatBoost=ml/test.csv,ml/valid.csv --model Qwen7B=llm/test.csv,llm/valid.csv --dca-range 0.30 0.70 --dca-steps 0.3 0.5 0.7 --out eval.json
Prediction CSV columns: classification y_true, y_pred, prob_<label>...; regression y_true, y_pred.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from llmbench.serialize import task_spec  # noqa: E402
from llmbench import metrics as M  # noqa: E402


def read_cls(path, pos):
    d = pd.read_csv(path)
    return ((d.y_true.astype(str) == pos).astype(int).values, d[f"prob_{pos}"].astype(float).values,
            (d.y_pred.astype(str) == pos).astype(int).values)


def ci(dct, d=4):
    return {k: (round(v, d) if isinstance(v, float) else v) for k, v in dct.items()}


def eval_cls(T, models, ref, dca_range, dca_steps, B):
    pos = T["positive_label"]; data = {}
    for name, (tp, vp) in models.items():
        data[name] = dict(test=read_cls(tp, pos), valid=(read_cls(vp, pos) if vp else None))
    y = next(iter(data.values()))["test"][0]
    for name, d in data.items():
        assert np.array_equal(d["test"][0], y), f"test rows of {name} differ from the first model (paired evaluation needs the same order)"
    IDX = M.boot_indices(len(y), B=B); W = M.count_matrix(IDX)
    out = dict(n_test=int(len(y)), prevalence_test=float(y.mean()), models={})
    E = {}
    for name, d in data.items():
        _, p, pred = d["test"]
        est, reps = M.cls_eval(y, p, pred=pred, IDX=IDX, W=W); E[name] = (est, reps)
        rec = {k: ci(M.summarize(est[k], reps[k])) for k in ("AUROC", "AUPRC", "ECE", "Brier")}
        rec.update(slope=round(est["slope"], 4), intercept=round(est["intercept"], 4))
        if d["valid"] is not None:
            yv, pv, _ = d["valid"]; rec["recalibration"] = {}
            for meth, fitter in (("platt", M.fit_platt), ("temperature", M.fit_temperature)):
                par, fn = fitter(yv, pv); e2, r2 = M.cls_eval(y, fn(p), IDX=IDX, W=W)
                rec["recalibration"][meth] = dict(params=par, **{k: ci(M.summarize(e2[k], r2[k])) for k in ("AUROC", "ECE", "Brier")},
                                                  slope=round(e2["slope"], 4), intercept=round(e2["intercept"], 4))
            th = M.choose_thresholds(yv, pv); rec["thresholds"] = {}
            for rule, t in th.items():
                tpred = (p >= t).astype(int) if rule != "fixed_0_5" else pred
                est_c, reps_c = M.confusion_stats(y, tpred, W)
                rec["thresholds"][rule] = dict(threshold=round(t, 4), pct_flagged=round(100 * float(tpred.mean()), 1),
                                               **{k: ci(M.summarize(est_c[k], reps_c[k])) for k in ("sens", "spec", "PPV", "NPV", "F1", "J")})
        if dca_steps:
            ts = np.array(dca_steps); nb, nbb = M.net_benefit(y, p, ts, W)
            rec["net_benefit_at_steps"] = {f"{t:.2f}": ci(M.summarize(nb[j], nbb[:, j])) for j, t in enumerate(ts)}
        out["models"][name] = rec
    if ref:
        for name in data:
            if name != ref:
                out["models"][name]["delta_vs_" + ref] = {k: ci(M.paired(E[name][0][k], E[ref][0][k], E[name][1][k], E[ref][1][k]))
                                                         for k in ("AUROC", "AUPRC", "ECE", "Brier")}
    if dca_steps:
        out["treat_all_net_benefit_at_steps"] = {f"{t:.2f}": round(float(v), 4) for t, v in zip(dca_steps, M.treat_all(y, np.array(dca_steps)))}
    if dca_range:
        a, b = dca_range
        summ = M.dca_range_summary(y, {n: d["test"][1] for n, d in data.items()}, ref or next(iter(data)), a, b, IDX)
        out["dca_range_summary"] = dict(range=[a, b], models={m: {k: (ci(v) if isinstance(v, dict) else round(v, 2)) for k, v in s.items()}
                                                              for m, s in summ.items()})
    return out


def eval_reg(models, ref, B):
    data = {n: pd.read_csv(tp) for n, (tp, _) in models.items()}
    y = next(iter(data.values())).y_true.astype(float).values
    for n, d in data.items():
        assert np.allclose(d.y_true.astype(float).values, y), f"test rows of {n} differ"
    W = M.count_matrix(M.boot_indices(len(y), B=B)); E = {}; out = dict(n_test=int(len(y)), models={})
    for n, d in data.items():
        est, reps = M.reg_eval(y, d.y_pred.astype(float).values, W); E[n] = (est, reps)
        out["models"][n] = dict(MAE=ci(M.summarize(est["MAE"], reps["MAE"])), within20=ci(M.summarize(est["within20"], reps["within20"])),
                                n_parsed=est["n_parsed"], n_distinct=int(d.y_pred.nunique()))
    if ref:
        for n in data:
            if n != ref:
                out["models"][n]["delta_vs_" + ref] = {k: ci(M.paired(E[n][0][k], E[ref][0][k], E[n][1][k], E[ref][1][k])) for k in ("MAE", "within20")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["warfarin", "mace", "mortality"])
    ap.add_argument("--model", action="append", required=True, help="NAME=TEST_CSV[,VALID_CSV]")
    ap.add_argument("--reference", help="NAME of the comparator for paired differences")
    ap.add_argument("--dca-range", nargs=2, type=float)
    ap.add_argument("--dca-steps", nargs="*", type=float)
    ap.add_argument("--bootstrap", type=int, default=M.B)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = {}
    for s in a.model:
        name, paths = s.split("=", 1); parts = paths.split(",")
        models[name] = (parts[0], parts[1] if len(parts) > 1 else None)
    T = task_spec(a.task)
    res = eval_reg(models, a.reference, a.bootstrap) if T["type"] == "regression" else \
        eval_cls(T, models, a.reference, a.dca_range, a.dca_steps, a.bootstrap)
    res["bootstrap"] = f"simple paired, B={a.bootstrap}, seed {M.SEED}"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
