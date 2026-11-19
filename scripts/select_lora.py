"""LoRA r/alpha selection on the validation split (rule A).

For every model, the primary configuration and the five candidates of configs/llm_training.json (lora_search) are trained with
seed 3407 on the training split (scripts/train_llm.py) and predict the validation split. This script reads those validation
predictions, computes the selection metric (classification: AUROC; warfarin: MAE of the greedy prediction) and adopts the best
configuration; if the best value equals the primary value within 1e-6 the primary is kept. The paired bootstrap 95% CI of the
gain over the primary (simple, B=1000, seed 2026; one index matrix per task) is reported for information only.
The test split is not read.

  python scripts/select_lora.py --task mace --pred-dir RUNS/mace/preds --out adopted_mace.json
Expected file names (as written by train_llm.py): pred_<prefix>_<model>_r<r>_a<alpha>_seed3407_valid.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from llmbench import load_config, MODELS  # noqa: E402
from llmbench.serialize import task_spec  # noqa: E402
from llmbench import metrics as M  # noqa: E402


def load_valid(path, T):
    d = pd.read_csv(path)
    if T["type"] == "regression":
        return d.y_true.astype(float).values, d.y_pred.astype(float).values
    pos = T["positive_label"]
    return (d.y_true.astype(str) == pos).astype(int).values, d[f"prob_{pos}"].astype(float).values


def metric_with_reps(T, y, p, IDX, W):
    if T["type"] == "regression":
        ae = np.abs(p - y); ok = ~np.isnan(ae)
        return float(np.nanmean(ae)), (W @ np.where(ok, ae, 0.0)) / (W @ ok.astype(float))
    return float(roc_auc_score(y, p)), M.boot_metric(roc_auc_score, IDX, y, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["warfarin", "mace", "mortality"])
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()
    T = task_spec(a.task); S = load_config("llm_training")["lora_search"]["candidates"][a.task]
    sgn = 1 if S["direction"] == "max" else -1
    prim = tuple(S["primary"]); cfgs = [prim] + [tuple(c) for c in S["others"]]
    out = dict(task=a.task, metric=S["metric"], rule=load_config("llm_training")["lora_search"]["rule"], models={})
    IDX = W = yref = None
    for m in MODELS:
        ev = {}
        for r, al in cfgs:
            f = os.path.join(a.pred_dir, f"pred_{T['jsonl_prefix']}_{m}_r{r}_a{al}_seed{a.seed}_valid.csv")
            if not os.path.exists(f):
                continue
            y, p = load_valid(f, T)
            if IDX is None:
                IDX = M.boot_indices(len(y)); W = M.count_matrix(IDX); yref = y
            assert len(y) == len(yref) and np.allclose(y, yref), f"validation rows differ: {f}"
            ev[(r, al)] = metric_with_reps(T, y, p, IDX, W)
        rec = dict(n_configs=len(ev), values={f"r{r}_a{al}": round(v[0], 4) for (r, al), v in ev.items()})
        if len(ev) == len(cfgs):
            best = max(ev, key=lambda c: (sgn * ev[c][0], c == prim))
            if abs(ev[best][0] - ev[prim][0]) < 1e-6:
                best = prim
            d = M.paired(ev[best][0], ev[prim][0], ev[best][1], ev[prim][1])
            gain = sgn * d["est"]; glo, ghi = (d["lo"], d["hi"]) if sgn == 1 else (-d["hi"], -d["lo"])
            rec.update(r=best[0], alpha=best[1], is_primary=best == prim, valid_primary=round(ev[prim][0], 4), valid_selected=round(ev[best][0], 4),
                       gain=round(gain, 4), gain_ci=[round(glo, 4), round(ghi, 4)], gain_ci_excludes_0=d["excludes_0"])
        else:
            rec["status"] = f"incomplete: {len(ev)} of {len(cfgs)} configurations found"
        out["models"][m] = rec
        print(m, {k: v for k, v in rec.items() if k in ("r", "alpha", "valid_selected", "gain", "status")})
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
