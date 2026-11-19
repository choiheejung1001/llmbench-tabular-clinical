"""ML baselines: models of the original studies refitted with their ORIGINAL settings (no retuning, training seed 3407; CPU).

  warfarin   XGBoost (selected in the study), ANN, random forest, linear regression; missing -> -1 except XGBoost (native NaN); ANN on MinMax inputs
  mace       LightGBM (selected in the study), logistic regression, random forest, feed-forward network;
             no imputation, a missing-value indicator for every feature with missing values in the training split, missing set to 0, no scaling
  mortality  CatBoost, random forest, MLP, logistic regression, AdaBoost (SAMME.R; selected in the study) on two karyotype representations:
             TE (all categorical columns target-encoded) and Structured (chrom/chrom1 -> binary flags, llmbench/karyotype.py)
Settings: configs/ml_baselines.json. Every model is fitted on the training split only. The baseline that every LLM seed is compared with is
chosen on the validation split (warfarin lowest MAE of the 0.5-mg-rounded prediction; MACE / mortality highest AUROC; mortality on the TE
representation; ties -> first model of the config); the test split plays no role in the choice.

  CUDA_VISIBLE_DEVICES="" python scripts/ml_baselines.py fit --task mace --split-dir DATA/mace --out-dir RUNS/ml/mace
  CUDA_VISIBLE_DEVICES="" python scripts/ml_baselines.py fit --task mortality --split-dir DATA/mortality --out-dir RUNS/ml/mortality
Writes <out-dir>/preds/pred_<task>_<model>[_<representation>]_seed<seed>_{valid,test}.csv (format of scripts/evaluate.py) and
<out-dir>/selection.json (validation metrics, ranking, selected baseline).
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from llmbench import load_config  # noqa: E402
from llmbench.serialize import task_spec, apply_cleaning  # noqa: E402

NJ = 8
SPLITS = ("valid", "test")


def rnd05(p):
    return np.round(np.asarray(p, float) * 2) / 2


# ------------------------------------------------------------------------------------------------ data
def load(task, split_dir):
    """Return X, Y per split with the same cleaning and columns as the LLM inputs (warfarin: cleaning rules -> NaN)."""
    T = task_spec(task)
    sp = {s: pd.read_csv(os.path.join(split_dir, f"split_{s}.csv"), dtype=str) for s in ("train", "valid", "test")}
    X, Y = {}, {}
    for s, d in sp.items():
        if task == "warfarin":
            d, _ = apply_cleaning(d, task)
            X[s] = d[T["ml_features"]].apply(pd.to_numeric, errors="coerce")
            Y[s] = d[T["target_column"]].astype(float).values
        elif task == "mace":
            F = [c for c in d.columns if c not in T["split_csv_extra_columns"]]
            X[s] = d[F].apply(pd.to_numeric, errors="coerce")
            Y[s] = (d["event"] == "positive").astype(int).values
        else:
            F = [c for c in d.columns if c not in T["split_csv_extra_columns"]]
            X[s] = d[F].copy()
            for c in T["ml_numeric_columns"]:
                X[s][c] = pd.to_numeric(X[s][c], errors="coerce")
            Y[s] = (d["event"] == "dead").astype(int).values
    return X, Y


def write_preds(task, p, y, path):
    if task == "warfarin":
        pd.DataFrame({"y_true": y, "y_pred": rnd05(p)}).to_csv(path, index=False); return
    T = task_spec(task); pos, neg = T["labels"]
    pd.DataFrame({"y_true": np.where(y == 1, pos, neg), "y_pred": np.where(p >= 0.5, pos, neg),
                  f"prob_{pos}": p, f"prob_{neg}": 1 - p}).to_csv(path, index=False)


# ------------------------------------------------------------------------------------------------ warfarin
def fit_warfarin(X, Y, seed, models, S):
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import MinMaxScaler
    F = {s: X[s].fillna(-1) for s in X}                                   # original study: missing -> -1
    sc = MinMaxScaler().fit(F["train"]); Z = {s: sc.transform(F[s]) for s in F}
    out = {}
    for m in models:
        st = dict(S[m])
        if m == "XGBoost":
            est = xgb.XGBRegressor(**st, random_state=seed, n_jobs=NJ).fit(X["train"], Y["train"]); A = X   # native NaN handling
        elif m == "RandomForest":
            est = RandomForestRegressor(**st, random_state=seed, n_jobs=NJ).fit(F["train"], Y["train"]); A = F
        elif m == "LinearRegression":
            est = LinearRegression().fit(F["train"], Y["train"]); A = F
        else:
            st.pop("note", None); st["hidden_layer_sizes"] = tuple(st["hidden_layer_sizes"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = MLPRegressor(**st, random_state=seed).fit(Z["train"], Y["train"])
            A = Z
        out[m] = {s: est.predict(A[s]) for s in SPLITS}
    return out


# ------------------------------------------------------------------------------------------------ MACE
def fit_mace(X, Y, seed, models, S):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    miss = [c for c in X["train"].columns if X["train"][c].isna().any()]

    def ind(d):                                                            # NaN -> 0 plus one indicator per feature with missing values in train
        Z = d.fillna(0.0).copy()
        for c in miss:
            Z[f"{c}__missing"] = d[c].isna().astype(float)
        return Z.values.astype(np.float64)
    Z = {s: ind(X[s]) for s in X}
    out = {}
    for m in models:
        st = dict(S[m])
        if m == "LightGBM":
            import lightgbm as lgb
            est = lgb.LGBMClassifier(**st, random_state=seed, n_jobs=NJ, verbose=-1)
        elif m == "LogisticRegression":
            est = LogisticRegression(**st, random_state=seed)
        elif m == "RandomForest":
            est = RandomForestClassifier(**st, random_state=seed, n_jobs=NJ)
        else:
            out[m] = fnn(Z, Y, seed, st); continue
        est.fit(Z["train"], Y["train"]); out[m] = {s: est.predict_proba(Z[s])[:, 1] for s in SPLITS}
    return out


def fnn(Z, Y, seed, st):
    import keras
    import tensorflow as tf
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    h = st["hidden"]
    net = keras.Sequential([keras.Input(shape=(Z["train"].shape[1],)),
                            keras.layers.Dense(h[0], activation=st["activation"]), keras.layers.Dropout(st["dropout"]),
                            keras.layers.Dense(h[1], activation=st["activation"]), keras.layers.Dropout(st["dropout"]),
                            keras.layers.Dense(1, activation="sigmoid")])
    net.compile(optimizer=keras.optimizers.Adam(learning_rate=st["learning_rate"], beta_1=st["beta_1"], beta_2=st["beta_2"]), loss=st["loss"])
    net.fit(Z["train"].astype("float32"), Y["train"].astype("float32"), batch_size=st["batch_size"], epochs=st["epochs"], verbose=0, shuffle=True)
    return {s: net.predict(Z[s].astype("float32"), batch_size=1024, verbose=0).ravel().astype(np.float64) for s in SPLITS}


# ------------------------------------------------------------------------------------------------ mortality
FLAGS = ["avail", "normal", "complex", "monosomal", "m7_del7q", "m5_del5q", "t9_22", "cbf_apl", "kmt2a", "inv3", "t6_9", "abn17p", "tri8"]
KARYO = ("chrom", "chrom1")


def structured(X):
    """Replace the two karyotype text columns by their binary flags (llmbench/karyotype.py)."""
    from llmbench.karyotype import parse
    Xs = X.drop(columns=list(KARYO)).copy()
    for col in KARYO:
        R = pd.DataFrame([parse(v if isinstance(v, str) else None) for v in X[col]])
        for k in FLAGS:
            Xs[f"{col}_{k}"] = R[k].astype(float).values
    return Xs


def make_adaboost():
    """scikit-learn 1.6 AdaBoostClassifier with the SAMME.R boost step and probability formula of scikit-learn <= 1.5."""
    from scipy.special import xlogy
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble._weight_boosting import _samme_proba

    class AdaBoostSAMMER(AdaBoostClassifier):
        def _boost(self, iboost, X_, y_, sample_weight, random_state):
            estimator = self._make_estimator(random_state=random_state)
            estimator.fit(X_, y_, sample_weight=sample_weight)
            proba = estimator.predict_proba(X_)
            if iboost == 0:
                self.classes_ = getattr(estimator, "classes_", None); self.n_classes_ = len(self.classes_)
            y_predict = self.classes_.take(np.argmax(proba, axis=1), axis=0)
            incorrect = y_predict != y_
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
            if estimator_error <= 0:
                return sample_weight, 1.0, 0.0
            Kc = self.n_classes_
            y_coding = np.array([-1.0 / (Kc - 1), 1.0]).take(self.classes_ == y_[:, np.newaxis])
            np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
            ew = -1.0 * self.learning_rate * ((Kc - 1.0) / Kc) * xlogy(y_coding, proba).sum(axis=1)
            if not iboost == self.n_estimators - 1:
                sample_weight *= np.exp(ew * ((sample_weight > 0) | (ew < 0)))
            return sample_weight, 1.0, estimator_error

        def decision_function(self, X_):
            X_ = self._check_X(X_)
            pred = sum(_samme_proba(e, self.n_classes_, X_) for e in self.estimators_)
            pred /= self.estimator_weights_.sum()
            pred[:, 0] *= -1
            return pred.sum(axis=1)
    return AdaBoostSAMMER


def encode(X, Y, cat):
    """Target encoding fitted on the training split, then mean imputation fitted on the encoded training split."""
    import category_encoders as ce
    from sklearn.impute import SimpleImputer
    enc = ce.TargetEncoder(cols=cat, drop_invariant=False).fit(X["train"], pd.Series(Y["train"]))
    E = {s: enc.transform(X[s]) for s in X}
    imp = SimpleImputer(strategy="mean").fit(E["train"])
    return {s: pd.DataFrame(imp.transform(E[s]), columns=E["train"].columns) for s in X}


def fit_mortality(X, Y, seed, models, S, rep, out_dir):
    import category_encoders as ce
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    T = task_spec("mortality")
    if rep == "Structured":
        X = {s: structured(X[s]) for s in X}; cat = [c for c in X["train"].columns if c not in T["ml_numeric_columns"] and not c.startswith(KARYO)]
    else:
        cat = [c for c in X["train"].columns if c not in T["ml_numeric_columns"]]
    M = encode(X, Y, cat); out = {}
    for m in models:
        st = {k: v for k, v in S[m].items() if k != "note"}
        if m == "CatBoost":
            from catboost import CatBoostClassifier
            enc2 = ce.TargetEncoder(cols=cat, drop_invariant=False).fit(M["train"], pd.Series(Y["train"]))       # second encoding, as in the original code
            A = {s: enc2.transform(M[s]) for s in M}
            est = CatBoostClassifier(**st, random_seed=seed, verbose=0, thread_count=NJ, train_dir=os.path.join(out_dir, "catboost_info"))
        else:
            A = M
            if m == "RandomForest":
                est = RandomForestClassifier(**st, random_state=seed, n_jobs=NJ)
            elif m == "MLP":
                st["hidden_layer_sizes"] = tuple(st["hidden_layer_sizes"]); est = MLPClassifier(**st, random_state=seed)
            elif m == "LogisticRegression":
                est = LogisticRegression(**st, random_state=seed)
            else:
                st.pop("algorithm", None); est = make_adaboost()(**st, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(A["train"], Y["train"])
        out[m] = {s: np.round(est.predict_proba(A[s])[:, 1], 12) for s in SPLITS}
    return out


# ------------------------------------------------------------------------------------------------ main
def select(task, P, Y, models):
    """Validation selection with exact comparison; warfarin integer sum of |error| in 0.25-mg units, classification AUROC."""
    if task == "warfarin":
        v = {m: int(np.rint(np.abs(rnd05(P[m]["valid"]) - Y["valid"]) * 4).sum()) for m in models}; best = min(v.values())
        val = {m: v[m] / 4 / len(Y["valid"]) for m in models}; key = "valid_MAE"
    else:
        val = {m: float(roc_auc_score(Y["valid"], P[m]["valid"])) for m in models}; best = max(val.values()); key = "valid_AUROC"
    tied = [m for m in models if (v[m] if task == "warfarin" else val[m]) == best]
    return {key: val, "selected": tied[0], "co_best": tied}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fit"])
    ap.add_argument("--task", required=True, choices=["warfarin", "mace", "mortality"])
    ap.add_argument("--split-dir", required=True, help="folder with split_{train,valid,test}.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="*", help="subset of the models of the config (default: all)")
    ap.add_argument("--representation", nargs="*", choices=["TE", "Structured"], default=["TE", "Structured"], help="mortality only")
    a = ap.parse_args()
    C = load_config("ml_baselines"); seed = C["seed"]; cfg = C[a.task]; models = a.models or cfg["models"]
    X, Y = load(a.task, a.split_dir)
    os.makedirs(os.path.join(a.out_dir, "preds"), exist_ok=True)
    sel = dict(task=a.task, seed=seed, models=models)
    reps = a.representation if a.task == "mortality" else [None]
    for rep in reps:
        if a.task == "warfarin":
            P = fit_warfarin(X, Y, seed, models, cfg["settings"])
        elif a.task == "mace":
            P = fit_mace(X, Y, seed, models, cfg["settings"])
        else:
            P = fit_mortality(X, Y, seed, models, cfg["settings"], rep, a.out_dir)
        for m, ps in P.items():
            for s in SPLITS:
                write_preds(a.task, ps[s], Y[s], os.path.join(a.out_dir, "preds", f"pred_{a.task}_{m}{'_' + rep if rep else ''}_seed{seed}_{s}.csv"))
        r = select(a.task, P, Y, models)
        if a.task != "warfarin":
            r["valid_AUPRC"] = {m: float(average_precision_score(Y["valid"], P[m]["valid"])) for m in models}
        sel[rep or "result"] = r
        print(rep or a.task, "selected baseline:", r["selected"])
    json.dump(sel, open(os.path.join(a.out_dir, "selection.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
