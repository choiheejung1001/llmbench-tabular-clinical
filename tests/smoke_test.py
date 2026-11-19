"""CPU smoke test (no GPU, no patient data). Run from the package root:
    CUDA_VISIBLE_DEVICES="" python tests/smoke_test.py [--ml]

1. python -m py_compile on every .py file of the package;
2. serialization: scripts/make_jsonl.py on the synthetic split CSVs in examples/ for every naming condition, compared byte-for-byte
   with the example JSONL files, plus rule checks (cleaning -> 'Unknown', mortality age first, identifier keys V1..Vk);
3. evaluation: scripts/evaluate.py and scripts/select_lora.py on a tiny randomly generated prediction set, and direct checks of the
   metric functions on cases with known answers; scripts/extra_baselines.py on random tables with the example CSV as the test split;
4. figures/make_fig3_dose_distribution.py on the synthetic warfarin doses; --help of every script (imports without GPU);
5. figures/make_final_figures.py and make_forest_plots.py on randomly generated predictions;
6. --ml: scripts/ml_baselines.py fit (original settings, seed 3407, validation selection) on randomly generated tabular data for the three tasks.
Writes nothing outside a temporary folder. Exit code 0 = all checks passed.
"""
import glob
import json
import os
import py_compile
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True                                     # no __pycache__ (it would hold local paths) in the package
from llmbench import metrics as M, MODELS  # noqa: E402
from llmbench.serialize import task_spec, load_jsonl  # noqa: E402

PY = sys.executable
RESULTS = {}


def check(name, ok, detail=""):
    RESULTS[name] = bool(ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def run(args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([PY] + args, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
    return r.returncode == 0, r.stdout


def t_compile():
    files = sorted(glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)); bad = []
    for f in files:
        try:
            py_compile.compile(f, cfile=os.path.join(tempfile.gettempdir(), "llmbench_pyc_check.pyc"), doraise=True)
        except py_compile.PyCompileError as e:
            bad.append(f"{f}: {e}")
    check("py_compile", not bad, f"{len(files)} files" + (f", errors: {bad}" if bad else ""))


def t_serialization(tmp):
    for task in ("warfarin", "mace", "mortality"):
        T = task_spec(task); d = os.path.join(tmp, task); os.makedirs(d, exist_ok=True)
        pd.read_csv(os.path.join(ROOT, f"examples/synthetic_{task}_split.csv"), dtype=str).to_csv(os.path.join(d, "split_test.csv"), index=False)
        for nm in T["naming_conditions"]:
            ok, _ = run(["scripts/make_jsonl.py", "--task", task, "--split-dir", d, "--out-dir", os.path.join(d, nm), "--naming", nm, "--splits", "test"])
            new = open(os.path.join(d, nm, f"{T['jsonl_prefix']}_test.jsonl"), "rb").read()
            ref = open(os.path.join(ROOT, f"examples/synthetic_{task}_{nm}.jsonl"), "rb").read()
            check(f"serialize {task} {nm} == example", ok and new == ref)
            rows = load_jsonl(os.path.join(d, nm, f"{T['jsonl_prefix']}_test.jsonl"))
            keys = [[p.split(":", 1)[0] for p in r["input"].split(", ") if ":" in p] for r in rows]
            expect = list(T["keys"][{"original": "original_key", "descriptive": "descriptive_key", "identifier": "identifier_key"}[nm]])
            check(f"serialize {task} {nm} key order", all(k[:len(expect)] == expect for k in keys) and all(r["instruction"] == T["instruction"] for r in rows))
    w = load_jsonl(os.path.join(ROOT, "examples/synthetic_warfarin_original.jsonl"))
    check("warfarin cleaning (height 0, BP <= 20 -> Unknown)", "Height:Unknown" in w[0]["input"] and "Systolic blood pressure:Unknown" in w[1]["input"]
          and "Gender:Male" in w[0]["input"] and w[0]["output"] == "2.5")
    mo = load_jsonl(os.path.join(ROOT, "examples/synthetic_mortality_original.jsonl"))
    check("mortality age serialized first", all(r["input"].startswith("Age:") for r in mo) and {r["output"] for r in mo} <= {"dead", "alive"})


def t_metrics_known():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.8, 0.9])
    check("AUROC perfect = 1", abs(M.cls_eval(y, p, slope=False)[0]["AUROC"] - 1) < 1e-12)
    check("Brier", abs(M.brier(y, p) - np.mean((p - y) ** 2)) < 1e-12)
    check("ECE all-correct conf 1 = 0", M.conf_ece([1, 0], [1, 0], np.array([1.0, 0.0])) == 0.0)
    nb = M.net_benefit(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.2, 0.1]), np.array([0.5]))[0]
    check("net benefit by hand", abs(nb - (1 / 4 - 1 / 4 * 1.0)) < 1e-12)
    th = M.choose_thresholds(np.array([0, 0, 1, 1, 1, 0]), np.array([0.1, 0.3, 0.35, 0.8, 0.9, 0.4]))
    check("thresholds youden/sens80/f1max", set(th) == {"youden", "sens80", "f1max", "fixed_0_5"} and 0 < th["youden"] <= 1)
    IDX = M.boot_indices(50); check("bootstrap matrix shape (1000, n), seed 2026 reproducible", IDX.shape == (1000, 50) and np.array_equal(IDX, M.boot_indices(50)))
    st, _ = M.confusion_stats(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1]))
    check("confusion stats", abs(st["sens"] - 0.5) < 1e-12 and abs(st["spec"] - 0.5) < 1e-12)


def t_evaluate(tmp):
    rng = np.random.default_rng(0); d = os.path.join(tmp, "eval"); os.makedirs(d, exist_ok=True)
    for split, n in (("valid", 60), ("test", 80)):
        y = rng.integers(0, 2, n)
        for name, noise in (("ML", 1.0), ("LLM", 1.5)):
            p = 1 / (1 + np.exp(-(1.2 * (2 * y - 1) + rng.normal(0, noise, n))))
            pd.DataFrame({"y_true": np.where(y == 1, "dead", "alive"), "y_pred": np.where(p >= 0.5, "dead", "alive"),
                          "prob_dead": p, "prob_alive": 1 - p}).to_csv(os.path.join(d, f"{name}_{split}.csv"), index=False)
    out = os.path.join(d, "eval_cls.json")
    ok, _ = run(["scripts/evaluate.py", "--task", "mortality", "--reference", "ML",
                 "--model", f"ML={d}/ML_test.csv,{d}/ML_valid.csv", "--model", f"LLM={d}/LLM_test.csv,{d}/LLM_valid.csv",
                 "--dca-range", "0.30", "0.70", "--dca-steps", "0.3", "0.5", "0.7", "--out", out])
    R = json.load(open(out)) if ok else {}
    L = R.get("models", {}).get("LLM", {})
    check("evaluate.py classification (metrics, CI, paired delta, recalibration, thresholds, DCA)",
          ok and {"AUROC", "AUPRC", "ECE", "Brier", "recalibration", "thresholds", "delta_vs_ML", "net_benefit_at_steps"} <= set(L)
          and set(L["thresholds"]) == {"youden", "sens80", "f1max", "fixed_0_5"} and "dca_range_summary" in R)
    y = rng.choice([1.5, 2.0, 2.5, 3.0, 4.0], 80)
    for name, s in (("XGBoost", 0.4), ("LLM", 0.6)):
        pd.DataFrame({"y_true": y, "y_pred": np.round((y + rng.normal(0, s, 80)) * 2) / 2}).to_csv(os.path.join(d, f"{name}_reg.csv"), index=False)
    ok, _ = run(["scripts/evaluate.py", "--task", "warfarin", "--reference", "XGBoost", "--model", f"XGBoost={d}/XGBoost_reg.csv",
                 "--model", f"LLM={d}/LLM_reg.csv", "--out", os.path.join(d, "eval_reg.json")])
    R = json.load(open(os.path.join(d, "eval_reg.json"))) if ok else {}
    check("evaluate.py regression (MAE, within 20%, paired delta)", ok and "delta_vs_XGBoost" in R.get("models", {}).get("LLM", {}))
    # rule A selection on synthetic validation predictions
    S = json.load(open(os.path.join(ROOT, "configs/llm_training.json")))["lora_search"]["candidates"]["mace"]
    yv = rng.integers(0, 2, 70); pdir = os.path.join(d, "lora"); os.makedirs(pdir, exist_ok=True)
    for m in MODELS:
        for k, (r, a) in enumerate([S["primary"]] + S["others"]):
            p = 1 / (1 + np.exp(-((0.5 + 0.2 * k) * (2 * yv - 1) + rng.normal(0, 1, 70))))
            pd.DataFrame({"y_true": np.where(yv == 1, "positive", "negative"), "y_pred": np.where(p >= 0.5, "positive", "negative"),
                          "prob_positive": p, "prob_negative": 1 - p}).to_csv(os.path.join(pdir, f"pred_mace_cls_{m}_r{r}_a{a}_seed3407_valid.csv"), index=False)
    ok, _ = run(["scripts/select_lora.py", "--task", "mace", "--pred-dir", pdir, "--out", os.path.join(d, "adopted.json")])
    A = json.load(open(os.path.join(d, "adopted.json"))) if ok else {}
    check("select_lora.py rule A", ok and all("r" in A["models"][m] and "gain_ci" in A["models"][m] for m in MODELS))


def synthetic_table(task, n, rng):
    """Random values only (format test), columns from the key map."""
    T = task_spec(task); cols = list(T["keys"].csv_column); df = pd.DataFrame({"patient_id": np.arange(n)})
    for c in cols:
        if task == "mortality" and c not in T["ml_numeric_columns"]:
            df[c] = rng.choice(["a", "b", "c", None], n)
        else:
            df[c] = np.round(rng.normal(50, 10, n), 1)
    if task == "warfarin":
        df["Gender"] = rng.integers(1, 3, n); df["initial_dose"] = 2.5
        df["final_dose"] = rng.choice([1.5, 2.0, 2.5, 3.0, 3.5], n); df["dose_class"] = "maintained"
    else:
        lab = T["labels"]; df["event"] = rng.choice(lab, n)
    return df


def t_ml(tmp):
    rng = np.random.default_rng(1)
    for task, n_expect in (("warfarin", 8), ("mace", 8), ("mortality", 20)):
        d = os.path.join(tmp, "ml", task); os.makedirs(d, exist_ok=True)
        for s, n in (("train", 120), ("valid", 50), ("test", 50)):
            synthetic_table(task, n, rng).to_csv(os.path.join(d, f"split_{s}.csv"), index=False)
        ok, _ = run(["scripts/ml_baselines.py", "fit", "--task", task, "--split-dir", d, "--out-dir", os.path.join(d, "out")])
        n_pred = len(glob.glob(os.path.join(d, "out", "preds", "*.csv")))
        sel = json.load(open(os.path.join(d, "out", "selection.json"))) if ok else {}
        check(f"ml_baselines.py fit {task} (synthetic random data, original settings, seed 3407)", ok and n_pred == n_expect and bool(sel), f"{n_pred} prediction files")


def t_extra(tmp):
    """scripts/extra_baselines.py: random training/validation tables + the synthetic example CSV as the test split."""
    rng = np.random.default_rng(2)
    for task in ("warfarin", "mace", "mortality"):
        d = os.path.join(tmp, "extra", task); os.makedirs(d, exist_ok=True)
        for s, n in (("train", 120), ("valid", 50)):
            synthetic_table(task, n, rng).to_csv(os.path.join(d, f"split_{s}.csv"), index=False)
        cols = list(pd.read_csv(os.path.join(d, "split_train.csv"), nrows=0).columns)          # same column order in every split
        ex = pd.read_csv(os.path.join(ROOT, f"examples/synthetic_{task}_split.csv"), dtype=str)[cols]; ex.to_csv(os.path.join(d, "split_test.csv"), index=False)
        ok, _ = run(["scripts/extra_baselines.py", "--task", task, "--split-dir", d, "--out-dir", os.path.join(d, "out")])
        if not ok:
            check(f"extra_baselines.py {task}", False); continue
        rd = lambda m, s="test": pd.read_csv(os.path.join(d, "out", f"pred_{task}_{m}_{s}.csv"))
        if task == "warfarin":
            med = float(pd.read_csv(os.path.join(d, "split_train.csv")).final_dose.median())
            ok = (np.allclose(rd("Physician").y_pred, ex.initial_dose.astype(float)) and np.allclose(rd("Constant").y_pred, med)
                  and True)
        else:
            p = rd("LogReg").filter(like="prob_").sum(1)
            ok = len(rd("LogReg")) == len(ex) and np.allclose(p, 1) and len(rd("LogReg", "valid")) == 50
        check(f"extra_baselines.py {task} (physician = initial dose, constant = train median / LR probabilities)", ok)


def t_fig3(tmp):
    """figures/make_fig3_dose_distribution.py on the synthetic warfarin rows (the other figure scripts need the authors' result tree)."""
    d = os.path.join(tmp, "fig3"); os.makedirs(d, exist_ok=True)
    ex = pd.read_csv(os.path.join(ROOT, "examples/synthetic_warfarin_split.csv"))
    for s in ("train", "valid", "test"):
        ex.to_csv(os.path.join(d, f"split_{s}.csv"), index=False)
    out = os.path.join(d, "fig3.png")
    ok, _ = run(["figures/make_fig3_dose_distribution.py", "--split-dir", d, "--out", out, "--n-expected", "0"])
    check("figures/make_fig3_dose_distribution.py (synthetic doses)", ok and os.path.getsize(out) > 0)


def t_figs(tmp):
    rng = np.random.default_rng(3); d = os.path.join(tmp, "figs"); os.makedirs(d, exist_ok=True); args = []
    for name, noise in (("ML", 1.0), ("LLM", 1.5)):
        for split, n in (("test", 150), ("valid", 100)):
            y = np.random.default_rng(7 if split == "test" else 8).integers(0, 2, n)
            p = 1 / (1 + np.exp(-(1.0 * (2 * y - 1) + rng.normal(0, noise, n) - 1.5)))
            pd.DataFrame({"y_true": np.where(y == 1, "positive", "negative"), "y_pred": np.where(p >= 0.5, "positive", "negative"),
                          "prob_positive": p, "prob_negative": 1 - p}).to_csv(os.path.join(d, f"{name}_{split}.csv"), index=False)
        args += ["--model", f"{name}={d}/{name}_test.csv,{d}/{name}_valid.csv"]
    ok, _ = run(["figures/make_final_figures.py", "--task", "mace", "--out-dir", d, "--dca-range", "0.10", "0.30", "--dca-steps", "0.1", "0.2", "0.3"] + args)
    check("figures/make_final_figures.py (DCA, reliability, confusion)", ok and all(os.path.getsize(os.path.join(d, f"{k}_mace.png")) > 0 for k in ("DCA", "reliability", "confusion")))
    ev = []
    for sd in ("3407", "1"):
        ok2, _ = run(["scripts/evaluate.py", "--task", "mace", "--reference", "ML"] + args + ["--out", os.path.join(d, f"ev{sd}.json")]); ev += ["--eval", f"{sd}={d}/ev{sd}.json"]
    ok3, _ = run(["figures/make_forest_plots.py", "--reference", "ML", "--metric", "AUROC", "AUPRC", "--out", os.path.join(d, "forest.png")] + ev)
    check("figures/make_forest_plots.py", ok2 and ok3 and os.path.getsize(os.path.join(d, "forest.png")) > 0)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        t_compile()
        t_serialization(tmp)
        t_metrics_known()
        t_evaluate(tmp)
        t_extra(tmp)
        if "--ml" in sys.argv:
            t_ml(tmp)
        t_fig3(tmp)
        t_figs(tmp)
        for s in ("make_jsonl", "train_llm", "infer_llm", "select_lora", "ml_baselines", "evaluate", "warfarin_probe", "extra_baselines",
                  "token_lengths"):
            ok, _ = run([f"scripts/{s}.py", "--help"]); check(f"{s}.py --help (imports without GPU)", ok)
    n_fail = sum(not v for v in RESULTS.values())
    print(json.dumps(dict(n_checks=len(RESULTS), n_failed=n_fail)))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
