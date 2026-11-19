"""Row -> 'Name:value, Name:value, ...' serialization and JSONL construction for the three tasks.

Rules (identical for all tasks):
  * fields are written in the order of naming/<task>_keys.csv (column `position`);
  * key text = original_key | descriptive_key | identifier_key of that file (naming condition);
  * a missing value is written as 'Unknown'; every other value is written as its text in the split CSV
    (the CSV is read with dtype=str so numbers keep their text), after the task value maps (warfarin Gender 1/2 -> Male/Female);
  * fields are joined with ', ' and each field is '<key>:<value>'.
Task specifics:
  * warfarin: implausible values are set to missing first (Height == 0 cm; systolic / diastolic BP <= 20 mmHg);
    output = str(float(final_dose)).
  * mortality: the age field (CSV column age_at_hct) is serialized first as an integer ('Age:57').
  * classification outputs are the label words of the `event` column (MACE positive/negative, mortality dead/alive).
JSONL lines: {"instruction": ..., "input": ..., "output": ...} written with json.dumps(ensure_ascii=False).
"""
import json
import os

import numpy as np
import pandas as pd

from . import PKG_ROOT, load_config

NAMING_COLUMN = {"original": "original_key", "descriptive": "descriptive_key", "identifier": "identifier_key"}


def task_spec(task):
    T = load_config("tasks")[task]
    keys = pd.read_csv(os.path.join(PKG_ROOT, T["key_map"])).sort_values("position")
    T = dict(T)
    T["keys"] = keys.reset_index(drop=True)
    return T


def read_split(path):
    """Split CSV as text (dtype=str) so that values are serialized exactly as written."""
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def _cmp(op, v, thr):
    return {"==": v == thr, "<=": v <= thr, "<": v < thr, ">=": v >= thr, ">": v > thr}[op]


def apply_cleaning(df, task):
    """Set implausible values to missing (warfarin rules in configs/tasks.json); returns a copy and the count per column."""
    T = load_config("tasks")[task]
    out = df.copy(); counts = {}
    for col, rule in T.get("cleaning_rules", {}).items():
        v = pd.to_numeric(out[col], errors="coerce")
        m = _cmp(rule["op"], v, rule["value"]).fillna(False).values.astype(bool)
        out.loc[m, col] = np.nan
        counts[col] = int(m.sum())
    return out, counts


def _value_text(T, col, v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return T["missing_token"]
    s = str(v)
    vm = T.get("value_maps", {}).get(col)
    if vm:
        s = vm.get(s, s)
    if col in T.get("integer_columns", []):
        s = str(int(float(s)))
    return s


def serialize_row(row, T, naming="original"):
    kcol = NAMING_COLUMN[naming]
    parts = []
    for _, k in T["keys"].iterrows():
        parts.append(f"{k[kcol]}:{_value_text(T, k['csv_column'], row[k['csv_column']])}")
    return ", ".join(parts)


def output_text(row, T):
    v = row[T["target_column"]]
    if T["type"] == "regression":
        return f"{float(v)}"
    return str(v)


def build_records(df, task, naming="original", clean=True, exclude=()):
    """DataFrame (split CSV read as text) -> list of {"instruction", "input", "output"} dicts.
    exclude: csv columns left out of the input (e.g. ('age_at_hct',) for the 49-variable mortality sensitivity analysis)."""
    T = task_spec(task)
    if exclude:
        T["keys"] = T["keys"][~T["keys"].csv_column.isin(exclude)].reset_index(drop=True)
    if naming not in T["naming_conditions"]:
        raise ValueError(f"naming '{naming}' not defined for {task}: {T['naming_conditions']}")
    if clean:
        df, _ = apply_cleaning(df, task)
    return [dict(instruction=T["instruction"], input=serialize_row(r, T, naming), output=output_text(r, T)) for _, r in df.iterrows()]


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def ml_matrix(df, task, clean=True):
    """Feature matrix for the ML baselines (numeric columns coerced; mortality categoricals kept as text)."""
    T = task_spec(task)
    if clean:
        df, _ = apply_cleaning(df, task)
    cols = list(T["keys"].csv_column)
    if task == "warfarin":
        cols = T["ml_features"]
    X = df[cols].copy()
    num = T.get("ml_numeric_columns", cols)
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if T["type"] == "regression":
        y = pd.to_numeric(df[T["target_column"]], errors="coerce").astype(float).values
    else:
        y = (df[T["target_column"]].astype(str) == T["positive_label"]).astype(int).values
    return X, y
