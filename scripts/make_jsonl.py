"""Build the LLM JSONL files (instruction / serialized input / output) from split CSVs.

Example:
  python scripts/make_jsonl.py --task warfarin --split-dir DATA/warfarin --out-dir DATA/warfarin/jsonl --naming original
  python scripts/make_jsonl.py --task mortality --split-dir DATA/mortality --out-dir DATA/mortality/jsonl_identifier --naming identifier

Input : <split-dir>/split_{train,valid,test}.csv with the columns of naming/<task>_keys.csv (csv_column) and the target column.
Output: <out-dir>/<prefix>_{train,valid,test}.jsonl (prefix warfarin_reg | mace_cls | mortality_cls). Only counts are printed.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from llmbench.serialize import task_spec, read_split, build_records, write_jsonl, apply_cleaning  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["warfarin", "mace", "mortality"])
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--naming", default="original", choices=["original", "descriptive", "identifier"])
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--split-file-pattern", default="split_{split}.csv")
    ap.add_argument("--exclude", default="", help="comma-separated csv columns to leave out (mortality 49-variable analysis: age_at_hct)")
    a = ap.parse_args()
    T = task_spec(a.task)
    for s in a.splits.split(","):
        df = read_split(os.path.join(a.split_dir, a.split_file_pattern.format(split=s)))
        _, cleaned = apply_cleaning(df, a.task)
        recs = build_records(df, a.task, naming=a.naming, exclude=tuple(c for c in a.exclude.split(",") if c))
        out = os.path.join(a.out_dir, f"{T['jsonl_prefix']}_{s}.jsonl")
        write_jsonl(recs, out)
        print(f"{s}: {len(recs)} records -> {out}" + (f" (set to missing: {cleaned})" if cleaned else ""))


if __name__ == "__main__":
    main()
