"""Inference with a saved LoRA adapter (4-bit base + adapter, not merged).

Modes
  auto        classification task -> label log-probability scoring; regression task -> greedy decoding + first number
  candidates  warfarin candidate-dose scoring: log P(candidate + EOS | prompt) for every distinct output string of --train-jsonl;
              writes y_argmax (most likely candidate) and y_expected (probability-weighted dose), plus the log-probability matrix (.npy)
  features    final-layer hidden state of the last prompt token for --jsonl (input of scripts/warfarin_probe.py)

  python scripts/infer_llm.py --task mortality --model qwen2.5-7B --adapter RUNS/mortality/adapters/<run> --jsonl DATA/.../mortality_cls_test.jsonl --out preds.csv
  python scripts/infer_llm.py --task warfarin --model qwen2.5-3B --adapter <adapter> --jsonl <test.jsonl> --mode candidates \
         --train-jsonl <train.jsonl> --out scores.csv
Requires one CUDA GPU.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from llmbench import load_config, MODELS  # noqa: E402
from llmbench.serialize import task_spec, load_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["warfarin", "mace", "mortality"])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True, help=".csv (auto, candidates) or .npy (features)")
    ap.add_argument("--mode", default="auto", choices=["auto", "candidates", "features"])
    ap.add_argument("--naming", default="original", choices=["original", "descriptive", "identifier"])
    ap.add_argument("--max-seq-length", type=int)
    ap.add_argument("--train-jsonl", help="candidates mode: training JSONL whose distinct outputs form the candidate set")
    a = ap.parse_args()

    from llmbench import llm as L
    T = task_spec(a.task)
    msl = a.max_seq_length or load_config("llm_training")["max_seq_length"][a.task][a.naming]
    rows = load_jsonl(a.jsonl)
    model, tok = L.load_with_adapter(a.model, a.adapter, msl)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.mode == "auto":
        df = L.infer_classification(model, tok, rows, T["labels"]) if T["type"] == "classification" else L.infer_greedy(model, tok, rows)
        df.to_csv(a.out, index=False)
        print(f"{len(df)} predictions -> {a.out}")
    elif a.mode == "candidates":
        assert T["type"] == "regression" and a.train_jsonl, "candidates mode needs a regression task and --train-jsonl"
        cands = sorted(set(r["output"] for r in load_jsonl(a.train_jsonl)), key=float)
        df, LP = L.score_candidates(model, tok, rows, cands)
        df.to_csv(a.out, index=False)
        np.save(a.out.replace(".csv", "_logprob.npy"), LP)
        json.dump(dict(candidates=cands), open(a.out.replace(".csv", "_candidates.json"), "w"))
        print(f"{len(df)} records x {len(cands)} candidates -> {a.out}")
    else:
        X = L.last_token_features(model, tok, rows)
        np.save(a.out, X)
        print(f"features {X.shape} -> {a.out}")


if __name__ == "__main__":
    main()
