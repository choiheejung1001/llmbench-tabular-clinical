"""Token lengths of the training texts (Alpaca prompt + answer + EOS, exactly as fed to the SFT trainer) per tokenizer.

Used for the input-length statements of the field-name analysis (Supplementary Table S20 footnote) and to check that no record
exceeds max_seq_length. CPU only; needs the tokenizers of the base models (Hugging Face ids, or local copies via LLM_BASE_DIR).

  CUDA_VISIBLE_DEVICES="" python scripts/token_lengths.py --task warfarin --naming identifier \
         --jsonl DATA/warfarin/jsonl_identifier/warfarin_reg_{train,valid,test}.jsonl --out lengths.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from llmbench import load_config, MODELS  # noqa: E402
from llmbench.serialize import load_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["warfarin", "mace", "mortality"])
    ap.add_argument("--naming", default="original", choices=["original", "descriptive", "identifier"])
    ap.add_argument("--jsonl", required=True, nargs="+", help="JSONL files pooled into one length distribution")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--max-seq-length", type=int, help="default: configs/llm_training.json for --task/--naming")
    ap.add_argument("--out")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from llmbench import llm as L
    msl = a.max_seq_length or load_config("llm_training")["max_seq_length"][a.task][a.naming]
    texts = L.to_text([r for p in a.jsonl for r in load_jsonl(p)], with_output=True)
    out = {"task": a.task, "naming": a.naming, "n_records": len(texts), "max_seq_length": msl}
    for m in a.models:
        tok = AutoTokenizer.from_pretrained(L.base_path(m))
        n = [len(tok(t + tok.eos_token)["input_ids"]) for t in texts]
        out[m] = dict(max=max(n), mean=round(sum(n) / len(n), 1), n_over_max_seq=sum(x > msl for x in n))
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
