"""LoRA fine-tuning of one LLM on one task, then validation and test inference with the saved adapter.

  python scripts/train_llm.py --task mace --model qwen2.5-7B --data-dir DATA/mace/jsonl --out-dir RUNS/mace \
         --r 16 --alpha 32 --seed 3407 [--naming original]
  python scripts/train_llm.py --task warfarin --model qwen2.5-3B --data-dir DATA/warfarin/jsonl --out-dir RUNS/warfarin --adopted --seed 1

Steps (= the runs reported in the paper):
  1. 4-bit base (Unsloth), LoRA r/alpha on q,k,v,o,gate,up,down (dropout 0), get_peft_model(random_state=seed);
  2. SFTTrainer for 3 epochs, batch 2 x accumulation 4, lr 2e-4 linear, warm-up 5, AdamW-8bit, weight decay 0.01, fp16,
     TrainingArguments(seed=seed); training text = Alpaca prompt + answer + EOS; max_seq_length from configs/llm_training.json;
  3. the last-epoch adapter is saved (<out-dir>/adapters/<run>/);
  4. the adapter is reloaded on the 4-bit base (not merged) and predicts the validation and test JSONL:
     classification = label log-probability scoring, regression = greedy decoding + first number.
Outputs: <out-dir>/preds/pred_<run>_{valid,test}.csv, <out-dir>/meta/<run>.json (loss history and run facts, no patient data).
r/alpha: --r/--alpha, or --adopted (configs/llm_training.json adopted entry of the task and model).
Requires one CUDA GPU (set CUDA_VISIBLE_DEVICES).
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from llmbench import load_config, MODELS  # noqa: E402
from llmbench.serialize import task_spec, load_jsonl  # noqa: E402


def resolve_lora(task, model, r, alpha, adopted):
    if not adopted:
        if r is None or alpha is None:
            sys.exit("give --r and --alpha, or --adopted")
        return r, alpha
    ad = load_config("llm_training")["adopted"][task]
    if model not in ad:
        sys.exit(f"no adopted r/alpha for {task} {model} ({ad.get('status')}); give --r/--alpha or run scripts/select_lora.py")
    return int(ad[model]["r"]), int(ad[model]["alpha"])


def run_name(task, model, naming, r, alpha, seed):
    T = task_spec(task)
    tag = "" if naming == "original" else f"_{naming}"
    return f"{T['jsonl_prefix']}_{model}{tag}_r{r}_a{alpha}_seed{seed}"


def predict(task, model, tok, rows):
    from llmbench import llm as L
    T = task_spec(task)
    if T["type"] == "classification":
        return L.infer_classification(model, tok, rows, T["labels"])
    return L.infer_greedy(model, tok, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["warfarin", "mace", "mortality"])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--data-dir", required=True, help="folder with <prefix>_{train,valid,test}.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--naming", default="original", choices=["original", "descriptive", "identifier"],
                    help="naming condition of the JSONL (sets max_seq_length and the run name)")
    ap.add_argument("--r", type=int); ap.add_argument("--alpha", type=int)
    ap.add_argument("--adopted", action="store_true", help="use the adopted r/alpha from configs/llm_training.json")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-seq-length", type=int, help="override configs/llm_training.json")
    ap.add_argument("--skip-valid", action="store_true"); ap.add_argument("--skip-test", action="store_true")
    a = ap.parse_args()
    a.data_dir, a.out_dir = os.path.abspath(a.data_dir), os.path.abspath(a.out_dir)   # resolved before the os.chdir below

    from llmbench import llm as L
    import torch
    T = task_spec(a.task); cfg = load_config("llm_training")
    r, alpha = resolve_lora(a.task, a.model, a.r, a.alpha, a.adopted)
    msl = a.max_seq_length or cfg["max_seq_length"][a.task][a.naming]
    name = run_name(a.task, a.model, a.naming, r, alpha, a.seed)
    P = lambda s: os.path.join(a.data_dir, f"{T['jsonl_prefix']}_{s}.jsonl")
    adapter = os.path.join(a.out_dir, "adapters", name)
    for d in ("preds", "meta", "adapters", f"work/{name}"):
        os.makedirs(os.path.join(a.out_dir, d), exist_ok=True)
    os.chdir(os.path.join(a.out_dir, "work", name))                 # the trainer writes checkpoints to ./outputs

    train_rows = load_jsonl(P("train"))
    model, tok = L.load_model(a.model, msl)
    model = L.add_lora(model, r, alpha, seed=a.seed)
    trainer = L.make_trainer(model, tok, L.to_text(train_rows, with_output=True), max_seq_length=msl, seed=a.seed)
    t0 = time.time(); tr = trainer.train(); train_s = time.time() - t0
    if os.path.exists(adapter):
        shutil.rmtree(adapter)
    model.save_pretrained(adapter); tok.save_pretrained(adapter)    # last-epoch adapter only
    hist = [h for h in trainer.state.log_history if "loss" in h]
    meta = dict(run=name, task=a.task, model=a.model, naming=a.naming, r=r, alpha=alpha, seed=a.seed, max_seq_length=msl,
                n_train=len(train_rows), train_runtime_s=float(tr.metrics.get("train_runtime", train_s)), global_step=int(trainer.state.global_step),
                train_loss=float(tr.training_loss), loss_history=hist)

    def run_split(split, mdl):
        rows = load_jsonl(P(split)); t1 = time.time()
        df = predict(a.task, mdl, tok, rows)
        df.to_csv(os.path.join(a.out_dir, "preds", f"pred_{name}_{split}.csv"), index=False)
        meta[f"{split}_n"] = len(df); meta[f"{split}_infer_s"] = time.time() - t1
        if T["type"] == "regression":
            meta[f"{split}_parse_failures"] = int(df.y_pred.isna().sum())

    # as in the paper's runs: test predictions from the trained model in memory, validation predictions from the reloaded adapter
    if not a.skip_test and os.path.exists(P("test")):
        run_split("test", model)
    del model, trainer; L.free()
    shutil.rmtree("outputs", ignore_errors=True)
    if not a.skip_valid and os.path.exists(P("valid")):
        model, tok = L.load_with_adapter(a.model, adapter, msl)     # 4-bit base + adapter, not merged
        run_split("valid", model)
    meta["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    json.dump(meta, open(os.path.join(a.out_dir, "meta", f"{name}.json"), "w"), indent=1)
    print(f"done {name}: train {meta['train_runtime_s']:.0f}s, loss {meta['train_loss']:.4f}")


if __name__ == "__main__":
    main()
