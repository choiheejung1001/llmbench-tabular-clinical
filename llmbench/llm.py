"""Unsloth / LoRA helpers: model loading, LoRA set-up, SFT trainer, and the three inference modes.

  * infer_classification : label log-probability scoring (sum of log P(label tokens | prompt), softmax over the labels)
  * infer_greedy         : greedy decoding (do_sample=False, max_new_tokens=8) and first-number parsing (regression)
  * candidate_logprobs   : log P(candidate + EOS | prompt) for a fixed candidate set (warfarin candidate-dose scoring)
  * last_token_features  : final-layer hidden state of the last prompt token (warfarin linear probe)

torch / unsloth / trl are imported inside the functions so that this module can be imported without a GPU.
"""
import os
import re
import gc

import numpy as np
import pandas as pd

from . import load_config

ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)
GREEDY_KW = dict(do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
NUMBER = re.compile(r"[-+]?\d*\.?\d+")


def prompt_of(r):
    return ALPACA_PROMPT.format(instruction=r["instruction"], input=r["input"], output="")


def to_text(rows, with_output=True):
    return [ALPACA_PROMPT.format(instruction=r["instruction"], input=r["input"], output=(r["output"] if with_output else "")) for r in rows]


def base_path(model_key):
    """Local 4-bit checkpoint under $LLM_BASE_DIR if set, else the Hugging Face id from configs/llm_training.json."""
    base = load_config("llm_training")["models"][model_key]["base"]
    root = os.environ.get("LLM_BASE_DIR")
    return os.path.join(root, os.path.basename(base)) if root else base


def _psutil_guard():
    import builtins
    import psutil
    builtins.psutil = psutil          # some Unsloth versions look up psutil as a global


def load_model(model_key, max_seq_length, load_in_4bit=True):
    _psutil_guard()
    import torch
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(model_name=base_path(model_key), max_seq_length=max_seq_length,
                                                   dtype=torch.float16, load_in_4bit=load_in_4bit, device_map={"": 0})
    return model, tok


def add_lora(model, r, alpha, seed):
    from unsloth import FastLanguageModel
    c = load_config("llm_training")["common_training"]
    return FastLanguageModel.get_peft_model(
        model, r=r, lora_alpha=alpha, lora_dropout=c["lora_dropout"], bias=c["lora_bias"], target_modules=c["target_modules"],
        use_gradient_checkpointing=c["gradient_checkpointing"], random_state=seed, use_rslora=c["use_rslora"], loftq_config=None)


def make_trainer(model, tok, train_text, max_seq_length, seed, output_dir="outputs"):
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    c = load_config("llm_training")["common_training"]
    ds = Dataset.from_dict({"text": [t + tok.eos_token for t in train_text]})
    return SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text", max_seq_length=max_seq_length,
        dataset_num_proc=2, packing=c["packing"],
        args=TrainingArguments(
            per_device_train_batch_size=c["per_device_train_batch_size"], gradient_accumulation_steps=c["gradient_accumulation_steps"],
            warmup_steps=c["warmup_steps"], num_train_epochs=c["epochs"], learning_rate=c["learning_rate"], fp16=c["fp16"], bf16=False,
            logging_steps=c["logging_steps"], optim=c["optimizer"], weight_decay=c["weight_decay"], lr_scheduler_type=c["lr_scheduler_type"],
            seed=seed, output_dir=output_dir, report_to="none", average_tokens_across_devices=False))


def load_with_adapter(model_key, adapter_dir, max_seq_length):
    """4-bit base + saved LoRA adapter (not merged)."""
    from peft import PeftModel
    from unsloth import FastLanguageModel
    model, tok = load_model(model_key, max_seq_length)
    model = PeftModel.from_pretrained(model, adapter_dir)
    FastLanguageModel.for_inference(model)
    return model, tok


# ------------------------------------------------------------------------------------------------ classification
def label_token_logprob(model, tok, prompt, label):
    """Teacher forcing: sum of log P(label tokens | prompt)."""
    import torch
    full_ids = tok(prompt + label, return_tensors="pt").input_ids.to(model.device)
    prompt_len = tok(prompt, return_tensors="pt").input_ids.shape[1]
    with torch.no_grad():
        logits = model(full_ids).logits[0]
    logprobs = torch.log_softmax(logits[:-1], dim=-1)
    targets = full_ids[0, 1:]
    sel = logprobs[range(prompt_len - 1, targets.shape[0]), targets[prompt_len - 1:]]
    return float(sel.sum().item())


def infer_classification(model, tok, rows, labels):
    """DataFrame y_true, y_pred, prob_<label>... (probabilities = softmax of the label log-probabilities)."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    recs = []
    for r in rows:
        p = prompt_of(r)
        lps = np.array([label_token_logprob(model, tok, p, lab) for lab in labels])
        pr = np.exp(lps - lps.max()); pr = pr / pr.sum()
        rec = {"y_true": r["output"], "y_pred": labels[int(pr.argmax())]}
        rec.update({f"prob_{lab}": float(q) for lab, q in zip(labels, pr)})
        recs.append(rec)
    return pd.DataFrame(recs)


# ------------------------------------------------------------------------------------------------ regression
def parse_first_number(text):
    m = NUMBER.search(text)
    return float(m.group()) if m else np.nan


def greedy_one(model, tok, r, max_new_tokens=8):
    ids = tok([prompt_of(r)], return_tensors="pt").to(model.device)
    g = model.generate(**ids, max_new_tokens=max_new_tokens, use_cache=True, **GREEDY_KW)
    gen = tok.batch_decode(g, skip_special_tokens=True)[0].split("### Response:")[-1].strip()
    return parse_first_number(gen), gen


def infer_greedy(model, tok, rows):
    """DataFrame y_true, y_pred (first number of the greedy output; NaN if none), raw."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    out = [greedy_one(model, tok, r) for r in rows]
    return pd.DataFrame({"y_true": [float(r["output"]) for r in rows], "y_pred": [a for a, _ in out], "raw": [b for _, b in out]})


def candidate_logprobs(model, tok, prompt, cands, chunk=64):
    """log P(cand + EOS | prompt) for every candidate, right-padded batches. Scored tokens = tokens whose character span ends after
    len(prompt) (offset mapping). Pads are trailing and excluded from the sum, so no attention mask is needed under causal attention."""
    import torch
    out = []
    for i in range(0, len(cands), chunk):
        seqs, starts = [], []
        for c in cands[i:i + chunk]:
            enc = tok(prompt + c + tok.eos_token, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            starts.append(next(k for k, (a, b) in enumerate(offs) if b > len(prompt))); seqs.append(ids)
        L = max(len(x) for x in seqs); pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long); msk = torch.zeros((len(seqs), L), dtype=torch.float32)
        for k, (x, st) in enumerate(zip(seqs, starts)):
            ids[k, :len(x)] = torch.tensor(x); msk[k, st:len(x)] = 1.0
        ids, msk = ids.to(model.device), msk.to(model.device); lo = min(starts)
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        sub = torch.log_softmax(logits[:, lo - 1:-1, :].float(), dim=-1)
        lp = sub.gather(-1, ids[:, lo:].unsqueeze(-1)).squeeze(-1) * msk[:, lo:]
        out.extend(lp.sum(1).tolist())
        del logits, sub
    return out


def score_candidates(model, tok, rows, cands):
    """Candidate-dose scoring. cands = distinct training target strings. Returns DataFrame y_true, y_argmax, y_expected, p_argmax
    and the (n, K) log-probability matrix."""
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vals = np.array([float(c) for c in cands])
    LP = np.array([candidate_logprobs(model, tok, prompt_of(r), cands) for r in rows])
    pr = np.exp(LP - LP.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    df = pd.DataFrame({"y_true": [float(r["output"]) for r in rows], "y_argmax": vals[LP.argmax(1)], "y_expected": pr @ vals, "p_argmax": pr.max(1)})
    return df, LP


# ------------------------------------------------------------------------------------------------ linear probe
def last_token_features(model, tok, rows):
    """Final-layer hidden state (after the final norm) of the last prompt token, one record at a time -> (n, hidden) float32."""
    import torch
    dec = model.get_base_model().model
    H = []
    with torch.no_grad():
        for r in rows:
            ids = tok([prompt_of(r)], return_tensors="pt").to(model.device)
            o = dec(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"], use_cache=False, return_dict=True)
            H.append(o.last_hidden_state[0, -1].float().cpu().numpy())
    return np.stack(H).astype(np.float32)


def free(*objs):
    import torch
    for o in objs:
        del o
    gc.collect(); torch.cuda.empty_cache()
