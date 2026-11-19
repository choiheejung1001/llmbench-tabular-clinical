# LLMbench code (v3): fine-tuned LLMs vs machine-learning models on tabular clinical prediction

Code accompanying the article *Task-specific performance of locally fine-tuned large language models for tabular clinical
prediction: a controlled comparison with machine learning models* (Scientific Reports, under revision).

## 1. Purpose

The study fine-tunes four open-weight LLMs (Qwen2.5-3B/7B/14B-Instruct and Llama-3.1-8B-Instruct, 4-bit, Unsloth + LoRA) on
key-value serialized tabular records and compares them with machine-learning baselines of the original studies on three tasks:

| Task | Type | LLM output | ML baseline (original-study model, original settings, seed 3407, chosen on validation) |
|---|---|---|---|
| Warfarin dose at discharge | regression (mg) | generated number (greedy decoding) | XGBoost regressor (candidates: ANN, random forest, linear regression; lowest validation MAE) |
| One-year MACE after myocardial infarction | binary | label probability (`positive` / `negative`) | LightGBM (candidates: logistic regression, random forest, feed-forward network; highest validation AUROC) |
| Five-year mortality after allogeneic HCT | binary | label probability (`dead` / `alive`) | AdaBoost, SAMME.R, on target-encoded karyotype (candidates: CatBoost, random forest, MLP, logistic regression; highest validation AUROC) |

The ML baselines are the models of the original studies refitted with their original settings and **no retuning**
(`configs/ml_baselines.json`). Mortality compares two karyotype representations (target-encoded text `TE` vs `Structured` binary flags)
over the five models. Every LLM seed (3407, 1, 2) is compared with the same fixed baseline predictions.

This package contains what is needed to apply the same pipeline to one's own data: serialization and instruction templates,
LoRA training and inference, ML baselines at their original settings with validation selection, and the evaluation code. It is a minimal, cleaned
version of the analysis code; experiment bookkeeping (job queues, internal checks, manuscript builds) is not included.

## 2. Data availability

**No patient data are included.** The patient-level data of the three cohorts cannot be shared publicly; they are available
from the corresponding author upon reasonable request and with the approval required by the data owners. The fine-tuned LoRA
adapters are not shared, because they were trained on patient-level data. Predictions, logs and intermediate results are also
not included. The files in `examples/` are **entirely synthetic**: every value was invented by hand to show the file format;
categorical code values (for example diagnosis or HLA fields) are illustrative only and do not follow the coding of the original data.

## 3. Structure

```
LLMbench_code_v3/
├── README.md, LICENSE (MIT), CITATION.cff, .zenodo.json, requirements.txt
├── configs/
│   ├── tasks.json               instructions, label words, field order source, missing token, cleaning rules, max_seq per naming condition
│   ├── llm_training.json        common LoRA/SFT settings, inference settings, r/alpha search grid and selection rule (rule A),
│   │                            seeds, adopted r/alpha per task x model
│   └── ml_baselines.json        original settings of the ML baselines, seed 3407, candidate models, selection metric
├── naming/                      name-only key maps (no values): csv_column -> original / descriptive / identifier (V1..Vk) key
│   ├── warfarin_keys.csv, mace_keys.csv, mortality_keys.csv
├── llmbench/                    library
│   ├── serialize.py             row -> "Name:value, ..." serialization, cleaning rules, JSONL I/O
│   ├── karyotype.py             rule-based ISCN karyotype parser (structured karyotype flags)
│   ├── llm.py                   Unsloth model loading, LoRA, SFT trainer, label log-prob scoring, greedy decoding, candidate scoring, features
│   └── metrics.py               metrics, paired bootstrap, recalibration, decision curves, operating thresholds
├── scripts/                     command-line entry points (all have --help)
│   ├── make_jsonl.py            split CSV -> LLM JSONL (naming condition original | descriptive | identifier)
│   ├── train_llm.py             LoRA fine-tuning of one model, then validation + test predictions        [GPU]
│   ├── infer_llm.py             inference with a saved adapter: label scoring / greedy / candidate scoring / features  [GPU]
│   ├── select_lora.py           rule-A selection of r/alpha from validation predictions
│   ├── ml_baselines.py          original-setting ML baselines (3 tasks; mortality: TE and Structured), validation selection of the baseline
│   ├── evaluate.py              test metrics with CIs, paired differences, recalibration, thresholds, decision curves
│   ├── warfarin_probe.py        ridge probe on LLM hidden states (supplementary)
│   ├── extra_baselines.py       extra table rows: warfarin physicians / constant dose; MACE and mortality logistic regression
│   └── token_lengths.py         token lengths of the training texts per tokenizer (field-name analysis, max_seq check)
├── figures/                     drawing code (data-free; run on your own prediction files)
│   ├── make_fig3_dose_distribution.py   Fig. 2: warfarin discharge-dose distribution (runs on the split tables)
│   ├── make_forest_plots.py             forest plots of paired differences per LLM and seed (Figs. 3-5, Supplementary Figs. S7-S8), from evaluate.py JSON
│   └── make_final_figures.py            decision curves (Fig. 6), reliability diagrams, confusion matrices (Supplementary Figs. S2-S6), from prediction CSVs
├── examples/                    SYNTHETIC example inputs (3 invented patients per task)
│   ├── synthetic_<task>_split.csv          tabular CSV schema used by the ML models and by make_jsonl.py
│   └── synthetic_<task>_<naming>.jsonl     JSONL used for the LLMs (generated from the CSV by make_jsonl.py)
└── tests/smoke_test.py          CPU smoke test (no GPU, no data)
```

## 4. Environment

Python 3.10; `pip install -r requirements.txt`. LLM training and inference need one CUDA GPU (the study used NVIDIA RTX 3090,
24 GB; the 14B model fits in 4-bit). The ML and evaluation scripts run on CPU; run them with `CUDA_VISIBLE_DEVICES=""`.

Base models: by default the 4-bit checkpoints are downloaded from Hugging Face (`unsloth/Qwen2.5-3B-Instruct-bnb-4bit`,
`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`, `unsloth/Qwen2.5-14B-Instruct-bnb-4bit`, `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`).
To use local copies set `LLM_BASE_DIR=/path/to/checkpoints` (the folder names must equal the last part of these ids).

## 5. Running the pipeline on your own data

**Step 0 - split tables.** For each task provide `split_train.csv`, `split_valid.csv`, `split_test.csv` with the columns of
`naming/<task>_keys.csv` (`csv_column`) and the target (`final_dose` for warfarin; `event` with the label words for MACE and
mortality), as in `examples/synthetic_<task>_split.csv`. The study used one stratified split (test 20%, then 20% of the rest for
validation, random state 42). The validation split is used only for selection (LoRA r/alpha, choice of the ML baseline, recalibration,
thresholds); every model is fitted on the training split only.

**Step 1 - serialization.**
```
python scripts/make_jsonl.py --task warfarin --split-dir DATA/warfarin --out-dir DATA/warfarin/jsonl
python scripts/make_jsonl.py --task mortality --split-dir DATA/mortality --out-dir DATA/mortality/jsonl_descriptive --naming descriptive
```
Each record becomes `Name:value, Name:value, ...` in the field order of the key map; missing values become `Unknown`.
Warfarin: height 0 cm and blood pressure <= 20 mmHg are set to missing first; Gender 1/2 is written as Male/Female.
Mortality: recipient age is serialized first as `Age:<integer>` (`--exclude age_at_hct` gives the 49-variable sensitivity
input). Naming conditions (field-name analysis): `original`;
`descriptive` (MACE: the six echocardiography names spelled out; mortality: full names); `identifier` (`V1..Vk`).
The instructions are in `configs/tasks.json`; the prompt is the Alpaca template in `llmbench/llm.py`.

**Step 2 - LoRA r/alpha selection (seed 3407).** Train the primary configuration and the five candidates of
`configs/llm_training.json` (`lora_search`) for every model, then select:
```
python scripts/train_llm.py --task mace --model qwen2.5-7B --data-dir DATA/mace/jsonl --out-dir RUNS/mace --r 16 --alpha 32 --seed 3407
...   (6 configurations x 4 models)
python scripts/select_lora.py --task mace --pred-dir RUNS/mace/preds --out RUNS/mace/adopted.json
```
Rule A: the configuration with the best validation metric (AUROC; warfarin greedy MAE) is adopted, the primary is kept on ties.
Copy the adopted r/alpha into `configs/llm_training.json` (`adopted`) to use `--adopted` below (the values adopted in the
study are already filled in for the three tasks).

**Step 3 - final runs.** Seeds 3407, 1 and 2 at the adopted configuration (seed 3407 is the selection run):
```
python scripts/train_llm.py --task mace --model qwen2.5-7B --data-dir DATA/mace/jsonl --out-dir RUNS/mace --adopted --seed 1
```
Training settings (all runs): LoRA dropout 0 on q/k/v/o/gate/up/down, 3 epochs, batch 2 x gradient accumulation 4, learning rate
2e-4 (linear, 5 warm-up steps), AdamW-8bit, weight decay 0.01, fp16, last-epoch adapter, max_seq 384 (warfarin), 512 (MACE),
1024 (mortality; 1280 with descriptive names). Inference: 4-bit base + adapter (not merged); classification probabilities from the
label log-probabilities; warfarin = greedy decoding (max 8 new tokens) and the first number of the output.

**Step 4 - other inference modes** (`scripts/infer_llm.py`): `--mode auto` re-predicts any JSONL with a saved adapter;
`--mode candidates` scores every distinct training dose string for warfarin (most likely candidate and expected dose);
`--mode features` extracts hidden states for `scripts/warfarin_probe.py`.

**Step 5 - ML baselines (original settings, no tuning).**
```
CUDA_VISIBLE_DEVICES="" python scripts/ml_baselines.py fit --task warfarin  --split-dir DATA/warfarin  --out-dir RUNS/ml/warfarin
CUDA_VISIBLE_DEVICES="" python scripts/ml_baselines.py fit --task mace      --split-dir DATA/mace      --out-dir RUNS/ml/mace
CUDA_VISIBLE_DEVICES="" python scripts/ml_baselines.py fit --task mortality --split-dir DATA/mortality --out-dir RUNS/ml/mortality
```
Every candidate model is fitted on the training split with seed 3407; `selection.json` lists the validation metric of each candidate
(warfarin: MAE of the 0.5-mg-rounded prediction; MACE, mortality: AUROC, mortality on the TE representation) and the selected baseline.
Preprocessing follows the original studies: warfarin missing -> -1 (XGBoost: native missing), ANN on MinMax inputs; MACE no imputation,
missing-value indicators and missing set to 0, no scaling; mortality target encoding then mean imputation (CatBoost: second target
encoding). SAMME.R is re-implemented because scikit-learn 1.6 removed it. Extra rows of the main tables (no tuning):
```
CUDA_VISIBLE_DEVICES="" python scripts/extra_baselines.py --task warfarin --split-dir DATA/warfarin --out-dir RUNS/extra/warfarin
```
Warfarin: physicians' initial dose and the constant training-median dose. MACE and mortality: unpenalised logistic regression with mean
imputation and standardisation (mortality: target encoding first).

**Step 6 - evaluation (every LLM seed against the same fixed baseline predictions).**
```
CUDA_VISIBLE_DEVICES="" python scripts/evaluate.py --task mortality --reference AdaBoost \
   --model AdaBoost=RUNS/ml/mortality/preds/pred_mortality_AdaBoost_TE_seed3407_test.csv,RUNS/ml/mortality/preds/pred_mortality_AdaBoost_TE_seed3407_valid.csv \
   --model Qwen2.5-7B=RUNS/mortality/preds/<llm_test>.csv,RUNS/mortality/preds/<llm_valid>.csv \
   --dca-range 0.30 0.70 --dca-steps 0.30 0.50 0.70 --out RUNS/eval_mortality_seed3407.json
```
Reports AUROC, AUPRC, confidence-based ECE, Brier score, calibration slope and intercept with 95% CIs (simple paired bootstrap, B = 1000,
seed 2026, the same resamples for every model), differences vs the reference with CI and bootstrap p, Platt and temperature
recalibration fitted on validation, operating thresholds chosen on validation (Youden J as the primary point; sensitivity >= 0.80 and
F1-max as robustness; 0.5 for reference) and applied unchanged to test, and decision-curve net benefit. Decision-curve threshold ranges
are exploratory (no clinical-range claim). For warfarin: MAE and % within +-20% of the actual dose. Seed-averaged differences:
`llmbench.metrics.seed_mean_delta`. Validation-set predictions of any LLM run (needed for Sens./Spec. at the validation threshold,
also in the field-name experiments): `scripts/infer_llm.py --jsonl <valid.jsonl>`.

**Step 7 - figures.**
```
python figures/make_forest_plots.py --reference LightGBM --metric AUROC AUPRC --out Fig4.png --eval 3407=eval_seed3407.json --eval 1=eval_seed1.json --eval 2=eval_seed2.json
python figures/make_final_figures.py --task mace --out-dir figs --model LightGBM=ml_test.csv,ml_valid.csv --model Qwen2.5-7B=llm_test.csv,llm_valid.csv --dca-range 0.10 0.30
```

## 6. Final study settings encoded here

* **Warfarin.** Instruction "Predict the warfarin dose (mg) that will be prescribed at discharge, based on the patient's clinical
  information at admission."; 17 inputs; height 0 cm and blood pressure <= 20 mmHg treated as missing in both model types; greedy
  decoding; adopted LoRA r/alpha per model (`configs/llm_training.json`); ML baseline = original-setting XGBoost; supplementary
  candidate-dose scoring, regression head (ridge probe on hidden states) and identifier-name condition.
* **MACE.** Outcome: all-cause death, recurrent myocardial infarction, or repeat coronary revascularization (therapeutic PCI or
  CABG) within one year after discharge; cohort restricted to index discharges with a complete one-year observation window,
  in-hospital moribund discharges excluded, age in completed years at admission; 34 inputs; ML baseline = LightGBM at the original
  settings. The cohort extraction from the hospital records is not part of this package.
* **Mortality.** 50 inputs (49 variables of the original cohort study + recipient age at transplantation); instruction "Predict
  whether the following patient with a hematologic malignancy who underwent allogeneic hematopoietic stem cell transplantation will
  be alive or dead five years after transplantation. Answer 'alive' or 'dead'."; ML baseline = AdaBoost (SAMME.R) on the TE representation,
  chosen on validation among five original-setting models; TE vs Structured karyotype compared over the five models.
* **All tasks.** LoRA r/alpha selection over six candidates with rule A; seeds 3407, 1, 2; field-name conditions; paired simple
  bootstrap (B = 1000, seed 2026); confidence-based ECE; recalibration; exploratory decision curves; validation-chosen thresholds.

## 7. Code-to-manuscript correspondence

Display-item labels follow the revised manuscript (main text) and Supplementary Information.

| Manuscript element | Code in this package |
|---|---|
| Table 1 (instructions, label words) | `configs/tasks.json`, `llmbench/serialize.py`, `scripts/make_jsonl.py` |
| Fig. 2 (warfarin discharge doses) | `figures/make_fig3_dose_distribution.py` |
| Table 2 (warfarin MAE, overall and by dose range; physicians, constant, original-study models vs XGBoost) | `scripts/train_llm.py` (greedy), `scripts/ml_baselines.py`, `scripts/extra_baselines.py`, `scripts/evaluate.py` (IWPC row and dose bands: Section 8) |
| Table 3 (MACE classification; LightGBM baseline) | `scripts/train_llm.py`, `scripts/ml_baselines.py`, `scripts/extra_baselines.py` (logistic regression), `scripts/evaluate.py` (Sens./Spec. at the validation Youden threshold) |
| Table 4 (mortality classification; AdaBoost-TE baseline) | same, `--task mortality` |
| Figs. 3-5 (per-seed paired differences vs the fixed baseline; field-name conditions) | `scripts/evaluate.py` per seed, then `figures/make_forest_plots.py` |
| Fig. 6 (decision curves, exploratory threshold ranges) | `scripts/evaluate.py --dca-range --dca-steps`; `figures/make_final_figures.py` |
| Supplementary Table S1, S4 (baseline characteristics, outcome classes) | not reproduced (Section 8) |
| Supplementary Table S2 (structured karyotype variables) | `llmbench/karyotype.py`, `scripts/ml_baselines.py --representation Structured` |
| Supplementary Table S3, S6 (training settings, LoRA r/alpha selection) | `configs/llm_training.json`, `scripts/select_lora.py`, `scripts/train_llm.py` |
| Supplementary Table S5 (original-study models refitted; (a) warfarin, (b) MACE, (c) mortality TE vs Structured) | `configs/ml_baselines.json`, `scripts/ml_baselines.py`, `scripts/evaluate.py` |
| Supplementary Table S7 (three seeds, paired differences) | `scripts/train_llm.py --seed`, `scripts/evaluate.py`, `llmbench.metrics.seed_mean_delta` |
| Supplementary Table S8 (training time, memory, latency) | run metadata written by `scripts/train_llm.py` |
| Supplementary Tables S9, S10 (subgroups, calendar period) | `scripts/evaluate.py` on the rows of each subgroup (Section 8) |
| Supplementary Table S11 (field-name analysis, classification) | `scripts/make_jsonl.py --naming`, `naming/*.csv`, `scripts/train_llm.py --naming`, `scripts/infer_llm.py --jsonl <valid.jsonl>`, `scripts/evaluate.py`, `scripts/token_lengths.py` |
| Supplementary Table S12 (field-name analysis, warfarin; candidate scoring, regression head) | `scripts/make_jsonl.py --naming identifier`, `scripts/infer_llm.py --mode candidates / features`, `scripts/warfarin_probe.py` |
| Supplementary Tables S13, S14 (secondary metrics, confusion matrices at the validation Youden threshold) | `scripts/evaluate.py` |
| Supplementary Table S15 (decision curve net benefit) | `scripts/evaluate.py` |
| Supplementary Tables S16, S17 (calibration, recalibration) | `scripts/evaluate.py`, `llmbench/metrics.py` |
| Supplementary Figs. S2-S6 (reliability, confusion matrices), S7-S8 (ECE / Brier differences) | `figures/make_final_figures.py`, `figures/make_forest_plots.py --metric ECE Brier` |

The figure scripts are data-free: they draw from prediction CSVs or `evaluate.py` JSON files that you create from your own data.

## 8. Numbers not reproduced by this package

* **Table 2, IWPC clinical-algorithm row**: the algorithm (Klein et al. 2009, no genotype) needs an admission enzyme-inducer variable
  that is not in the split tables.
* **Table 2, dose-range columns**: `scripts/evaluate.py` reports overall values only.
* **MACE cohort extraction** from the hospital records (Section 6) and the **descriptive tables** (Supplementary Tables S1, S4), the
  subgroup and calendar-period selections (Supplementary Tables S9, S10) and the discharge-INR analysis (Supplementary Table S18).
* **The authors' study-specific data splits and fine-tuned adapters**: not distributed (Section 2).

## 9. Smoke test

```
CUDA_VISIBLE_DEVICES="" python tests/smoke_test.py          # compile, serialization of the synthetic examples, evaluation code, figures
CUDA_VISIBLE_DEVICES="" python tests/smoke_test.py --ml     # + ML baseline fits on randomly generated tables
```

## 10. Notes on this release

* Entry points were written for this release from the authors' analysis code; definitions, settings and selection rules are
  unchanged. File names of outputs differ from the internal ones.
* Version 3 replaces the validation-tuned XGBoost/CatBoost baselines and the comparison panel of v2 by the original-setting baselines
  above; the grid search, `ml_zoo.py` and the figure scripts that read an internal result tree were removed.
* The release was scanned for file-system paths, account names, e-mail addresses, tokens, identifiers and institution names.
* The scripts read split CSVs as text and serialize the values as written; numbers are not re-formatted.
* Hard-coded server paths were replaced by command-line arguments and the optional `LLM_BASE_DIR` variable.

## 11. License and citation

MIT (see `LICENSE`). Please cite the article and this archive (`CITATION.cff`). The archive DOI is shown on the Zenodo record of the release.
