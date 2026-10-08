#!/usr/bin/env python
# CARDIO:DE CHA2DS2-VA Qwen3.5 9B zero-shot benchmark.
# Score/criteria benchmark. No evidence spans are requested or saved.
# Full-precision/bfloat16 loading, no quantization.
# Deterministic decoding and deterministic runtime settings.

from pathlib import Path
import os

# Must be set before CUDA libraries are initialized.
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import random
import re

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm
from unsloth import FastLanguageModel


# =============================================================================
# 1. Reproducibility
# =============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.use_deterministic_algorithms(True, warn_only=True)


# =============================================================================
# 2. Paths and configuration
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = Path("../../../data/cardiode_hf_cv3/cha2ds2_va_CARDIODE400_cv3")

PROMPT_FILE = (
    REPO_ROOT
    / "prompts"
    / "cardiode"
    / "qwen_score_calculation_de.txt"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "runs"
    / "cardiode"
    / "score_benchmark"
    / "qwen35_9b_zero_shot"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen3.5-9B"
MAX_SEQ_LENGTH = 16384
MAX_NEW_TOKENS = 160

PARAMETERS = ["C", "H", "A2", "D", "S2", "V", "A"]


# =============================================================================
# 3. Prompt
# =============================================================================

SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")


# =============================================================================
# 4. Load folds
# =============================================================================

folds = []
for fold_number in range(1, 4):
    fold = load_from_disk(DATASET_PATH / f"cha2ds2_va_fold_{fold_number}")
    folds.append(fold)

print("Test letters per fold:", [len(fold["test"]) for fold in folds])


# =============================================================================
# 5. Load model once
# =============================================================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    load_in_16bit=True
)

print("Loaded model commit:", getattr(model.config, "_commit_hash", None))


FastLanguageModel.for_inference(model)
model.eval()


# =============================================================================
# 6. Run zero-shot inference
# =============================================================================

results = []
total_test_letters = sum(len(fold["test"]) for fold in folds)

with tqdm(total=total_test_letters, desc="Qwen3.5 9B zero-shot", unit="letter") as progress:
    for fold_number, fold in enumerate(folds, start=1):
        for row in fold["test"]:
            text = row["text"]

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Arztbrief:\n\n{text}"}],
                },
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            ).to(model.device)

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            input_length = inputs["input_ids"].shape[-1]
            generated_tokens = generated[0][input_length:]
            response_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            valid_json = True

            try:
                parsed = json.loads(response_text)
            except Exception:
                valid_json = False
                json_match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)

                if json_match is not None:
                    try:
                        parsed = json.loads(json_match.group(0))
                        valid_json = True
                    except Exception:
                        parsed = {}
                else:
                    parsed = {}

            predicted_criteria = {
                "C": 1 if int(parsed.get("C", 0)) > 0 else 0,
                "H": 1 if int(parsed.get("H", 0)) > 0 else 0,
                "A2": 0,
                "D": 1 if int(parsed.get("D", 0)) > 0 else 0,
                "S2": 2 if int(parsed.get("S2", 0)) > 0 else 0,
                "V": 1 if int(parsed.get("V", 0)) > 0 else 0,
                "A": 0,
            }

            lines = text.splitlines()
            if len(lines) >= 3:
                age_match = re.search(r"Alter:\s*(\d+)", lines[2])
                if age_match is not None:
                    age = int(age_match.group(1))
                    if age >= 75:
                        predicted_criteria["A2"] = 2
                        predicted_criteria["A"] = 0
                    elif age >= 65:
                        predicted_criteria["A2"] = 0
                        predicted_criteria["A"] = 1

            gold_criteria = {
                "C": 0,
                "H": 0,
                "A2": 0,
                "D": 0,
                "S2": 0,
                "V": 0,
                "A": 0,
            }

            for annotation in row["annotations"]:
                if annotation["polarity"] != "POS":
                    continue

                if annotation["value"] == "Heart Failure":
                    gold_criteria["C"] = 1

                elif annotation["value"] == "Hypertension":
                    gold_criteria["H"] = 1

                elif annotation["value"] == "Age_ge_75":
                    gold_criteria["A2"] = 2
                    gold_criteria["A"] = 0

                elif annotation["value"] == "Age_65_74":
                    if gold_criteria["A2"] == 0:
                        gold_criteria["A"] = 1

                elif annotation["value"] == "Diabetes mellitus":
                    gold_criteria["D"] = 1

                elif annotation["value"] == "Stroke_TIA_ATE":
                    gold_criteria["S2"] = 2

                elif annotation["value"] == "Vascular disease":
                    gold_criteria["V"] = 1

            predicted_score = sum(predicted_criteria.values())
            gold_score = sum(gold_criteria.values())

            results.append(
                {
                    "fold": fold_number,
                    "letter_id": row["letter_id"],
                    "valid_json": valid_json,
                    "raw_response": response_text,
                    "gold_score": gold_score,
                    "predicted_score": predicted_score,
                    "gold_criteria": gold_criteria,
                    "predicted_criteria": predicted_criteria,
                }
            )

            progress.update(1)

results_df = pd.DataFrame(results)


# =============================================================================
# 7. Metrics
# =============================================================================

results_df["score_correct"] = results_df["predicted_score"] == results_df["gold_score"]

results_df["criteria_correct"] = [
    predicted == gold
    for predicted, gold in zip(results_df["predicted_criteria"], results_df["gold_criteria"])
]

results_df["absolute_error"] = (
    pd.to_numeric(results_df["predicted_score"], errors="coerce")
    - results_df["gold_score"]
).abs()

fold_summary = (
    results_df.groupby("fold")
    .agg(
        n_letters=("letter_id", "count"),
        valid_json_rate=("valid_json", "mean"),
        score_accuracy=("score_correct", "mean"),
        criteria_accuracy=("criteria_correct", "mean"),
        score_mae=("absolute_error", "mean"),
    )
    .reset_index()
)

overall_summary = pd.DataFrame(
    {
        "n_letters": [len(results_df)],
        "valid_json_rate": [results_df["valid_json"].mean()],
        "score_accuracy": [results_df["score_correct"].mean()],
        "criteria_accuracy": [results_df["criteria_correct"].mean()],
        "score_mae": [results_df["absolute_error"].mean()],
    }
)

y_true = []
y_pred = []

for _, row in results_df.iterrows():
    gold = row["gold_criteria"]
    predicted = row["predicted_criteria"]

    y_true.append([int(gold[parameter] > 0) for parameter in PARAMETERS])
    y_pred.append([int(predicted.get(parameter, 0) > 0) for parameter in PARAMETERS])

classification_report_dict = classification_report(
    y_true,
    y_pred,
    target_names=PARAMETERS,
    output_dict=True,
    zero_division=0,
)

classification_report_df = pd.DataFrame(classification_report_dict).T.round(4)

micro_f1 = f1_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0,
)

print("\nFold summary:")
print(fold_summary.round(4).to_string(index=False))

print("\nOverall summary:")
print(overall_summary.round(4).to_string(index=False))

print(f"\nPooled micro-F1: {micro_f1:.4f}")
print(classification_report_df.to_string())


# =============================================================================
# 8. Save outputs
# =============================================================================

results_df.to_json(
    OUTPUT_DIR / "test_results.json",
    orient="records",
    force_ascii=False,
    indent=2,
)

fold_summary.to_csv(OUTPUT_DIR / "fold_summary.csv", index=False)
overall_summary.to_csv(OUTPUT_DIR / "overall_summary.csv", index=False)
classification_report_df.to_csv(OUTPUT_DIR / "classification_report.csv")

with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "pipeline": "qwen35_9b_zero_shot",
            "task": "score_criteria_extraction",
            "evidence_output": False,
            "model_name": MODEL_NAME,
            "dataset_path": str(DATASET_PATH),
            "prompt_file": str(PROMPT_FILE),
            "max_seq_length": MAX_SEQ_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "dtype": "torch.bfloat16",
            "load_in_4bit": False,
            "seed": SEED,
            "do_sample": False,
            "num_beams": 1,
            "deterministic_algorithms": True,
            "parameters": PARAMETERS,
        },
        file,
        ensure_ascii=False,
        indent=2,
    )

print("\nSaved outputs to:")
print(OUTPUT_DIR)

