#!/usr/bin/env python
# CARDIO:DE CHA2DS2-VA Qwen3.5 27b LoRA benchmark.
# Score/criteria benchmark. No evidence spans are requested or saved.
# Full-precision/bfloat16 loading, no quantization.
# Deterministic decoding and deterministic runtime settings.

from pathlib import Path
import os

os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import gc
import json
import random
import re

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm
import unsloth
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

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
    / "qwen35_27b_lora"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen3.5-27B"

MAX_SEQ_LENGTH = 16384
MAX_NEW_TOKENS = 160

PARAMETERS = ["C", "H", "A2", "D", "S2", "V", "A"]

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0

PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 2
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 5


# =============================================================================
# 3. Prompt and folds
# =============================================================================

SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")

folds = []
for fold_number in range(1, 4):
    fold = load_from_disk(DATASET_PATH / f"cha2ds2_va_fold_{fold_number}")
    folds.append(fold)

print("Train letters per fold:", [len(fold["train"]) for fold in folds])
print("Test letters per fold:", [len(fold["test"]) for fold in folds])


# =============================================================================
# 4. Train one LoRA adapter per fold and evaluate on the held-out test split
# =============================================================================

all_results = []

for fold_number, fold in enumerate(folds, start=1):
    print("\n" + "=" * 80)
    print(f"Fold {fold_number}")
    print("=" * 80)

    fold_output_dir = OUTPUT_DIR / f"fold_{fold_number}"
    adapter_output_dir = fold_output_dir / "adapter"
    trainer_output_dir = fold_output_dir / "trainer_output"

    fold_output_dir.mkdir(parents=True, exist_ok=True)
    adapter_output_dir.mkdir(parents=True, exist_ok=True)
    trainer_output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    if hasattr(tokenizer, "tokenizer"):
        base_tokenizer = tokenizer.tokenizer
    else:
        base_tokenizer = tokenizer

    EOS_TOKEN = base_tokenizer.eos_token
    EOS_TOKEN_ID = base_tokenizer.eos_token_id

    if EOS_TOKEN is None:
        EOS_TOKEN = "<|im_end|>"
        EOS_TOKEN_ID = base_tokenizer.convert_tokens_to_ids(EOS_TOKEN)

    if EOS_TOKEN_ID is None or EOS_TOKEN_ID < 0:
        raise ValueError(
            f"Could not determine valid EOS token id for EOS token {EOS_TOKEN!r}."
        )

    if getattr(base_tokenizer, "pad_token", None) is None:
        base_tokenizer.pad_token = EOS_TOKEN
        base_tokenizer.pad_token_id = EOS_TOKEN_ID

    print("EOS_TOKEN:", EOS_TOKEN)
    print("EOS_TOKEN_ID:", EOS_TOKEN_ID)

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    train_records = []

    for row in fold["train"]:
        gold_text_criteria = {
            "C": 0,
            "H": 0,
            "D": 0,
            "S2": 0,
            "V": 0,
        }

        for annotation in row["annotations"]:
            if annotation["polarity"] != "POS":
                continue

            if annotation["value"] == "Heart Failure":
                gold_text_criteria["C"] = 1

            elif annotation["value"] == "Hypertension":
                gold_text_criteria["H"] = 1

            elif annotation["value"] == "Diabetes mellitus":
                gold_text_criteria["D"] = 1

            elif annotation["value"] == "Stroke_TIA_ATE":
                gold_text_criteria["S2"] = 2

            elif annotation["value"] == "Vascular disease":
                gold_text_criteria["V"] = 1

        assistant_json = json.dumps(
            gold_text_criteria,
            ensure_ascii=False,
            indent=2,
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Arztbrief:\n\n{row['text']}"}],
            },
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        train_records.append(
            {
                "prompt": prompt_text,
                "completion": assistant_json + EOS_TOKEN,
            }
        )

    train_dataset = Dataset.from_list(train_records)

    training_args = SFTConfig(
        output_dir=str(trainer_output_dir),
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        eos_token=EOS_TOKEN,
	per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        bf16=True,
        fp16=False,
        dataset_num_proc=1,
    )

    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            args=training_args,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            args=training_args,
        )

    trainer.train()

    pd.DataFrame(trainer.state.log_history).to_csv(
        fold_output_dir / "training_log.csv",
        index=False,
    )

    model.save_pretrained(adapter_output_dir)
    tokenizer.save_pretrained(adapter_output_dir)

    FastLanguageModel.for_inference(model)
    model.eval()

    with tqdm(
        total=len(fold["test"]),
        desc=f"Qwen3.5 27B LoRA fold {fold_number}",
        unit="letter",
    ) as progress:
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
                    pad_token_id=EOS_TOKEN_ID,
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

            all_results.append(
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

    del trainer
    del model
    del tokenizer
    del train_dataset

    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# 5. Metrics
# =============================================================================

results_df = pd.DataFrame(all_results)

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
# 6. Save outputs
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

training_config = {
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "warmup_ratio": WARMUP_RATIO,
    "weight_decay": WEIGHT_DECAY,
    "logging_steps": LOGGING_STEPS,
    "optim": "adamw_8bit",
}

with open(OUTPUT_DIR / "training_config.json", "w", encoding="utf-8") as file:
    json.dump(training_config, file, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "pipeline": "qwen35_27b_lora",
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
            "training_config": training_config,
        },
        file,
        ensure_ascii=False,
        indent=2,
    )

print("\nSaved outputs to:")
print(OUTPUT_DIR)

