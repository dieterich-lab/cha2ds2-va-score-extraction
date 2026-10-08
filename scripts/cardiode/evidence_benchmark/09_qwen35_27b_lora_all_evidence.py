#!/usr/bin/env python
# CARDIO:DE CHA2DS2-VA Qwen3.5 27B LoRA evidence benchmark.
# Score/criteria plus all positive evidence spans for text-derived criteria.
# Full-precision/bfloat16 loading, no quantization.
# Deterministic decoding and deterministic runtime settings.

from pathlib import Path
import os

# Must be set before CUDA libraries are initialized.
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import gc
import json
import random
import re
from typing import Any

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
    / "qwen_all_evidence_de.txt"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "runs"
    / "cardiode"
    / "evidence_benchmark"
    / "qwen35_27b_lora_all_evidence"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen3.5-27B"

MAX_SEQ_LENGTH = 16384
MAX_NEW_TOKENS = 2048
IOU_THRESHOLD = 0.50

TEXT_PARAMETERS = ["C", "H", "D", "S2", "V"]
PARAMETERS = ["C", "H", "A2", "D", "S2", "V", "A"]

ANNOTATION_TO_CRITERION = {
    "Heart Failure": "C",
    "Hypertension": "H",
    "Diabetes mellitus": "D",
    "Stroke_TIA_ATE": "S2",
    "Vascular disease": "V",
}

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
# 4. Utility functions: labels, JSON, evidence anchoring and span evaluation
# =============================================================================


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return {}

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}

    return {}


def normalize_binary_score(value: Any, positive_value: int) -> int:
    try:
        return positive_value if int(value) > 0 else 0
    except Exception:
        return 0


def parse_prediction(parsed: dict[str, Any]) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Return model criteria C/H/D/S2/V plus evidence lists."""
    criteria_obj = parsed.get("criteria", {})
    if not isinstance(criteria_obj, dict):
        criteria_obj = {}

    evidence_obj = parsed.get("evidence", {})
    if not isinstance(evidence_obj, dict):
        evidence_obj = {}

    predicted_text_criteria = {
        "C": normalize_binary_score(criteria_obj.get("C", parsed.get("C", 0)), 1),
        "H": normalize_binary_score(criteria_obj.get("H", parsed.get("H", 0)), 1),
        "D": normalize_binary_score(criteria_obj.get("D", parsed.get("D", 0)), 1),
        "S2": normalize_binary_score(criteria_obj.get("S2", parsed.get("S2", 0)), 2),
        "V": normalize_binary_score(criteria_obj.get("V", parsed.get("V", 0)), 1),
    }

    predicted_evidence: dict[str, list[str]] = {criterion: [] for criterion in TEXT_PARAMETERS}

    for criterion in TEXT_PARAMETERS:
        raw_items = evidence_obj.get(criterion, [])

        # Fallback format: {"C": {"value": 1, "evidence": [...]}}
        if not raw_items and isinstance(parsed.get(criterion), dict):
            raw_items = parsed[criterion].get("evidence", [])

        if isinstance(raw_items, str):
            raw_items = [raw_items]
        elif not isinstance(raw_items, list):
            raw_items = []

        seen = set()
        cleaned_items = []
        for item in raw_items:
            if isinstance(item, dict):
                item = item.get("text") or item.get("quote") or item.get("evidence") or ""
            if not isinstance(item, str):
                continue

            item = item.strip()
            if not item:
                continue

            key = re.sub(r"\s+", " ", item.lower()).strip()
            if key in seen:
                continue

            seen.add(key)
            cleaned_items.append(item)

        predicted_evidence[criterion] = cleaned_items

    return predicted_text_criteria, predicted_evidence


def add_programmatic_age_criteria(criteria: dict[str, int], text: str) -> dict[str, int]:
    criteria = dict(criteria)
    criteria["A2"] = 0
    criteria["A"] = 0

    lines = text.splitlines()
    if len(lines) >= 3:
        age_match = re.search(r"Alter:\s*(\d+)", lines[2])
        if age_match is not None:
            age = int(age_match.group(1))
            if age >= 75:
                criteria["A2"] = 2
                criteria["A"] = 0
            elif age >= 65:
                criteria["A2"] = 0
                criteria["A"] = 1

    return criteria


def extract_gold_criteria(row: dict[str, Any]) -> dict[str, int]:
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

    return gold_criteria


def trim_gold_span(text: str, begin: int, end: int) -> tuple[int, int, str]:
    """Trim leading/trailing whitespace from an offset span while preserving offsets."""
    begin = max(0, begin)
    end = min(len(text), end)

    while begin < end and text[begin].isspace():
        begin += 1
    while end > begin and text[end - 1].isspace():
        end -= 1

    return begin, end, text[begin:end]


def extract_gold_text_criteria_and_evidence(row: dict[str, Any]) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Build LoRA training target from positive manual text-criterion annotations."""
    gold_text_criteria = {criterion: 0 for criterion in TEXT_PARAMETERS}
    gold_evidence = {criterion: [] for criterion in TEXT_PARAMETERS}
    seen_offsets = set()
    seen_texts = {criterion: set() for criterion in TEXT_PARAMETERS}

    text = row["text"]

    positive_annotations = []
    for annotation in row["annotations"]:
        if annotation["polarity"] != "POS":
            continue

        criterion = ANNOTATION_TO_CRITERION.get(annotation["value"])
        if criterion is None:
            continue

        try:
            begin = int(annotation["begin"])
            end = int(annotation["end"])
        except Exception:
            continue

        if begin < 0 or end <= begin or begin >= len(text):
            continue

        begin, end, span_text = trim_gold_span(text, begin, end)
        if not span_text:
            continue

        positive_annotations.append((begin, end, criterion, span_text))

    positive_annotations.sort(key=lambda item: (item[0], item[1], item[2]))

    for begin, end, criterion, span_text in positive_annotations:
        gold_text_criteria[criterion] = 2 if criterion == "S2" else 1

        offset_key = (criterion, begin, end)
        normalized_text_key = re.sub(r"\s+", " ", span_text.lower()).strip()

        if offset_key in seen_offsets:
            continue
        if normalized_text_key in seen_texts[criterion]:
            continue

        seen_offsets.add(offset_key)
        seen_texts[criterion].add(normalized_text_key)
        gold_evidence[criterion].append(span_text)

    return gold_text_criteria, gold_evidence


def extract_gold_evidence(row: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    gold_evidence: dict[str, list[dict[str, Any]]] = {
        criterion: [] for criterion in TEXT_PARAMETERS
    }
    seen = set()
    text = row["text"]

    for annotation in row["annotations"]:
        if annotation["polarity"] != "POS":
            continue

        criterion = ANNOTATION_TO_CRITERION.get(annotation["value"])
        if criterion is None:
            continue

        try:
            begin = int(annotation["begin"])
            end = int(annotation["end"])
        except Exception:
            continue

        if begin < 0 or end <= begin or begin >= len(text):
            continue

        begin, end, span_text = trim_gold_span(text, begin, end)
        if not span_text:
            continue

        key = (criterion, begin, end)
        if key in seen:
            continue
        seen.add(key)

        gold_evidence[criterion].append(
            {
                "begin": begin,
                "end": end,
                "text": span_text,
            }
        )

    for criterion in TEXT_PARAMETERS:
        gold_evidence[criterion].sort(key=lambda item: (item["begin"], item["end"]))

    return gold_evidence


def normalize_with_char_map(text: str) -> tuple[str, list[int]]:
    """Normalize text and keep a map from normalized character positions to original offsets."""
    normalized_chars = []
    char_map = []
    previous_was_space = False

    for original_index, char in enumerate(text):
        if char.isspace():
            if not previous_was_space:
                normalized_chars.append(" ")
                char_map.append(original_index)
                previous_was_space = True
            continue

        normalized_char = char.lower()
        if normalized_char in {"‐", "‑", "‒", "–", "—", "−"}:
            normalized_char = "-"

        normalized_chars.append(normalized_char)
        char_map.append(original_index)
        previous_was_space = False

    joined = "".join(normalized_chars)
    normalized = joined.strip()

    left_trim = len(joined) - len(joined.lstrip())
    if normalized != joined:
        char_map = char_map[left_trim:left_trim + len(normalized)]

    return normalized, char_map


def anchor_evidence_string(document_text: str, evidence_text: str) -> dict[str, Any]:
    """Anchor a predicted evidence string to character offsets in the source letter."""
    evidence_text = evidence_text.strip()

    if not evidence_text:
        return {
            "text": evidence_text,
            "anchored": False,
            "begin": None,
            "end": None,
            "match_type": "empty",
        }

    exact_begin = document_text.find(evidence_text)
    if exact_begin >= 0:
        return {
            "text": evidence_text,
            "anchored": True,
            "begin": exact_begin,
            "end": exact_begin + len(evidence_text),
            "match_type": "exact",
        }

    collapsed_evidence = re.sub(r"\s+", " ", evidence_text).strip()
    if collapsed_evidence != evidence_text:
        exact_begin = document_text.find(collapsed_evidence)
        if exact_begin >= 0:
            return {
                "text": evidence_text,
                "anchored": True,
                "begin": exact_begin,
                "end": exact_begin + len(collapsed_evidence),
                "match_type": "exact_collapsed_whitespace",
            }

    normalized_document, document_char_map = normalize_with_char_map(document_text)
    normalized_evidence, _ = normalize_with_char_map(evidence_text)

    normalized_begin = normalized_document.find(normalized_evidence)
    if normalized_begin >= 0 and document_char_map:
        normalized_end = normalized_begin + len(normalized_evidence)
        original_begin = document_char_map[normalized_begin]
        original_end = document_char_map[normalized_end - 1] + 1

        return {
            "text": evidence_text,
            "anchored": True,
            "begin": original_begin,
            "end": original_end,
            "match_type": "normalized",
        }

    return {
        "text": evidence_text,
        "anchored": False,
        "begin": None,
        "end": None,
        "match_type": "unanchored",
    }


def char_iou(span_a: dict[str, Any], span_b: dict[str, Any]) -> float:
    if span_a.get("begin") is None or span_a.get("end") is None:
        return 0.0
    if span_b.get("begin") is None or span_b.get("end") is None:
        return 0.0

    begin_a = int(span_a["begin"])
    end_a = int(span_a["end"])
    begin_b = int(span_b["begin"])
    end_b = int(span_b["end"])

    overlap = max(0, min(end_a, end_b) - max(begin_a, begin_b))
    union = max(end_a, end_b) - min(begin_a, begin_b)

    if union <= 0:
        return 0.0

    return overlap / union


def greedy_match_spans(
    predicted_spans: list[dict[str, Any]],
    gold_spans: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    candidate_pairs = []

    for pred_index, predicted_span in enumerate(predicted_spans):
        if not predicted_span.get("anchored", False):
            continue

        for gold_index, gold_span in enumerate(gold_spans):
            iou = char_iou(predicted_span, gold_span)
            if iou >= iou_threshold:
                candidate_pairs.append((iou, pred_index, gold_index))

    candidate_pairs.sort(reverse=True, key=lambda item: item[0])

    matched_predicted_indices = set()
    matched_gold_indices = set()
    matches = []

    for iou, pred_index, gold_index in candidate_pairs:
        if pred_index in matched_predicted_indices or gold_index in matched_gold_indices:
            continue

        matched_predicted_indices.add(pred_index)
        matched_gold_indices.add(gold_index)
        matches.append(
            {
                "predicted_index": pred_index,
                "gold_index": gold_index,
                "char_iou": round(float(iou), 4),
            }
        )

    return matches, matched_predicted_indices, matched_gold_indices


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def f1_score_from_pr(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_evidence_for_letter(
    document_text: str,
    predicted_evidence: dict[str, list[str]],
    gold_evidence: dict[str, list[dict[str, Any]]],
    predicted_criteria: dict[str, int],
    gold_criteria: dict[str, int],
    fold_number: int,
    letter_id: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    anchored_prediction: dict[str, list[dict[str, Any]]] = {
        criterion: [] for criterion in TEXT_PARAMETERS
    }
    criterion_rows = []
    match_rows = []

    for criterion in TEXT_PARAMETERS:
        predicted_spans = [
            anchor_evidence_string(document_text, evidence_text)
            for evidence_text in predicted_evidence.get(criterion, [])
        ]
        anchored_prediction[criterion] = predicted_spans

        gold_spans = gold_evidence.get(criterion, [])
        matches, matched_predicted, matched_gold = greedy_match_spans(
            predicted_spans=predicted_spans,
            gold_spans=gold_spans,
            iou_threshold=IOU_THRESHOLD,
        )

        for match in matches:
            predicted_span = predicted_spans[match["predicted_index"]]
            gold_span = gold_spans[match["gold_index"]]
            match_rows.append(
                {
                    "fold": fold_number,
                    "letter_id": letter_id,
                    "criterion": criterion,
                    "predicted_begin": predicted_span["begin"],
                    "predicted_end": predicted_span["end"],
                    "predicted_text": predicted_span["text"],
                    "gold_begin": gold_span["begin"],
                    "gold_end": gold_span["end"],
                    "gold_text": gold_span["text"],
                    "char_iou": match["char_iou"],
                }
            )

        pred_total = len(predicted_spans)
        pred_anchored = sum(1 for span in predicted_spans if span.get("anchored", False))
        gold_total = len(gold_spans)
        matched_total = len(matches)

        criterion_rows.append(
            {
                "fold": fold_number,
                "letter_id": letter_id,
                "criterion": criterion,
                "predicted_criterion_positive": int(predicted_criteria.get(criterion, 0) > 0),
                "gold_criterion_positive": int(gold_criteria.get(criterion, 0) > 0),
                "predicted_evidence_spans": pred_total,
                "anchored_predicted_evidence_spans": pred_anchored,
                "gold_evidence_spans": gold_total,
                "matched_predicted_evidence_spans": matched_total,
                "matched_gold_evidence_spans": matched_total,
                "criterion_evidence_sufficient_tp": int(
                    predicted_criteria.get(criterion, 0) > 0
                    and gold_criteria.get(criterion, 0) > 0
                    and matched_total > 0
                ),
                "criterion_evidence_required_tp": int(
                    predicted_criteria.get(criterion, 0) > 0
                    and gold_criteria.get(criterion, 0) > 0
                ),
                "criterion_evidence_covered_gold_positive": int(
                    gold_criteria.get(criterion, 0) > 0
                    and matched_total > 0
                ),
                "criterion_evidence_required_gold_positive": int(
                    gold_criteria.get(criterion, 0) > 0
                ),
            }
        )

    return anchored_prediction, criterion_rows, match_rows


def summarize_evidence_counts(grouped_df: pd.DataFrame) -> pd.DataFrame:
    count_columns = [
        "predicted_evidence_spans",
        "anchored_predicted_evidence_spans",
        "gold_evidence_spans",
        "matched_predicted_evidence_spans",
        "matched_gold_evidence_spans",
        "criterion_evidence_sufficient_tp",
        "criterion_evidence_required_tp",
        "criterion_evidence_covered_gold_positive",
        "criterion_evidence_required_gold_positive",
    ]

    summary = grouped_df[count_columns].sum().reset_index()

    summary["evidence_grounding_rate"] = summary.apply(
        lambda row: safe_divide(
            row["anchored_predicted_evidence_spans"],
            row["predicted_evidence_spans"],
        ),
        axis=1,
    )
    summary["evidence_precision_anchored"] = summary.apply(
        lambda row: safe_divide(
            row["matched_predicted_evidence_spans"],
            row["anchored_predicted_evidence_spans"],
        ),
        axis=1,
    )
    summary["evidence_precision_strict"] = summary.apply(
        lambda row: safe_divide(
            row["matched_predicted_evidence_spans"],
            row["predicted_evidence_spans"],
        ),
        axis=1,
    )
    summary["evidence_recall_coverage"] = summary.apply(
        lambda row: safe_divide(
            row["matched_gold_evidence_spans"],
            row["gold_evidence_spans"],
        ),
        axis=1,
    )
    summary["evidence_f1_anchored"] = summary.apply(
        lambda row: f1_score_from_pr(
            row["evidence_precision_anchored"],
            row["evidence_recall_coverage"],
        ),
        axis=1,
    )
    summary["evidence_f1_strict"] = summary.apply(
        lambda row: f1_score_from_pr(
            row["evidence_precision_strict"],
            row["evidence_recall_coverage"],
        ),
        axis=1,
    )
    summary["criterion_evidence_sufficiency_tp"] = summary.apply(
        lambda row: safe_divide(
            row["criterion_evidence_sufficient_tp"],
            row["criterion_evidence_required_tp"],
        ),
        axis=1,
    )
    summary["criterion_evidence_coverage_gold_positive"] = summary.apply(
        lambda row: safe_divide(
            row["criterion_evidence_covered_gold_positive"],
            row["criterion_evidence_required_gold_positive"],
        ),
        axis=1,
    )

    return summary


def compute_length_stats(records: list[dict[str, str]], tokenizer_for_lengths: Any) -> dict[str, Any]:
    lengths = []
    for record in records:
        encoded = tokenizer_for_lengths(
            record["prompt"] + record["completion"],
            add_special_tokens=False,
        )
        lengths.append(len(encoded["input_ids"]))

    if not lengths:
        return {
            "n_train_records": 0,
            "max_tokens": 0,
            "p95_tokens": 0,
            "n_over_max_seq_length": 0,
        }

    return {
        "n_train_records": len(lengths),
        "max_tokens": int(max(lengths)),
        "p95_tokens": float(np.percentile(lengths, 95)),
        "n_over_max_seq_length": int(sum(length > MAX_SEQ_LENGTH for length in lengths)),
    }


# =============================================================================
# 5. Train one LoRA adapter per fold and evaluate on the held-out test split
# =============================================================================

all_results = []
evidence_rows = []
evidence_match_rows = []
length_stat_rows = []

for fold_number, fold in enumerate(folds, start=1):
    print("\n" + "=" * 80)
    print(f"Fold {fold_number}")
    print("=" * 80)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    gc.collect()
    torch.cuda.empty_cache()

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
        gold_text_criteria, gold_evidence = extract_gold_text_criteria_and_evidence(row)

        target = {
            "criteria": gold_text_criteria,
            "evidence": gold_evidence,
        }

        assistant_json = json.dumps(
            target,
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

    length_stats = compute_length_stats(train_records, base_tokenizer)
    length_stats["fold"] = fold_number
    length_stat_rows.append(length_stats)
    print("Training length stats:", length_stats)

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
        desc=f"Qwen3.5 27B LoRA all evidence fold {fold_number}",
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

            parsed = extract_json_object(response_text)
            valid_json = bool(parsed)

            predicted_text_criteria, predicted_evidence = parse_prediction(parsed)
            predicted_criteria = add_programmatic_age_criteria(predicted_text_criteria, text)
            predicted_score = sum(predicted_criteria.values())

            gold_criteria = extract_gold_criteria(row)
            gold_score = sum(gold_criteria.values())
            gold_evidence = extract_gold_evidence(row)

            anchored_predicted_evidence, letter_evidence_rows, letter_match_rows = evaluate_evidence_for_letter(
                document_text=text,
                predicted_evidence=predicted_evidence,
                gold_evidence=gold_evidence,
                predicted_criteria=predicted_criteria,
                gold_criteria=gold_criteria,
                fold_number=fold_number,
                letter_id=row["letter_id"],
            )

            evidence_rows.extend(letter_evidence_rows)
            evidence_match_rows.extend(letter_match_rows)

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
                    "gold_evidence": gold_evidence,
                    "predicted_evidence": predicted_evidence,
                    "anchored_predicted_evidence": anchored_predicted_evidence,
                }
            )

            progress.update(1)

    del trainer
    del model
    del tokenizer
    del train_dataset
    del train_records

    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# 6. Score and criterion metrics
# =============================================================================

results_df = pd.DataFrame(all_results)
evidence_records_df = pd.DataFrame(evidence_rows)
evidence_matches_df = pd.DataFrame(evidence_match_rows)
length_stats_df = pd.DataFrame(length_stat_rows)

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


# =============================================================================
# 7. Evidence metrics
# =============================================================================

if len(evidence_records_df) > 0:
    evidence_by_criterion = summarize_evidence_counts(
        evidence_records_df.groupby("criterion", as_index=False)
    )
    evidence_by_fold = summarize_evidence_counts(
        evidence_records_df.groupby("fold", as_index=False)
    )
    evidence_summary = summarize_evidence_counts(
        evidence_records_df.assign(all_letters="all").groupby("all_letters", as_index=False)
    )
else:
    evidence_by_criterion = pd.DataFrame()
    evidence_by_fold = pd.DataFrame()
    evidence_summary = pd.DataFrame()


# =============================================================================
# 8. Print outputs
# =============================================================================

print("\nFold summary:")
print(fold_summary.round(4).to_string(index=False))

print("\nOverall summary:")
print(overall_summary.round(4).to_string(index=False))

print(f"\nPooled micro-F1: {micro_f1:.4f}")
print(classification_report_df.to_string())

print("\nEvidence summary:")
if len(evidence_summary) > 0:
    print(evidence_summary.round(4).to_string(index=False))
else:
    print("No evidence records.")

print("\nEvidence by criterion:")
if len(evidence_by_criterion) > 0:
    print(evidence_by_criterion.round(4).to_string(index=False))
else:
    print("No evidence records.")

print("\nTraining length stats:")
print(length_stats_df.round(2).to_string(index=False))


# =============================================================================
# 9. Save outputs
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

evidence_records_df.to_csv(OUTPUT_DIR / "evidence_records.csv", index=False)
evidence_matches_df.to_csv(OUTPUT_DIR / "evidence_matches.csv", index=False)
evidence_summary.to_csv(OUTPUT_DIR / "evidence_summary.csv", index=False)
evidence_by_fold.to_csv(OUTPUT_DIR / "evidence_by_fold.csv", index=False)
evidence_by_criterion.to_csv(OUTPUT_DIR / "evidence_by_criterion.csv", index=False)
length_stats_df.to_csv(OUTPUT_DIR / "training_length_stats.csv", index=False)

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
            "pipeline": "qwen35_27b_lora_all_evidence",
            "task": "score_criteria_evidence_extraction",
            "evidence_output": True,
            "evidence_scope": "all_positive_spans_for_text_criteria",
            "evidence_parameters": TEXT_PARAMETERS,
            "age_evidence": False,
            "model_name": MODEL_NAME,
            "dataset_path": str(DATASET_PATH),
            "prompt_file": str(PROMPT_FILE),
            "output_dir": str(OUTPUT_DIR),
            "max_seq_length": MAX_SEQ_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "dtype": "torch.bfloat16",
            "load_in_4bit": False,
            "seed": SEED,
            "do_sample": False,
            "num_beams": 1,
            "deterministic_algorithms": True,
            "parameters": PARAMETERS,
            "evidence_iou_type": "character",
            "evidence_iou_threshold": IOU_THRESHOLD,
            "training_config": training_config,
            "training_length_stats": length_stat_rows,
            "evidence_metrics": {
                "grounding_rate": "anchored_predicted_evidence_spans / predicted_evidence_spans",
                "precision_anchored": "matched_predicted_evidence_spans / anchored_predicted_evidence_spans",
                "precision_strict": "matched_predicted_evidence_spans / predicted_evidence_spans",
                "recall_coverage": "matched_gold_evidence_spans / gold_evidence_spans",
                "f1_anchored": "F1(precision_anchored, recall_coverage)",
                "f1_strict": "F1(precision_strict, recall_coverage)",
                "criterion_sufficiency_tp": "TP criteria with at least one matched evidence span / TP criteria",
                "criterion_coverage_gold_positive": "gold-positive criteria with at least one matched evidence span / gold-positive criteria",
            },
        },
        file,
        ensure_ascii=False,
        indent=2,
    )

print("\nSaved outputs to:")
print(OUTPUT_DIR)
