#!/usr/bin/env python
"""Post-hoc faithfulness evaluation for CARDIO:DE all-evidence runs.

Sufficiency: keep only the model-generated evidence for one criterion and test
whether the frozen model preserves its full-input positive prediction.

Comprehensiveness: remove the model-generated evidence for one criterion from
full input and test whether the frozen model loses its full-input positive
prediction.

Model, decoding settings, and score-only prompt are fixed. Only the input
text changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import gc
import json
import os
import random
import re

# Must be set before CUDA libraries are initialized.
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
import unsloth  # noqa: F401; keeps Unsloth patches active before model loading
from unsloth import FastLanguageModel


SEED = 42
TEXT_PARAMETERS = ["C", "H", "D", "S2", "V"]


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


set_seed()


@dataclass(frozen=True)
class FaithfulnessConfig:
    run_name: str
    model_name: str
    is_lora: bool
    max_seq_length: int = 16384
    max_new_tokens: int = 160


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return {}

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def as_positive(value: Any, positive_value: int = 1) -> int:
    try:
        return positive_value if int(value) > 0 else 0
    except Exception:
        return 0


def parse_text_criteria(parsed: dict[str, Any]) -> dict[str, int]:
    criteria_obj = parsed.get("criteria", parsed)
    if not isinstance(criteria_obj, dict):
        criteria_obj = {}

    return {
        "C": as_positive(criteria_obj.get("C", parsed.get("C", 0)), 1),
        "H": as_positive(criteria_obj.get("H", parsed.get("H", 0)), 1),
        "D": as_positive(criteria_obj.get("D", parsed.get("D", 0)), 1),
        "S2": as_positive(criteria_obj.get("S2", parsed.get("S2", 0)), 2),
        "V": as_positive(criteria_obj.get("V", parsed.get("V", 0)), 1),
    }


def get_base_tokenizer(processing_class: Any) -> Any:
    return processing_class.tokenizer if hasattr(processing_class, "tokenizer") else processing_class


def get_eos_token_id(processing_class: Any) -> int:
    base_tokenizer = get_base_tokenizer(processing_class)
    eos_token_id = base_tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = base_tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_token_id is None or eos_token_id < 0:
        raise ValueError("Could not determine a valid EOS token id.")
    return int(eos_token_id)


def decode_tokens(processing_class: Any, token_ids: torch.Tensor) -> str:
    decoder = processing_class if hasattr(processing_class, "decode") else get_base_tokenizer(processing_class)
    return decoder.decode(token_ids, skip_special_tokens=True).strip()


def load_model(model_source: str, max_seq_length: int) -> tuple[Any, Any, int]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    eos_token_id = get_eos_token_id(tokenizer)
    return model, tokenizer, eos_token_id


def predict_criteria(
    model: Any,
    tokenizer: Any,
    eos_token_id: int,
    system_prompt: str,
    input_text: str,
    max_new_tokens: int,
) -> tuple[dict[str, int], str, bool]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": f"Arztbrief:\n\n{input_text}"}]},
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=eos_token_id,
            use_cache=True,
        )

    input_length = inputs["input_ids"].shape[-1]
    response_text = decode_tokens(tokenizer, generated[0][input_length:])
    parsed = extract_json_object(response_text)
    return parse_text_criteria(parsed), response_text, bool(parsed)


def normalized_span_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def get_anchored_spans(result: dict[str, Any], criterion: str, document_text: str) -> list[dict[str, Any]]:
    """Return anchored model evidence spans for one criterion."""
    spans = []

    anchored = result.get("anchored_predicted_evidence", {})
    if isinstance(anchored, dict):
        for span in anchored.get(criterion, []) or []:
            if not isinstance(span, dict) or not span.get("anchored", False):
                continue
            begin = span.get("begin")
            end = span.get("end")
            if begin is None or end is None:
                continue
            begin = int(begin)
            end = int(end)
            if begin < 0 or end <= begin or end > len(document_text):
                continue
            text = str(span.get("text") or document_text[begin:end]).strip()
            spans.append({"begin": begin, "end": end, "text": text})

    if spans:
        return deduplicate_spans(spans)

    # Fallback for older result files without anchored_predicted_evidence.
    predicted = result.get("predicted_evidence", {})
    if isinstance(predicted, dict):
        for item in predicted.get(criterion, []) or []:
            if isinstance(item, dict):
                item = item.get("text") or item.get("quote") or item.get("evidence") or ""
            if not isinstance(item, str) or not item.strip():
                continue
            begin = document_text.find(item.strip())
            if begin >= 0:
                spans.append({"begin": begin, "end": begin + len(item.strip()), "text": item.strip()})

    return deduplicate_spans(spans)


def deduplicate_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for span in sorted(spans, key=lambda x: (int(x["begin"]), int(x["end"]))):
        key = (int(span["begin"]), int(span["end"]), normalized_span_text(span["text"]).lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(span)
    return deduped


def merge_intervals(spans: list[dict[str, Any]]) -> list[tuple[int, int]]:
    intervals = sorted((int(s["begin"]), int(s["end"])) for s in spans)
    merged: list[tuple[int, int]] = []
    for begin, end in intervals:
        if not merged or begin > merged[-1][1]:
            merged.append((begin, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def evidence_only_text(spans: list[dict[str, Any]], document_text: str) -> str:
    snippets = []
    for span in spans:
        text = normalized_span_text(str(span.get("text") or ""))
        if not text:
            text = normalized_span_text(document_text[int(span["begin"]):int(span["end"])])
        if text:
            snippets.append(text)
    return "\n".join(snippets)


#### Helpers

HYPHEN_VARIANTS = {"‐", "‑", "‒", "–", "—", "−"}


def normalize_with_char_map(
    text: str,
) -> tuple[str, list[int]]:
    chars = []
    char_map = []
    previous_space = False

    for index, char in enumerate(text):
        if char.isspace():
            if not previous_space:
                chars.append(" ")
                char_map.append(index)
                previous_space = True
            continue

        chars.append(
            "-" if char in HYPHEN_VARIANTS else char.lower()
        )
        char_map.append(index)
        previous_space = False

    joined = "".join(chars)
    left_trim = len(joined) - len(joined.lstrip())
    normalized = joined.strip()

    return normalized, char_map[
        left_trim:left_trim + len(normalized)
    ]


def find_all_occurrences(
    document_text: str,
    quotation: str,
) -> list[tuple[int, int]]:
    normalized_document, char_map = normalize_with_char_map(
        document_text
    )
    normalized_quotation, _ = normalize_with_char_map(
        quotation
    )

    if not normalized_quotation or not char_map:
        return []

    occurrences = []
    start = 0

    while True:
        position = normalized_document.find(
            normalized_quotation,
            start,
        )
        if position < 0:
            break

        normalized_end = position + len(normalized_quotation)

        occurrences.append((
            char_map[position],
            char_map[normalized_end - 1] + 1,
        ))

        start = position + 1

    return occurrences

####


def evidence_removed_text(
    document_text: str,
    spans: list[dict[str, Any]],
) -> tuple[str, list[tuple[int, int]]]:
    """Remove all occurrences of every selected evidence quotation."""
    intervals = set()

    for span in spans:
        begin = int(span["begin"])
        end = int(span["end"])

        if not 0 <= begin < end <= len(document_text):
            continue

        # Use the actual anchored source text as the quotation.
        quotation = document_text[begin:end].strip()
        if not quotation:
            continue

        occurrences = find_all_occurrences(
            document_text,
            quotation,
        )

        if occurrences:
            intervals.update(occurrences)
        else:
            # Defensive fallback: remove the saved anchored span.
            intervals.add((begin, end))

    intervals = sorted(intervals)

    removal_spans = [
        {"begin": begin, "end": end}
        for begin, end in intervals
    ]

    chars = list(document_text)

    for begin, end in merge_intervals(removal_spans):
        chars[begin:end] = [" "] * (end - begin)

    return "".join(chars), intervals


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = [("all", df)] if not group_cols else df.groupby(group_cols, dropna=False)

    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        row = {col: value for col, value in zip(group_cols or ["all_letters"], key)}
        denominator = int(group["faithfulness_evaluable"].sum())
        row.update(
            {
                "criterion_cases": int(len(group)),
                "full_positive_cases": int(group["full_positive"].sum()),
                "faithfulness_evaluable_cases": denominator,
                "sufficiency_preserved_cases": int(group["sufficiency_preserved"].sum()),
                "comprehensiveness_dropped_cases": int(group["comprehensiveness_dropped"].sum()),
                "mean_anchored_evidence_spans": float(group["n_anchored_evidence_spans"].mean()),
                "full_json_valid_rate": float(group["full_valid_json"].mean()),
                "sufficiency_json_valid_rate": safe_divide(
                    group.loc[group["faithfulness_evaluable"], "sufficiency_valid_json"].sum(),
                    denominator,
                ),
                "comprehensiveness_json_valid_rate": safe_divide(
                    group.loc[group["faithfulness_evaluable"], "comprehensiveness_valid_json"].sum(),
                    denominator,
                ),
                "sufficiency_preservation_rate": safe_divide(
                    group["sufficiency_preserved"].sum(),
                    denominator,
                ),
                "comprehensiveness_drop_rate": safe_divide(
                    group["comprehensiveness_dropped"].sum(),
                    denominator,
                ),
                "removed_still_positive_rate": safe_divide(
                    group["comprehensiveness_still_positive"].sum(),
                    denominator,
                ),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def run_faithfulness(config: FaithfulnessConfig) -> None:
    root = repo_root()
    dataset_path = Path("../../../data/cardiode_hf_cv3/cha2ds2_va_CARDIODE400_cv3")
    score_prompt_file = root / "prompts" / "cardiode" / "qwen_score_calculation_de.txt"
    evidence_run_dir = root / "runs" / "cardiode" / "evidence_benchmark" / config.run_name
    evidence_results_file = evidence_run_dir / "test_results.json"
    output_dir = root / "runs" / "cardiode" / "faithfulness_benchmark" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = score_prompt_file.read_text(encoding="utf-8")
    evidence_results = pd.read_json(evidence_results_file).to_dict(orient="records")
    result_by_key = {
        (int(row["fold"]), str(row["letter_id"])): row
        for row in evidence_results
    }

    folds = [
        load_from_disk(dataset_path / f"cha2ds2_va_fold_{fold_number}")
        for fold_number in range(1, 4)
    ]

    print("Faithfulness run:", config.run_name)
    print("Input evidence results:", evidence_results_file)
    print("Output directory:", output_dir)

    rows = []
    shared_model = None
    shared_tokenizer = None
    shared_eos_token_id = None
    debug_example_printed = False

    if not config.is_lora:
        shared_model, shared_tokenizer, shared_eos_token_id = load_model(
            config.model_name,
            config.max_seq_length,
        )

    for fold_number, fold in enumerate(folds, start=1):
        set_seed()

        if config.is_lora:
            adapter_path = evidence_run_dir / f"fold_{fold_number}" / "adapter"
            model, tokenizer, eos_token_id = load_model(str(adapter_path), config.max_seq_length)
        else:
            model = shared_model
            tokenizer = shared_tokenizer
            eos_token_id = shared_eos_token_id

        assert model is not None and tokenizer is not None and eos_token_id is not None

        with tqdm(total=len(fold["test"]), desc=f"{config.run_name} fold {fold_number}", unit="letter") as progress:
            for letter in fold["test"]:
                letter_id = str(letter["letter_id"])
                text = letter["text"]
                evidence_result = result_by_key[(fold_number, letter_id)]

                full_pred, full_raw, full_valid = predict_criteria(
                    model=model,
                    tokenizer=tokenizer,
                    eos_token_id=eos_token_id,
                    system_prompt=system_prompt,
                    input_text=text,
                    max_new_tokens=config.max_new_tokens,
                )

                for criterion in TEXT_PARAMETERS:
                    spans = get_anchored_spans(evidence_result, criterion, text)
                    n_spans = len(spans)
                    full_positive = int(full_pred.get(criterion, 0) > 0)
                    evaluable = bool(full_positive and n_spans > 0)

                    suff_pred = {}
                    suff_raw = ""
                    suff_valid = False
                    comp_pred = {}
                    comp_raw = ""
                    comp_valid = False

                    removed_occurrences = []

                    if evaluable:
                        kept_text = evidence_only_text(spans, text)
                        removed_text, removed_occurrences = evidence_removed_text(text, spans)

                        if not debug_example_printed:
                            print("\n" + "=" * 100)
                            print("FAITHFULNESS INPUT EXAMPLE")
                            print("=" * 100)
                        
                            print("\nLETTER ID:")
                            print(letter_id)
                        
                            print("\nCRITERION:")
                            print(criterion)
                        
                            print("\nPREDICTED EVIDENCE SPANS:")
                            for span in spans:
                                print("-", span["text"])
                        
                            print("\n--- SUFFICIENCY INPUT: evidence only ---")
                            print(kept_text)
                        
                            print("\n--- COMPREHENSIVENESS INPUT: evidence removed ---")
                            print(removed_text[:4000])
                        
                            print("=" * 100 + "\n")
                            debug_example_printed = True

                        suff_pred, suff_raw, suff_valid = predict_criteria(
                            model=model,
                            tokenizer=tokenizer,
                            eos_token_id=eos_token_id,
                            system_prompt=system_prompt,
                            input_text=kept_text,
                            max_new_tokens=config.max_new_tokens,
                        )
                        comp_pred, comp_raw, comp_valid = predict_criteria(
                            model=model,
                            tokenizer=tokenizer,
                            eos_token_id=eos_token_id,
                            system_prompt=system_prompt,
                            input_text=removed_text,
                            max_new_tokens=config.max_new_tokens,
                        )

                    suff_positive = int(suff_pred.get(criterion, 0) > 0)
                    comp_positive = int(comp_pred.get(criterion, 0) > 0)

                    rows.append(
                        {
                            "fold": fold_number,
                            "letter_id": letter_id,
                            "criterion": criterion,
                            "n_anchored_evidence_spans": n_spans,
                            "comprehensiveness_removed_occurrences": (
                                len(removed_occurrences) if evaluable else 0
                            ),
                            "comprehensiveness_removed_characters": (
                                sum(
                                    end - begin
                                    for begin, end in merge_intervals([
                                        {"begin": begin, "end": end}
                                        for begin, end in removed_occurrences
                                    ])
                                )
                                if evaluable else 0
                            ),
                            "full_valid_json": bool(full_valid),
                            "full_positive": bool(full_positive),
                            "faithfulness_evaluable": bool(evaluable),
                            "sufficiency_valid_json": bool(suff_valid),
                            "sufficiency_positive": bool(suff_positive),
                            "sufficiency_preserved": bool(
                                evaluable and suff_positive
                            ),
                            "comprehensiveness_valid_json": bool(comp_valid),
                            "comprehensiveness_positive_after_removal": bool(
                                comp_positive
                            ),
                            "comprehensiveness_still_positive": bool(
                                evaluable and comp_valid and comp_positive
                            ),
                            "comprehensiveness_dropped": bool(
                                    evaluable and comp_valid and not comp_positive
                                ),
                            "full_prediction": full_pred,
                            "sufficiency_prediction": suff_pred,
                            "comprehensiveness_prediction": comp_pred,
                            "full_raw_response": full_raw,
                            "sufficiency_raw_response": suff_raw,
                            "comprehensiveness_raw_response": comp_raw,
                        }
                    )
                progress.update(1)

        if config.is_lora:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    if not config.is_lora:
        del shared_model, shared_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(rows)
    summary_df = summarize(results_df, [])
    by_fold_df = summarize(results_df, ["fold"])
    by_criterion_df = summarize(results_df, ["criterion"])

    print("\nFaithfulness summary:")
    print(summary_df.round(4).to_string(index=False))
    print("\nBy criterion:")
    print(by_criterion_df.round(4).to_string(index=False))

    results_df.to_json(
        output_dir / "faithfulness_results.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    summary_df.to_csv(output_dir / "faithfulness_summary.csv", index=False)
    by_fold_df.to_csv(output_dir / "faithfulness_by_fold.csv", index=False)
    by_criterion_df.to_csv(output_dir / "faithfulness_by_criterion.csv", index=False)

    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "pipeline": f"{config.run_name}_faithfulness",
                "task": "posthoc_sufficiency_comprehensiveness",
                "input_evidence_results": str(evidence_results_file),
                "score_prompt_file": str(score_prompt_file),
                "model_name": config.model_name,
                "is_lora": config.is_lora,
                "max_seq_length": config.max_seq_length,
                "max_new_tokens": config.max_new_tokens,
                "dtype": "torch.bfloat16",
                "load_in_4bit": False,
                "seed": SEED,
                "text_parameters": TEXT_PARAMETERS,
                "sufficiency_definition": (
                    "For a full-input positive criterion with anchored model evidence, "
                    "keep only that criterion's evidence spans and test whether the same "
                    "frozen model preserves the positive prediction. Higher is better."
                ),
                "comprehensiveness_definition": (
                    "For a full-input positive criterion with anchored model evidence, "
                    "remove all source occurrences of every model-selected evidence "
                    "quotation and test whether the same frozen model loses the "
                    "positive prediction. No gold evidence is used. Higher is better."
                ),
                "comprehensiveness_removal_policy": (
                    "Quotation-level removal of all matching occurrences, with "
                    "case-insensitive, whitespace-normalized and hyphen-normalized "
                    "matching."
                ),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved outputs to:")
    print(output_dir)
