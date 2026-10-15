# CARDIO:DE400 Evidence Results

These results supplement the manuscript
"Evidence-grounded CHA2DS2-VA extraction from German clinical letters using
locally deployable language models".
They use CARDIO:DE400 three-fold cross-validation with all-evidence output:
criteria plus source quotations. [Supplement overview](additional_results.md).

## Evidence-Generating Score Results

| System | Exact score accuracy | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Qwen 27B LoRA | 0.8950 | 0.1275 | 0.4153 |
| Qwen 3.8 27B LoRA | 0.8700 | 0.1500 | 0.4359 |
| Qwen 27B zero-shot | 0.7975 | 0.2225 | 0.5123 |
| Qwen 9B LoRA | 0.8600 | 0.1875 | 0.5454 |
| Qwen 9B zero-shot | 0.6950 | 0.4575 | 0.9014 |

The Qwen 3.8 27B LoRA run is an additional score-level experiment. The
plausibility and faithfulness tables below cover Qwen 3.5 9B/27B only.
[Complete score metrics](cv3_score_results.csv) and
[per-criterion accuracies](cv3_per_criterion_accuracy.csv) include both output types.
Score metric definitions are in the [overview](additional_results.md#score-and-criterion-metrics).

## Evidence Metrics

- Grounding rate: anchored predicted spans divided by all predicted spans.
- Anchored precision: matched predicted spans divided by anchored predicted spans.
- Strict precision: matched predicted spans divided by all predicted spans.
- Recall coverage: matched gold spans divided by all gold spans.
- Anchored and strict F1: F1 calculated from the corresponding precision and recall.
- Criterion evidence sufficiency TP: fraction of true-positive criterion cases
  with at least one evidence quotation matching the reference.
- Criterion evidence coverage gold positive: fraction of reference-positive
  criterion cases covered by at least one matching predicted quotation.
- Sufficiency preservation: fraction of evaluable positive criterion cases
  remaining positive when only selected evidence is retained.
- Comprehensiveness drop: fraction of evaluable positive criterion cases
  becoming negative after every occurrence of selected quotations is removed.
- Removed still positive: fraction remaining positive after quotation removal.

## Evidence Plausibility

Plausibility uses unique-quotation matching against expert annotations.

| System | Grounding rate | Anchored precision | Strict precision | Recall coverage | Anchored F1 | Strict F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 9B zero-shot | 0.9292 | 0.1689 | 0.1570 | 0.2805 | 0.2109 | 0.2013 |
| Qwen 27B zero-shot | 0.9744 | 0.2579 | 0.2513 | 0.3447 | 0.2950 | 0.2907 |
| Qwen 9B LoRA | 0.9888 | 0.8367 | 0.8274 | 0.7792 | 0.8070 | 0.8026 |
| Qwen 27B LoRA | 0.9936 | 0.8571 | 0.8517 | 0.8068 | 0.8312 | 0.8287 |

The complete [overall plausibility table](cv3_evidence_plausibility.csv) retains
all predicted, anchored, gold, and matched span counts and criterion-level
coverage measures. [Results by criterion](cv3_evidence_plausibility_by_criterion.csv)
provide the same metrics separately for `C`, `H`, `D`, `S2`, and `V`.

## Evidence Faithfulness

Faithfulness evaluates the five text-derived criteria over 400 letters,
giving 2000 criterion cases per system. Full, sufficiency, and comprehensiveness
responses all had a JSON-validity rate of 1.0000.

| System | Evaluable cases | Sufficiency preservation | Comprehensiveness drop | Removed still positive |
| --- | ---: | ---: | ---: | ---: |
| Qwen 9B zero-shot | 881 | 0.9807 | 0.3507 | 0.6493 |
| Qwen 27B zero-shot | 873 | 0.9851 | 0.4960 | 0.5040 |
| Qwen 9B LoRA | 885 | 0.9864 | 0.5243 | 0.4757 |
| Qwen 27B LoRA | 905 | 0.9912 | 0.5249 | 0.4751 |

[Overall faithfulness results](cv3_faithfulness.csv) retain all case counts,
mean anchored evidence spans, and JSON-validity rates.
[Faithfulness by criterion](cv3_faithfulness_by_criterion.csv) contains the
individual criterion results.
