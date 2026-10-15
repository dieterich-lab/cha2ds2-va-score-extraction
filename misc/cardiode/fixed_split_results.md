# CARDIO:DE Fixed-Split Results

These supplementary experiments use CARDIO:DE400 as the training set and
CARDIO:DE100 as the fixed held-out test set. They supplement the manuscript
"Evidence-grounded CHA2DS2-VA extraction from German clinical letters using
locally deployable language models",
but are not reported as its main results. CARDIODE100 CHA2DS2-VA annotations
are held in-house and are not published.

The corpus is available for research use via
[heiDATA: DOI 10.11588/data/AFYQDY](https://doi.org/10.11588/data/AFYQDY).
See the [supplement overview](additional_results.md) for the reproducible
CARDIO:DE400 evaluation and [score metric definitions](additional_results.md#metric-definitions).

## Score Results

| System | Output type | Exact score accuracy | MAE | RMSE | Criterion micro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rules | criteria-only | 0.4900 | 0.6300 | 0.9434 | 0.8464 |
| Qwen 9B zero-shot | criteria-only | 0.7700 | 0.2900 | 0.6557 | 0.9506 |
| Qwen 27B zero-shot | criteria-only | 0.8600 | 0.1600 | 0.4472 | 0.9667 |
| Qwen 9B LoRA | criteria-only | 0.8900 | 0.1300 | 0.4123 | 0.9805 |
| Qwen 27B LoRA | criteria-only | 0.9000 | 0.1100 | 0.3606 | 0.9823 |
| Qwen 9B zero-shot | all evidence | 0.7100 | 0.4800 | 1.0100 | 0.9242 |
| Qwen 27B zero-shot | all evidence | 0.7900 | 0.2600 | 0.6164 | 0.9464 |
| Qwen 9B LoRA | all evidence | 0.8600 | 0.1600 | 0.4472 | 0.9752 |
| Qwen 27B LoRA | all evidence | 0.8900 | 0.1200 | 0.3742 | 0.9805 |

All criteria-only Qwen runs had a JSON-validity rate of 1.0000. For all-evidence
output, the rate was 0.9700 for 9B zero-shot and 1.0000 for the other runs.
JSON validity does not apply to rules.

[Complete score results](fixed_split_score_results.csv) also report
all-criteria exact accuracy, mean criterion accuracy, Krippendorff alpha,
and JSON validity. [Per-criterion accuracies](fixed_split_per_criterion_accuracy.csv)
cover all nine configurations.

## Evidence Plausibility

Unique-quotation matching is used for all-evidence runs.
[Evidence metric definitions](evidence_results.md#evidence-metrics) apply here.

| System | Grounding rate | Anchored precision | Strict precision | Recall coverage | Anchored F1 | Strict F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 9B zero-shot | 0.9330 | 0.1510 | 0.1409 | 0.2635 | 0.1920 | 0.1836 |
| Qwen 27B zero-shot | 0.9806 | 0.2470 | 0.2422 | 0.3499 | 0.2895 | 0.2862 |
| Qwen 9B LoRA | 0.9954 | 0.8756 | 0.8716 | 0.8207 | 0.8473 | 0.8454 |
| Qwen 27B LoRA | 1.0000 | 0.8844 | 0.8844 | 0.8423 | 0.8628 | 0.8628 |

[Complete plausibility results](fixed_split_evidence_plausibility.csv) retain
all evidence span counts and criterion-level sufficiency and coverage measures.

## Evidence Faithfulness

Five text-derived criteria over 100 letters give 500 criterion cases per system.
Full, sufficiency, and comprehensiveness responses all had a JSON-validity rate
of 1.0000.

| System | Evaluable cases | Sufficiency preservation | Comprehensiveness drop | Removed still positive |
| --- | ---: | ---: | ---: | ---: |
| Qwen 9B zero-shot | 219 | 0.9817 | 0.3836 | 0.6164 |
| Qwen 27B zero-shot | 220 | 0.9955 | 0.3909 | 0.6091 |
| Qwen 9B LoRA | 225 | 0.9822 | 0.2178 | 0.7822 |
| Qwen 27B LoRA | 222 | 0.9910 | 0.2297 | 0.7703 |

[Complete faithfulness results](fixed_split_faithfulness.csv) retain the
positive, evaluable, preserved, and dropped case counts and mean anchored spans.

## NER Ablation

The NER ablation predicts line-level entities/spans and derives the score
criteria and total score from those entities. Explicit entity spans and
attributes support interpretability, but this approach is computationally more
demanding than document-level extraction and its overall score performance is
worse than the best direct extraction runs.

Line JSON validity is the fraction of line-level responses containing valid
JSON. Line length validity is the fraction satisfying the expected output
length constraint.

| System | n letters | Exact score accuracy | All-criteria exact accuracy | score MAE | micro-F1 | line JSON valid rate | line length valid rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 27B NER | 100 | 0.7700 | 0.7600 | 0.2800 | 0.9546 | 0.9981 | 0.9739 |

| NER span metric | Matched entities | Predicted entities | Gold entities | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| value_iou50 | 837 | 1212 | 909 | 0.6906 | 0.9208 | 0.7893 |
| full_attributes_iou50 | 732 | 1212 | 909 | 0.6040 | 0.8053 | 0.6902 |

### Criterion Support And Classification

Support counts positive reference letters in the 100-letter test set. The
seven criteria have 285 positive labels in total.

| Criterion | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| C | 0.7925 | 0.9767 | 0.8750 | 43 |
| H | 0.9872 | 1.0000 | 0.9935 | 77 |
| A2 | 1.0000 | 1.0000 | 1.0000 | 30 |
| D | 1.0000 | 1.0000 | 1.0000 | 24 |
| S2 | 0.8333 | 1.0000 | 0.9091 | 15 |
| V | 0.8690 | 1.0000 | 0.9299 | 73 |
| A | 1.0000 | 1.0000 | 1.0000 | 23 |

The [full NER classification report](fixed_split_ner_classification_report.csv)
also includes micro, macro, weighted, and samples averages.
