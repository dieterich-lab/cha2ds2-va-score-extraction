# CARDIO:DE and Charite Additional Experiment Results

This file provides supplementary result documentation for the manuscript
"Evidence-grounded CHA2DS2-VA extraction from German clinical letters using
locally deployable language models". It contains additional results and metrics,
including CARDIO:DE100 results, RMSE values, component-level accuracies, NER
ablations, evidence plausibility, evidence faithfulness, and Charite data set
experiments.

Source summary:

- `CARDIODE400_CV3`: 400 letters in three-fold cross-validation.
- `CARDIODE100_FIXED_TEST`: 100 internally held-out letters.
- Charite atrial fibrillation data set: 27B evaluation on 179 letters.
- Criteria-only systems: physician-curated rules, Qwen 3.5 9B/27B zero-shot,
  and Qwen 3.5 9B/27B LoRA.
- Evidence-generating systems: Qwen 3.5 9B/27B zero-shot and LoRA, plus one
  additional Qwen 3.8 27B LoRA all-evidence run in the aggregate score table.

CARDIO:DE can be requested for research use via [heiDATA: DOI 10.11588/data/AFYQDY](https://doi.org/10.11588/data/AFYQDY).
The public corpus distribution contains `CARDIODE400_main` and
`CARDIODE100_heldout`. CARDIODE100 text can be requested, but its annotations are held in-house and are not published. For this reason, the manuscript reports CARDIODE400 three-fold cross-validation as the reproducible
primary result. The fixed CARDIODE400-train/CARDIODE100-test experiments below
are supplementary results and not reported as main manuscript
results.

## Metric Definitions

- Exact score accuracy: proportion of letters with exactly matching predicted
  and reference total CHA2DS2-VA score.
- `MAE`: mean absolute error between predicted and reference total score.
- `RMSE`: root mean squared error between predicted and reference total score.
  It is retained here as an additional score-level error metric.
- Mean criterion accuracy: mean of the per-criterion accuracies across `C`,
  `H`, `A2`, `D`, `S2`, `V`, and `A`.
- All-criteria exact accuracy: proportion of letters for which the complete
  criteria vector matches the reference exactly.
- `Krippendorff_alpha`: agreement coefficient for predicted and reference
  score/category output.
- `evidence_grounding_rate`: anchored predicted evidence spans divided by all
  predicted evidence spans.
- `evidence_precision_anchored`: matched predicted spans divided by anchored
  predicted spans.
- `evidence_precision_strict`: matched predicted spans divided by all predicted
  spans.
- `evidence_recall_coverage`: matched gold spans divided by all gold spans.
- `sufficiency_preservation_rate`: fraction of evaluable positive criteria that
  remain positive when only model-selected evidence is retained.
- `comprehensiveness_drop_rate`: fraction of evaluable positive criteria that
  become negative after model-selected evidence quotations are removed.
- `line_json_valid_rate`: fraction of NER line-level model responses that are
  valid JSON.
- `line_length_valid_rate`: fraction of NER line-level responses satisfying the
  expected output length constraint.

## Aggregate Score Results

| Evaluation | Model | n letters | RMSE | Krippendorff alpha | Exact score accuracy | MAE | Mean criterion accuracy | All-criteria exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CARDIODE100_FIXED_TEST | Qwen 27B LoRA | 100 | 0.3606 | 1.0000 | 0.9000 | 0.1100 | 0.9857 | 0.9000 |
| CARDIODE100_FIXED_TEST | Qwen 27B LoRA, all evidence | 100 | 0.3742 | 1.0000 | 0.8900 | 0.1200 | 0.9843 | 0.8900 |
| CARDIODE100_FIXED_TEST | Qwen 9B LoRA | 100 | 0.4123 | 1.0000 | 0.8900 | 0.1300 | 0.9843 | 0.8900 |
| CARDIODE100_FIXED_TEST | Qwen 27B zero-shot | 100 | 0.4472 | 0.9353 | 0.8600 | 0.1600 | 0.9729 | 0.8500 |
| CARDIODE100_FIXED_TEST | Qwen 9B LoRA, all evidence | 100 | 0.4472 | 1.0000 | 0.8600 | 0.1600 | 0.9800 | 0.8600 |
| CARDIODE100_FIXED_TEST | Qwen 27B zero-shot, all evidence | 100 | 0.6164 | 0.9084 | 0.7900 | 0.2600 | 0.9571 | 0.7600 |
| CARDIODE100_FIXED_TEST | Qwen 9B zero-shot | 100 | 0.6557 | 0.9326 | 0.7700 | 0.2900 | 0.9586 | 0.7600 |
| CARDIODE100_FIXED_TEST | Rules | 100 | 0.9434 | 0.6370 | 0.4900 | 0.6300 | 0.8714 | 0.4000 |
| CARDIODE100_FIXED_TEST | Qwen 9B zero-shot, all evidence | 100 | 1.0100 | 0.9084 | 0.7100 | 0.4800 | 0.9386 | 0.7000 |
| CARDIODE400_CV3 | Qwen 27B LoRA | 400 | 0.4062 | 0.9508 | 0.9025 | 0.1200 | 0.9850 | 0.9025 |
| CARDIODE400_CV3 | Qwen 9B LoRA | 400 | 0.4123 | 0.9596 | 0.9050 | 0.1200 | 0.9854 | 0.9050 |
| CARDIODE400_CV3 | Qwen 27B LoRA, all evidence | 400 | 0.4153 | 0.9344 | 0.8950 | 0.1275 | 0.9832 | 0.8950 |
| CARDIODE400_CV3 | Qwen 3.8 27B LoRA, all evidence | 400 | 0.4359 | 0.9094 | 0.8700 | 0.1500 | 0.9804 | 0.8700 |
| CARDIODE400_CV3 | Qwen 27B zero-shot | 400 | 0.4637 | 0.8996 | 0.8450 | 0.1750 | 0.9750 | 0.8400 |
| CARDIODE400_CV3 | Qwen 27B zero-shot, all evidence | 400 | 0.5123 | 0.8702 | 0.7975 | 0.2225 | 0.9668 | 0.7900 |
| CARDIODE400_CV3 | Qwen 9B LoRA, all evidence | 400 | 0.5454 | 0.9036 | 0.8600 | 0.1875 | 0.9768 | 0.8575 |
| CARDIODE400_CV3 | Qwen 9B zero-shot | 400 | 0.6325 | 0.7903 | 0.7750 | 0.2750 | 0.9618 | 0.7725 |
| CARDIODE400_CV3 | Qwen 9B zero-shot, all evidence | 400 | 0.9014 | 0.7847 | 0.6950 | 0.4575 | 0.9479 | 0.6925 |
| CARDIODE400_CV3 | Rules | 400 | 1.0149 | 0.6061 | 0.4400 | 0.6950 | 0.8736 | 0.3675 |

## Per-Criterion Accuracy

| Evaluation | Model | C | H | A2 | D | S2 | V | A |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CARDIODE100_FIXED_TEST | Qwen 27B LoRA | 0.9300 | 1.0000 | 1.0000 | 1.0000 | 0.9900 | 0.9800 | 1.0000 |
| CARDIODE100_FIXED_TEST | Qwen 27B LoRA, all evidence | 0.9300 | 0.9900 | 1.0000 | 1.0000 | 0.9900 | 0.9800 | 1.0000 |
| CARDIODE100_FIXED_TEST | Qwen 27B zero-shot | 0.9200 | 0.9700 | 1.0000 | 0.9700 | 0.9900 | 0.9600 | 1.0000 |
| CARDIODE100_FIXED_TEST | Qwen 27B zero-shot, all evidence | 0.8600 | 0.9600 | 1.0000 | 0.9600 | 0.9800 | 0.9400 | 1.0000 |
| CARDIODE100_FIXED_TEST | Qwen 9B LoRA | 0.9300 | 1.0000 | 1.0000 | 1.0000 | 0.9800 | 0.9800 | 1.0000 |
| CARDIODE100_FIXED_TEST | Qwen 9B LoRA, all evidence | 0.9200 | 0.9900 | 1.0000 | 1.0000 | 0.9800 | 0.9700 | 1.0000 |
| CARDIODE100_FIXED_TEST | Qwen 9B zero-shot | 0.9000 | 0.9400 | 1.0000 | 0.9500 | 0.9800 | 0.9400 | 1.0000 |
| CARDIODE100_FIXED_TEST | Qwen 9B zero-shot, all evidence | 0.8700 | 0.9400 | 1.0000 | 0.9200 | 0.9100 | 0.9300 | 1.0000 |
| CARDIODE100_FIXED_TEST | Rules | 0.8700 | 0.8200 | 1.0000 | 0.7100 | 0.9500 | 0.7500 | 1.0000 |
| CARDIODE400_CV3 | Qwen 27B LoRA | 0.9575 | 0.9950 | 1.0000 | 0.9900 | 0.9800 | 0.9725 | 1.0000 |
| CARDIODE400_CV3 | Qwen 27B LoRA, all evidence | 0.9525 | 0.9875 | 1.0000 | 0.9925 | 0.9800 | 0.9700 | 1.0000 |
| CARDIODE400_CV3 | Qwen 27B zero-shot | 0.9450 | 0.9700 | 1.0000 | 0.9750 | 0.9850 | 0.9500 | 1.0000 |
| CARDIODE400_CV3 | Qwen 27B zero-shot, all evidence | 0.9100 | 0.9875 | 1.0000 | 0.9625 | 0.9750 | 0.9325 | 1.0000 |
| CARDIODE400_CV3 | Qwen 3.8 27B LoRA, all evidence | 0.9325 | 0.9925 | 1.0000 | 0.9950 | 0.9775 | 0.9650 | 1.0000 |
| CARDIODE400_CV3 | Qwen 9B LoRA | 0.9625 | 0.9950 | 1.0000 | 0.9925 | 0.9725 | 0.9750 | 1.0000 |
| CARDIODE400_CV3 | Qwen 9B LoRA, all evidence | 0.9350 | 0.9775 | 1.0000 | 0.9900 | 0.9650 | 0.9700 | 1.0000 |
| CARDIODE400_CV3 | Qwen 9B zero-shot | 0.9150 | 0.9575 | 1.0000 | 0.9700 | 0.9725 | 0.9175 | 1.0000 |
| CARDIODE400_CV3 | Qwen 9B zero-shot, all evidence | 0.9150 | 0.9650 | 1.0000 | 0.9600 | 0.8825 | 0.9125 | 1.0000 |
| CARDIODE400_CV3 | Rules | 0.8500 | 0.8700 | 1.0000 | 0.6200 | 0.9500 | 0.8250 | 1.0000 |

## Fixed Split Ablation Results

These ablations use CARDIODE400 as the training set and CARDIODE100 as the
fixed held-out test set. They are supplementary experiments and are not reported
as main manuscript results because the CARDIODE100 CHA2DS2-VA annotations are
held in-house.

| System | Output type | n letters | valid JSON rate | Exact score accuracy | All-criteria exact accuracy | score MAE | micro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules | criteria-only | 100 | NA | 0.4900 | 0.4000 | 0.6300 | 0.8464 |
| Qwen 9B zero-shot | criteria-only | 100 | 1.0000 | 0.7700 | 0.7600 | 0.2900 | 0.9506 |
| Qwen 27B zero-shot | criteria-only | 100 | 1.0000 | 0.8600 | 0.8500 | 0.1600 | 0.9667 |
| Qwen 9B LoRA | criteria-only | 100 | 1.0000 | 0.8900 | 0.8900 | 0.1300 | 0.9805 |
| Qwen 27B LoRA | criteria-only | 100 | 1.0000 | 0.9000 | 0.9000 | 0.1100 | 0.9823 |
| Qwen 9B zero-shot | all evidence | 100 | 0.9700 | 0.7100 | 0.7000 | 0.4800 | 0.9242 |
| Qwen 27B zero-shot | all evidence | 100 | 1.0000 | 0.7900 | 0.7600 | 0.2600 | 0.9464 |
| Qwen 9B LoRA | all evidence | 100 | 1.0000 | 0.8600 | 0.8600 | 0.1600 | 0.9752 |
| Qwen 27B LoRA | all evidence | 100 | 1.0000 | 0.8900 | 0.8900 | 0.1200 | 0.9805 |

### Fixed Split Evidence Plausibility

These are unique-quotation plausibility summaries for CARDIODE100 fixed-split
all-evidence runs.

| System | Predicted evidence spans | Anchored predicted spans | Gold evidence spans | Matched predicted spans | Matched gold spans | Evidence grounding rate | Precision anchored | Precision strict | Recall coverage | F1 anchored | F1 strict | Criterion evidence sufficiency TP | Criterion evidence coverage gold positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 9B zero-shot all evidence | 866 | 808 | 463 | 122 | 122 | 0.9330 | 0.1510 | 0.1409 | 0.2635 | 0.1920 | 0.1836 | 0.4498 | 0.4052 |
| Qwen 27B zero-shot all evidence | 669 | 656 | 463 | 162 | 162 | 0.9806 | 0.2470 | 0.2422 | 0.3499 | 0.2895 | 0.2862 | 0.5943 | 0.5431 |
| Qwen 9B LoRA all evidence | 436 | 434 | 463 | 380 | 380 | 0.9954 | 0.8756 | 0.8716 | 0.8207 | 0.8473 | 0.8454 | 0.9865 | 0.9440 |
| Qwen 27B LoRA all evidence | 441 | 441 | 463 | 390 | 390 | 1.0000 | 0.8844 | 0.8844 | 0.8423 | 0.8628 | 0.8628 | 0.9955 | 0.9569 |

### Fixed Split Evidence Faithfulness

| System | Criterion cases | Full positive cases | Evaluable cases | Sufficiency preserved cases | Comprehensiveness dropped cases | Mean anchored evidence spans | Full JSON valid rate | Sufficiency JSON valid rate | Comprehensiveness JSON valid rate | Sufficiency preservation rate | Comprehensiveness drop rate | Removed still positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 9B zero-shot all evidence | 500 | 249 | 219 | 215 | 84 | 1.6160 | 1.0000 | 1.0000 | 1.0000 | 0.9817 | 0.3836 | 0.6164 |
| Qwen 27B zero-shot all evidence | 500 | 248 | 220 | 219 | 86 | 1.3120 | 1.0000 | 1.0000 | 1.0000 | 0.9955 | 0.3909 | 0.6091 |
| Qwen 9B LoRA all evidence | 500 | 249 | 225 | 221 | 49 | 0.8680 | 1.0000 | 1.0000 | 1.0000 | 0.9822 | 0.2178 | 0.7822 |
| Qwen 27B LoRA all evidence | 500 | 248 | 222 | 220 | 51 | 0.8820 | 1.0000 | 1.0000 | 1.0000 | 0.9910 | 0.2297 | 0.7703 |

## Fixed Split NER Ablation

The NER ablation predicts line-level entities/spans and then derives the
CHA2DS2-VA criteria and score from those entities. This run is useful for
interpretability because it produces explicit entity spans with attributes.
However, it is computationally more demanding than document-level score or
all-evidence generation, and its overall score performance is worse than the
best direct extraction runs.

| System | n letters | Exact score accuracy | All-criteria exact accuracy | score MAE | micro-F1 | line JSON valid rate | line length valid rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 27B NER | 100 | 0.7700 | 0.7600 | 0.2800 | 0.9546 | 0.9981 | 0.9739 |

| NER span metric | Matched entities | Predicted entities | Gold entities | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| value_iou50 | 837 | 1212 | 909 | 0.6906 | 0.9208 | 0.7893 |
| full_attributes_iou50 | 732 | 1212 | 909 | 0.6040 | 0.8053 | 0.6902 |

| Criterion | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| C | 0.7925 | 0.9767 | 0.8750 | 43 |
| H | 0.9872 | 1.0000 | 0.9935 | 77 |
| A2 | 1.0000 | 1.0000 | 1.0000 | 30 |
| D | 1.0000 | 1.0000 | 1.0000 | 24 |
| S2 | 0.8333 | 1.0000 | 0.9091 | 15 |
| V | 0.8690 | 1.0000 | 0.9299 | 73 |
| A | 1.0000 | 1.0000 | 1.0000 | 23 |
| micro avg | 0.9161 | 0.9965 | 0.9546 | 285 |
| macro avg | 0.9260 | 0.9967 | 0.9582 | 285 |
| weighted avg | 0.9229 | 0.9965 | 0.9567 | 285 |
| samples avg | 0.8258 | 0.8867 | 0.8483 | 285 |

## Criteria-Only Overall Results

| System | n letters | valid JSON rate | Exact score accuracy | All-criteria exact accuracy | score MAE | pooled micro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules | 400 | NA | 0.4400 | 0.3675 | 0.6950 | 0.8515 |
| Qwen 9B zero-shot | 400 | 1.0000 | 0.7750 | 0.7725 | 0.2750 | 0.9544 |
| Qwen 27B zero-shot | 400 | 1.0000 | 0.8450 | 0.8400 | 0.1750 | 0.9694 |
| Qwen 9B LoRA | 400 | 1.0000 | 0.9050 | 0.9050 | 0.1200 | 0.9819 |
| Qwen 27B LoRA | 400 | 1.0000 | 0.9025 | 0.9025 | 0.1200 | 0.9815 |

## Criteria-Only Fold Results

| System | Fold | n letters | valid JSON rate | Exact score accuracy | All-criteria exact accuracy | score MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules | 1 | 134 | NA | 0.4328 | 0.3433 | 0.6791 |
| Rules | 2 | 133 | NA | 0.4361 | 0.3684 | 0.7293 |
| Rules | 3 | 133 | NA | 0.4511 | 0.3910 | 0.6767 |
| Qwen 9B zero-shot | 1 | 134 | 1.0000 | 0.8284 | 0.8284 | 0.2015 |
| Qwen 9B zero-shot | 2 | 133 | 1.0000 | 0.6992 | 0.6917 | 0.3534 |
| Qwen 9B zero-shot | 3 | 133 | 1.0000 | 0.7970 | 0.7970 | 0.2707 |
| Qwen 27B zero-shot | 1 | 134 | 1.0000 | 0.8657 | 0.8657 | 0.1493 |
| Qwen 27B zero-shot | 2 | 133 | 1.0000 | 0.8120 | 0.8045 | 0.2180 |
| Qwen 27B zero-shot | 3 | 133 | 1.0000 | 0.8571 | 0.8496 | 0.1579 |
| Qwen 9B LoRA | 1 | 134 | 1.0000 | 0.9179 | 0.9179 | 0.1045 |
| Qwen 9B LoRA | 2 | 133 | 1.0000 | 0.8872 | 0.8872 | 0.1429 |
| Qwen 9B LoRA | 3 | 133 | 1.0000 | 0.9098 | 0.9098 | 0.1128 |
| Qwen 27B LoRA | 1 | 134 | 1.0000 | 0.8955 | 0.8955 | 0.1194 |
| Qwen 27B LoRA | 2 | 133 | 1.0000 | 0.8872 | 0.8872 | 0.1429 |
| Qwen 27B LoRA | 3 | 133 | 1.0000 | 0.9248 | 0.9248 | 0.0977 |

## Criteria-Only Per-Criterion F1

| Criterion | Support | Rules | Qwen 9B zero-shot | Qwen 27B zero-shot | Qwen 9B LoRA | Qwen 27B LoRA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 167 | 0.8446 | 0.9056 | 0.9357 | 0.9558 | 0.9501 |
| H | 310 | 0.9088 | 0.9732 | 0.9810 | 0.9968 | 0.9968 |
| A2 | 125 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| D | 98 | 0.5449 | 0.9423 | 0.9515 | 0.9846 | 0.9798 |
| S2 | 51 | 0.7917 | 0.8932 | 0.9423 | 0.8889 | 0.9216 |
| V | 283 | 0.8732 | 0.9445 | 0.9642 | 0.9824 | 0.9805 |
| A | 98 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| micro avg | 1132 | 0.8515 | 0.9544 | 0.9694 | 0.9819 | 0.9815 |
| macro avg | 1132 | 0.8519 | 0.9513 | 0.9678 | 0.9726 | 0.9755 |
| weighted avg | 1132 | 0.8716 | 0.9551 | 0.9695 | 0.9818 | 0.9816 |
| samples avg | 1132 | 0.7513 | 0.8594 | 0.8763 | 0.8895 | 0.8900 |

## Evidence Plausibility

Plausibility was evaluated on `CARDIODE400_CV3` all-evidence runs with
unique-quotation evidence matching.

| System | Predicted evidence spans | Anchored predicted spans | Gold evidence spans | Matched predicted spans | Matched gold spans | Evidence grounding rate | Precision anchored | Precision strict | Recall coverage | F1 anchored | F1 strict | Criterion evidence sufficiency TP | Criterion evidence coverage gold positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 9B zero-shot all evidence | 3561 | 3309 | 1993 | 559 | 559 | 0.9292 | 0.1689 | 0.1570 | 0.2805 | 0.2109 | 0.2013 | 0.4941 | 0.4587 |
| Qwen 27B zero-shot all evidence | 2734 | 2664 | 1993 | 687 | 687 | 0.9744 | 0.2579 | 0.2513 | 0.3447 | 0.2950 | 0.2907 | 0.5857 | 0.5490 |
| Qwen 9B LoRA all evidence | 1877 | 1856 | 1993 | 1553 | 1553 | 0.9888 | 0.8367 | 0.8274 | 0.7792 | 0.8070 | 0.8026 | 0.9862 | 0.9439 |
| Qwen 27B LoRA all evidence | 1888 | 1876 | 1993 | 1608 | 1608 | 0.9936 | 0.8571 | 0.8517 | 0.8068 | 0.8312 | 0.8287 | 0.9887 | 0.9659 |

### Plausibility By Criterion

| System | Criterion | Predicted evidence spans | Anchored predicted spans | Gold evidence spans | Matched predicted spans | Evidence grounding rate | Precision anchored | Precision strict | Recall coverage | F1 anchored | F1 strict | Criterion evidence sufficiency TP | Criterion evidence coverage gold positive |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 9B zero-shot all evidence | C | 507 | 462 | 595 | 160 | 0.9112 | 0.3463 | 0.3156 | 0.2689 | 0.3027 | 0.2904 | 0.7172 | 0.6228 |
| Qwen 9B zero-shot all evidence | D | 339 | 290 | 123 | 44 | 0.8555 | 0.1517 | 0.1298 | 0.3577 | 0.2131 | 0.1905 | 0.4348 | 0.4082 |
| Qwen 9B zero-shot all evidence | H | 978 | 961 | 387 | 119 | 0.9826 | 0.1238 | 0.1217 | 0.3075 | 0.1766 | 0.1744 | 0.3688 | 0.3581 |
| Qwen 9B zero-shot all evidence | S2 | 199 | 190 | 67 | 8 | 0.9548 | 0.0421 | 0.0402 | 0.1194 | 0.0623 | 0.0602 | 0.1951 | 0.1569 |
| Qwen 9B zero-shot all evidence | V | 1538 | 1406 | 821 | 228 | 0.9142 | 0.1622 | 0.1482 | 0.2777 | 0.2048 | 0.1933 | 0.5811 | 0.5442 |
| Qwen 27B zero-shot all evidence | C | 534 | 506 | 595 | 180 | 0.9476 | 0.3557 | 0.3371 | 0.3025 | 0.3270 | 0.3189 | 0.7319 | 0.6048 |
| Qwen 27B zero-shot all evidence | D | 327 | 301 | 123 | 62 | 0.9205 | 0.2060 | 0.1896 | 0.5041 | 0.2925 | 0.2756 | 0.5918 | 0.5918 |
| Qwen 27B zero-shot all evidence | H | 798 | 798 | 387 | 197 | 1.0000 | 0.2469 | 0.2469 | 0.5090 | 0.3325 | 0.3325 | 0.5696 | 0.5677 |
| Qwen 27B zero-shot all evidence | S2 | 80 | 78 | 67 | 10 | 0.9750 | 0.1282 | 0.1250 | 0.1493 | 0.1379 | 0.1361 | 0.2174 | 0.1961 |
| Qwen 27B zero-shot all evidence | V | 995 | 981 | 821 | 238 | 0.9859 | 0.2426 | 0.2392 | 0.2899 | 0.2642 | 0.2621 | 0.5900 | 0.5442 |
| Qwen 9B LoRA all evidence | C | 563 | 543 | 595 | 426 | 0.9645 | 0.7845 | 0.7567 | 0.7160 | 0.7487 | 0.7358 | 0.9870 | 0.9102 |
| Qwen 9B LoRA all evidence | D | 116 | 116 | 123 | 101 | 1.0000 | 0.8707 | 0.8707 | 0.8211 | 0.8452 | 0.8452 | 0.9368 | 0.9082 |
| Qwen 9B LoRA all evidence | H | 361 | 361 | 387 | 331 | 1.0000 | 0.9169 | 0.9169 | 0.8553 | 0.8850 | 0.8850 | 0.9967 | 0.9742 |
| Qwen 9B LoRA all evidence | S2 | 65 | 65 | 67 | 48 | 1.0000 | 0.7385 | 0.7385 | 0.7164 | 0.7273 | 0.7273 | 0.9756 | 0.7843 |
| Qwen 9B LoRA all evidence | V | 772 | 771 | 821 | 647 | 0.9987 | 0.8392 | 0.8381 | 0.7881 | 0.8128 | 0.8123 | 0.9928 | 0.9717 |
| Qwen 27B LoRA all evidence | C | 590 | 583 | 595 | 466 | 0.9881 | 0.7993 | 0.7898 | 0.7832 | 0.7912 | 0.7865 | 1.0000 | 0.9581 |
| Qwen 27B LoRA all evidence | D | 124 | 124 | 123 | 105 | 1.0000 | 0.8468 | 0.8468 | 0.8537 | 0.8502 | 0.8502 | 0.9388 | 0.9388 |
| Qwen 27B LoRA all evidence | H | 354 | 354 | 387 | 336 | 1.0000 | 0.9492 | 0.9492 | 0.8682 | 0.9069 | 0.9069 | 0.9967 | 0.9839 |
| Qwen 27B LoRA all evidence | S2 | 66 | 64 | 67 | 52 | 0.9697 | 0.8125 | 0.7879 | 0.7761 | 0.7939 | 0.7820 | 0.9778 | 0.8627 |
| Qwen 27B LoRA all evidence | V | 754 | 751 | 821 | 649 | 0.9960 | 0.8642 | 0.8607 | 0.7905 | 0.8257 | 0.8241 | 0.9928 | 0.9788 |

## Faithfulness Overall Results

Faithfulness was evaluated for the five text-derived criteria (`C`, `H`, `D`,
`S2`, and `V`) over 400 letters, giving 2000 criterion cases per system.

| System | criterion cases | full positive cases | evaluable cases | sufficiency preserved cases | comprehensiveness dropped cases | mean anchored evidence spans | full JSON valid rate | sufficiency JSON valid rate | comprehensiveness JSON valid rate | sufficiency preservation rate | comprehensiveness drop rate | removed still positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 9B zero-shot all evidence | 2000 | 992 | 881 | 864 | 309 | 1.6545 | 1.0000 | 1.0000 | 1.0000 | 0.9807 | 0.3507 | 0.6493 |
| Qwen 27B zero-shot all evidence | 2000 | 931 | 873 | 860 | 433 | 1.3320 | 1.0000 | 1.0000 | 1.0000 | 0.9851 | 0.4960 | 0.5040 |
| Qwen 9B LoRA all evidence | 2000 | 908 | 885 | 873 | 464 | 0.9280 | 1.0000 | 1.0000 | 1.0000 | 0.9864 | 0.5243 | 0.4757 |
| Qwen 27B LoRA all evidence | 2000 | 920 | 905 | 897 | 475 | 0.9380 | 1.0000 | 1.0000 | 1.0000 | 0.9912 | 0.5249 | 0.4751 |

## Faithfulness By Criterion

| System | Criterion | full positive cases | evaluable cases | sufficiency preserved cases | comprehensiveness dropped cases | mean anchored evidence spans | sufficiency preservation rate | comprehensiveness drop rate | removed still positive rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 9B zero-shot all evidence | C | 193 | 153 | 151 | 18 | 1.1550 | 0.9869 | 0.1176 | 0.8824 |
| Qwen 9B zero-shot all evidence | D | 110 | 98 | 93 | 39 | 0.7250 | 0.9490 | 0.3980 | 0.6020 |
| Qwen 9B zero-shot all evidence | H | 325 | 306 | 305 | 173 | 2.4025 | 0.9967 | 0.5654 | 0.4346 |
| Qwen 9B zero-shot all evidence | S2 | 52 | 45 | 42 | 38 | 0.4750 | 0.9333 | 0.8444 | 0.1556 |
| Qwen 9B zero-shot all evidence | V | 312 | 279 | 273 | 41 | 3.5150 | 0.9785 | 0.1470 | 0.8530 |
| Qwen 27B zero-shot all evidence | C | 175 | 145 | 143 | 48 | 1.2650 | 0.9862 | 0.3310 | 0.6690 |
| Qwen 27B zero-shot all evidence | D | 108 | 104 | 102 | 50 | 0.7525 | 0.9808 | 0.4808 | 0.5192 |
| Qwen 27B zero-shot all evidence | H | 320 | 313 | 312 | 208 | 1.9950 | 0.9968 | 0.6645 | 0.3355 |
| Qwen 27B zero-shot all evidence | S2 | 53 | 50 | 47 | 43 | 0.1950 | 0.9400 | 0.8600 | 0.1400 |
| Qwen 27B zero-shot all evidence | V | 275 | 261 | 256 | 84 | 2.4525 | 0.9808 | 0.3218 | 0.6782 |
| Qwen 9B LoRA all evidence | C | 171 | 162 | 161 | 53 | 1.3575 | 0.9938 | 0.3272 | 0.6728 |
| Qwen 9B LoRA all evidence | D | 97 | 96 | 95 | 37 | 0.2900 | 0.9896 | 0.3854 | 0.6146 |
| Qwen 9B LoRA all evidence | H | 310 | 305 | 304 | 239 | 0.9025 | 0.9967 | 0.7836 | 0.2164 |
| Qwen 9B LoRA all evidence | S2 | 45 | 41 | 32 | 37 | 0.1625 | 0.7805 | 0.9024 | 0.0976 |
| Qwen 9B LoRA all evidence | V | 285 | 281 | 281 | 98 | 1.9275 | 1.0000 | 0.3488 | 0.6512 |
| Qwen 27B LoRA all evidence | C | 178 | 172 | 172 | 64 | 1.4575 | 1.0000 | 0.3721 | 0.6279 |
| Qwen 27B LoRA all evidence | D | 100 | 100 | 98 | 49 | 0.3100 | 0.9800 | 0.4900 | 0.5100 |
| Qwen 27B LoRA all evidence | H | 310 | 307 | 307 | 241 | 0.8850 | 1.0000 | 0.7850 | 0.2150 |
| Qwen 27B LoRA all evidence | S2 | 51 | 46 | 40 | 34 | 0.1600 | 0.8696 | 0.7391 | 0.2609 |
| Qwen 27B LoRA all evidence | V | 281 | 280 | 280 | 87 | 1.8775 | 1.0000 | 0.3107 | 0.6893 |

## Charite Data Set Results

Updated Qwen 3.5 27B results use all 179 letters. With the original reference,
LoRA achieved 0.8380 exact score accuracy, 0.5069 RMSE, and 0.8981 Krippendorff
alpha; the age-corrected sensitivity analysis yielded 0.8547, 0.4025, and
0.9159, respectively. Predictions were unchanged. See the
[updated Charite results](../roeschl/additional_results.md) for both references
and per-criterion accuracies.
