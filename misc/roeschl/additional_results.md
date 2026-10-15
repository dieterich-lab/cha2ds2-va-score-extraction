# Charite Data Set Results

The updated supplementary criteria-only CHA2DS2-VA evaluation uses all 179
Charite atrial fibrillation letters and Qwen 3.5 27B zero-shot and LoRA.
Original-reference and age-corrected results evaluate the same predictions
on the same letters.

Source reference: Roeschl, Tobias, et al. "Development of an LLM pipeline
exceeding physician-documented cardiovascular risk scores under routine
clinical conditions." *European Heart Journal - Digital Health* 7(7) (2026):
ztag124. [DOI: 10.1093/ehjdh/ztag124](https://doi.org/10.1093/ehjdh/ztag124).
The source study evaluated CHA2DS2-VASc; the experiments documented here
concern CHA2DS2-VA.

Associated source repository:
[charite-iaim/mcp-risk-server](https://github.com/charite-iaim/mcp-risk-server).
The [source article](https://doi.org/10.1093/ehjdh/ztag124) states that anonymized
data are available on reasonable request to the corresponding author.

These results supplement the manuscript
"Evidence-grounded CHA2DS2-VA extraction from German clinical letters using
locally deployable language models".
[Supplement overview](../cardiode/additional_results.md).

## Score Metrics

Exact score accuracy, MAE, and RMSE follow the
[metric definitions](../cardiode/additional_results.md#metric-definitions).
Ordinal Krippendorff alpha is calculated on binary score categories
(<2 versus >=2).

## Overall Results

| System | Reference | Exact score accuracy | MAE | RMSE | Krippendorff alpha |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen 27B zero-shot | Original | 0.5475 | 0.5140 | 0.8119 | 0.8395 |
| Qwen 27B zero-shot | Age-corrected | 0.5642 | 0.4749 | 0.7437 | 0.8568 |
| Qwen 27B LoRA | Original | 0.8380 | 0.1899 | 0.5069 | 0.8981 |
| Qwen 27B LoRA | Age-corrected | 0.8547 | 0.1508 | 0.4025 | 0.9159 |

The age-corrected reference revises five gold A/A2 labels and corresponding
score totals using dates visible in the letters; predictions are unchanged.
This is a sensitivity analysis, not a replacement for the original reference
or independent verification of original patient ages.

With the revised reference, zero-shot correctly scored 101/179 letters and
LoRA 153/179, a difference of 29.05 percentage points. Age correction added
three exactly correct scores for each model.

## Per-Criterion Accuracy

| Criterion | Zero-shot original | Zero-shot age-corrected | LoRA original | LoRA age-corrected |
| --- | ---: | ---: | ---: | ---: |
| C | 0.6089 | 0.6089 | 0.9274 | 0.9274 |
| H | 0.9721 | 0.9721 | 0.9721 | 0.9721 |
| A2 | 0.9888 | 1.0000 | 0.9888 | 1.0000 |
| D | 0.9944 | 0.9944 | 0.9944 | 0.9944 |
| S2 | 0.9944 | 0.9944 | 0.9944 | 0.9944 |
| V | 0.9497 | 0.9497 | 0.9665 | 0.9665 |
| A | 0.9832 | 1.0000 | 0.9832 | 1.0000 |

After age correction, LoRA had 26 criterion errors: C (13), V (6), H (5),
D (1), and S2 (1). C and V account for 19/26 (73%) of these errors.

## Context Against Roeschl Et Al.

Their GPT-OSS 120B CHA2DS2-VASc pipeline achieved RMSE 0.39 (95% CI 0.31-0.47)
and alpha 0.94 (95% CI 0.83-1.00), compared with 0.4025 and 0.9159 for our
age-corrected 27B LoRA evaluation
([supplementary Table S5](https://doi.org/10.1093/ehjdh/ztag124)).
Different score definitions and the revised age reference preclude claims of
equivalence or superiority.

## Earlier 170-Letter Evaluation

Earlier Qwen 9B/27B results are retained in the
[170-letter tables](../cardiode/additional_results.md#earlier-170-letter-evaluation),
[score summary](score_results.csv), and
[classification report](classification_report.csv).
Those reports describe the earlier subset, not the updated 179-letter evaluation.

The updated numerical results above were supplied as evaluation summaries;
model inference was not rerun for this documentation update.
