#!/usr/bin/env python
# CARDIO:DE CHA2DS2-VA physician-curated rule baseline.
# Score/criteria benchmark. No evidence spans are saved.

from pathlib import Path
import json
import re

import pandas as pd
from datasets import load_from_disk
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = Path("../../../data/cardiode_hf_cv3/cha2ds2_va_CARDIODE400_cv3")
OUTPUT_DIR = REPO_ROOT / "runs" / "cardiode" / "score_benchmark" / "physician_curated_rules"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMETERS = ["C", "H", "A2", "D", "S2", "V", "A"]


PHYSICIAN_RULES = [{'parameter': 'C',
  'clinical_name': 'Heart failure',
  'points': 1,
  'positive_clinical_meaning': 'Clear heart failure diagnosis or reduced LV systolic function. '
                               'Historical/anamnestic events count.',
  'positive_terms': ['Herzinsuffizienz',
                     'Herzschwäche',
                     'chronische Herzinsuffizienz',
                     'dekompensierte Herzinsuffizienz',
                     'kardiale Dekompensation',
                     'cardiale Dekompensation',
                     'Linksherzdekompensation',
                     'kardiogener Schock',
                     'eingeschränkte LV-Funktion',
                     'eingeschränkte systolische LV-Funktion',
                     'hochgradig eingeschränkte globale Pumpfunktion',
                     'reduzierte Pumpfunktion',
                     'LV-Funktion hochgradig eingeschränkt',
                     'HFrEF',
                     'HFmrEF',
                     'HFpEF',
                     'NYHA II-IV',
                     'dilatative Kardiomyopathie',
                     'DCM',
                     'LVEF unterhalb der Norm (siehe numeric_rules)'],
  'technical_regex_parts': ['\\bherzinsuffizienz\\w*',
                            '\\bherzschwäche\\b',
                            '\\b(?:links)?herzdekompensation\\b',
                            '\\b[ck]ardial\\w*\\s+dekompensation\\b',
                            '\\b[ck]ardiogen\\w*\\s+schock\\b',
                            '\\b(?:eingeschränkt\\w*|reduziert\\w*|vermindert\\w*)\\s+(?:\\w+\\s+){0,2}(?:lv[-\\s]?(?:funktion|pumpfunktion)|linksventrikulär\\w*\\s+(?:funktion|pumpfunktion)|pumpfunktion|systolische?\\s+(?:lv[-\\s]?)?funktion)',
                            '\\b(?:lv[-\\s]?|linksventrikulär\\w*\\s+)?(?:funktion|pumpfunktion)\\b[^.,;!?]{0,25}?\\b(?:eingeschränkt|reduziert)\\w*',
                            '\\bhf(?:r|mr|p)ef\\b',
                            '\\bnyha\\s*(?:stadium\\s*)?(?:iv|iii|ii|[234])\\b',
                            '\\bdilatative?\\s+kardiomyopathie\\b',
                            '\\bdcm\\b'],
  'numeric_rules': [{'name': 'LVEF below lower limit of normal',
                     'regex': '(?<!rv-)(?<!rv\\s)\\b(?:lv[-\\s]?)?ef\\b[^%\\d\\n.;!?]{0,20}?(\\d{1,3}(?:[.,]\\d)?)(?:\\s*(?:-|–|bis)\\s*\\d{1,3}(?:[.,]\\d)?)?\\s*(?:%|prozent)',
                     'capture_group': 1,
                     'value_parsing': "Replace ',' with '.' and parse as float ('EF 32,5 %').",
                     'operator': '<',
                     'threshold_male': 52,
                     'threshold_female': 54,
                     'threshold_sex_unknown': 52,
                     'source': 'Lower limits of normal per Lang RM et al., Recommendations for '
                               'cardiac chamber quantification by echocardiography in adults. J Am '
                               'Soc Echocardiogr 2015;28:1-39.'}]},
 {'parameter': 'H',
  'clinical_name': 'Hypertension',
  'points': 1,
  'positive_clinical_meaning': 'Explicit ARTERIAL hypertension or clearly antihypertensive '
                               'treatment. Pulmonary and portal hypertension do not count (see '
                               'exclusions).',
  'positive_terms': ['arterielle Hypertonie',
                     'aHT',
                     'Hypertonie',
                     'Hypertonus',
                     'Bluthochdruck',
                     'hypertensive Herzerkrankung',
                     'hypertensive Entgleisung',
                     'hypertensive Krise',
                     'sekundäre Hypertonie',
                     'antihypertensive Therapie',
                     'blutdrucksenkende Medikation'],
  'technical_regex_parts': ['\\barteriell\\w*\\s+hypertonie\\b',
                            '\\bhypertonie\\b',
                            '\\bhypertonus\\b',
                            '\\bbluthochdruck\\b',
                            '\\baht\\b',
                            '\\bhypertensive?\\s+(?:herzerkrankung|herzkrankheit|entgleisung|krise|kardiomyopathie)\\b',
                            '\\bsekundär\\w*\\s+hypertonie\\b',
                            '\\bantihypertensiv\\w*\\s+(?:therapie|behandlung|medikation|einstellung)\\b',
                            '\\bblutdrucksenkend\\w*\\s+(?:therapie|behandlung|medikation)\\b']},
 {'parameter': 'D',
  'clinical_name': 'Diabetes mellitus',
  'points': 1,
  'positive_clinical_meaning': 'Diabetes mellitus type 1/2 or clearly diabetes-specific '
                               'medication.',
  'positive_terms': ['Diabetes mellitus',
                     'Diabetes mellitus Typ 1',
                     'Diabetes mellitus Typ 2',
                     'DM',
                     'DM Typ 2',
                     'T2DM',
                     'IDDM',
                     'NIDDM',
                     'Diabetiker',
                     'diabetische Nephropathie',
                     'Metformin',
                     'insulinpflichtig'],
  'technical_regex_parts': ['\\bdiabetes\\s+mellitus(?:\\s+typ\\s*[12])?\\b',
                            '\\bdiabet\\w*',
                            '\\bdm\\s*(?:typ\\s*)?[12]\\b',
                            '\\bdm\\b',
                            '\\bdm2\\b',
                            '\\bt[12]dm\\b',
                            '\\biddm\\b',
                            '\\bniddm\\b',
                            '\\bmetformin\\b',
                            '\\binsulinpflichtig\\w*']},
 {'parameter': 'S2',
  'clinical_name': 'Stroke / TIA / arterial thromboembolism',
  'points': 2,
  'positive_clinical_meaning': 'Ischemic stroke, TIA, or arterial thromboembolism. Prior events '
                               'count. Haemorrhagic events and venous thromboembolism do not (see '
                               'exclusions).',
  'positive_terms': ['TIA',
                     'transiente ischämische Attacke',
                     'PRIND',
                     'Amaurosis fugax',
                     'Schlaganfall',
                     'Apoplex',
                     'Hirninfarkt',
                     'Kleinhirninfarkt',
                     'Mediainfarkt',
                     'Media-Infarkt',
                     'Mediateilinfarkt',
                     'Ponsinfarkt',
                     'Thalamusinfarkt',
                     'Stammhirninfarkt',
                     'Hirnstamminfarkt',
                     'Grenzzoneninfarkt',
                     'Posteriorinfarkt',
                     'Territorialinfarkt',
                     'lakunärer Infarkt',
                     'cerebrale Ischämie',
                     'zerebrale Ischämie',
                     'kardioembolischer Infarkt',
                     'Hirninsult',
                     'Insult',
                     'stroke',
                     'arterielle Thromboembolie',
                     'systemische Embolie',
                     'Niereninfarkt',
                     'Milzinfarkt',
                     'Mesenterialinfarkt',
                     'akute Extremitätenischämie'],
  'technical_regex_parts': ['\\btia\\b',
                            '\\btransi(?:ent|torisch)\\w*\\s+ischämisch\\w*\\s+attacke\\b',
                            '\\bprind\\b',
                            '\\bamaurosis\\s+fugax\\b',
                            '\\bschlaganfall\\w*',
                            '\\bapoplex\\w*',
                            '\\binsult\\w*',
                            '\\b(?:hirn|kleinhirn|media|mediateil|mca|pons|thalamus|stammhirn|hirnstamm|grenzzonen|posterior|territorial|okzipital|parietal|temporal|frontal)[-\\s]?infarkt\\w*',
                            '\\blakunär\\w*\\s+(?:infarkt|ischämie|läsion)\\w*',
                            '\\b[cz]erebral\\w*\\s+(?:ischämie|infarkt|embolie)\\w*',
                            '\\bischämisch\\w*\\s+(?:hirninsult|hirninfarkt|insult)\\b',
                            '\\bkardioembolisch\\w*',
                            '\\bstroke\\b',
                            '\\barteriell\\w*\\s+thromboembolie\\b',
                            '\\b(?:systemisch|peripher)\\w*\\s+(?:arteriell\\w*\\s+)?embolie\\b',
                            '\\bniereninfarkt\\b',
                            '\\bmilzinfarkt\\b',
                            '\\bmesenterial(?:infarkt|embolie|ischämie)\\b',
                            '\\bakute?\\s+extremitätenischämie\\b']},
 {'parameter': 'V',
  'clinical_name': 'Vascular disease',
  'points': 1,
  'positive_clinical_meaning': 'PAD, CAD, myocardial infarction, coronary revascularization, or '
                               'relevant aortic plaque (grade II or higher).',
  'positive_terms': ['pAVK',
                     'periphere arterielle Verschlusskrankheit',
                     'AVK',
                     'Claudicatio intermittens',
                     'KHK',
                     'koronare Herzkrankheit',
                     'koronare 1-/2-/3-Gefäßerkrankung',
                     'Koronarsklerose',
                     'Myokardinfarkt',
                     'Herzinfarkt',
                     'Hinterwandinfarkt',
                     'Vorderwandinfarkt',
                     'Re-Infarkt',
                     'STEMI',
                     'NSTEMI',
                     'akutes Koronarsyndrom',
                     'PCI',
                     'PTCA',
                     'Stent',
                     'Koronarstent',
                     'CABG',
                     'ACVB',
                     'aortokoronarer Bypass',
                     'LAD-Stenose',
                     'RIVA-Stenose',
                     'RCX-Stenose',
                     'RCA-Stenose',
                     'LCX-Stenose',
                     'Koronarstenose',
                     'ischämische Kardiomyopathie',
                     'thrombogener Aortenbogen Grad II',
                     'komplexe Aortenplaque'],
  'technical_regex_parts': ['\\bpavk\\b',
                            '\\bperiphere?\\s+arterielle?\\s+verschlusskrankheit\\b',
                            '\\bavk\\b',
                            '\\bclaudicatio\\b',
                            '\\bkhk\\b',
                            '\\bkoronare?\\s+(?:herzkrankheit|herzerkrankung)\\b',
                            '\\b(?:[123]|ein|zwei|drei)[-\\s]?gefä(?:ß|ss)erkrankung\\b',
                            '\\bkoronarsklerose\\b',
                            '\\b(?:myokard|herz|hinterwand|vorderwand|seitenwand)infarkt\\w*',
                            '\\bre[-\\s]?infarkt\\w*',
                            '\\bstemi\\b',
                            '\\bnstemi\\b',
                            '\\bakute?s?\\s+koronarsyndrom\\b',
                            '\\bacs\\b',
                            '\\bpci\\b',
                            '\\bptca\\b',
                            '\\bstent\\w*',
                            '\\bcabg\\b',
                            '\\bacvb\\b',
                            '\\b(?:aorto)?koronar\\w*[-\\s]?bypass\\w*',
                            '\\bbypass[-\\s]?(?:op|operation|versorgung)\\w*',
                            '\\b(?:lad|lcx|rca|riva|rcx|rivp)[-\\s]?(?:stenose|verschluss|stent|läsion)\\w*',
                            '\\bkoronarstenose\\w*',
                            '\\bischämische?\\s+kardiomyopathie\\b',
                            '\\bthrombogener?\\s+aortenbogen\\s+grad\\s*(?:iv|iii|ii|[234])\\b',
                            '\\bkomplexe?\\s+aortenplaque\\w*',
                            '\\baortenplaque\\w*\\s+grad\\s*(?:iv|iii|ii|[234])\\b']}]

PHYSICIAN_EXCLUSIONS = [{'parameter': 'C',
  'clinical_name': 'Heart failure',
  'scope': 'sentence',
  'excluded_meaning': 'Do not count isolated diastolic dysfunction, RV dysfunction, or clearly '
                      'normal/good LV function.',
  'exclusion_terms': ['diastolische Dysfunktion',
                      'RV-Funktion',
                      'rechtsventrikuläre Funktion',
                      'gute LV-Funktion',
                      'normale LV-Funktion',
                      'erhaltene LV-Funktion',
                      'gute linksventrikuläre Pumpfunktion',
                      'normale linksventrikuläre Pumpfunktion'],
  'technical_regex_parts': ['\\bdiastolische?\\s+dysfunktion\\b',
                            '\\brv[-\\s]?funktion\\b',
                            '\\brechtsventrikulär\\w*\\s+(?:funktion|pumpfunktion)\\b',
                            '\\b(?:gute?|normale?|erhaltene?|unauffällige?)\\s+(?:\\w+\\s+){0,1}(?:lv[-\\s]?(?:funktion|pumpfunktion)|linksventrikulär\\w*\\s+(?:funktion|pumpfunktion)|pumpfunktion)\\b']},
 {'parameter': 'H',
  'clinical_name': 'Hypertension',
  'scope': 'span',
  'excluded_meaning': 'Do not count pulmonary, pulmonary arterial, portal, intracranial or ocular '
                      'hypertension - only arterial (systemic) hypertension counts.',
  'exclusion_terms': ['pulmonale Hypertonie',
                      'pulmonalarterielle Hypertonie',
                      'pulmonale Hypertension',
                      'PAH',
                      'portale Hypertonie',
                      'intrakranielle Hypertension',
                      'okuläre Hypertension'],
  'technical_regex_parts': ['\\b(?:pulmonal\\w*|portal\\w*|intrakraniell\\w*|okulär\\w*)\\s+hyperton(?:ie|us|ie|ension)\\b',
                            '\\bpulmonal\\w*\\s+hypertension\\b',
                            '\\bpah\\b']},
 {'parameter': 'D',
  'clinical_name': 'Diabetes mellitus',
  'scope': 'span',
  'excluded_meaning': 'Do not count prediabetes or impaired glucose tolerance.',
  'exclusion_terms': ['Prädiabetes',
                      'Prä-Diabetes',
                      'gestörte Glukosetoleranz',
                      'pathologische Glukosetoleranz',
                      'IGT'],
  'technical_regex_parts': ['\\bprä[-\\s]?diabet\\w*',
                            '\\b(?:gestörte|pathologische|eingeschränkte)\\s+glu[ck]osetoleranz\\b',
                            '\\bigt\\b']},
 {'parameter': 'S2',
  'clinical_name': 'Stroke / TIA / arterial thromboembolism',
  'scope': 'sentence',
  'excluded_meaning': 'Do not count haemorrhagic events, venous thromboembolism, or a mere ward '
                      "mention ('Stroke Unit').",
  'exclusion_terms': ['hämorrhagischer Infarkt',
                      'Hirnblutung',
                      'intrazerebrale Blutung',
                      'ICB',
                      'Subarachnoidalblutung',
                      'SAB',
                      'subdurales Hämatom',
                      'Lungenembolie',
                      'Stroke Unit'],
  'technical_regex_parts': ['\\bhämorrhagisch\\w*',
                            '\\bhirnblutung\\w*',
                            '\\bintra[cz]erebral\\w*\\s+blutung\\w*',
                            '\\bicb\\b',
                            '\\bsubarachnoidalblutung\\w*',
                            '\\bsab\\b',
                            '\\bsub(?:dural|arachnoidal)\\w*\\s+hämatom\\w*',
                            '\\blungenembolie\\w*',
                            '\\bstroke[-\\s]?unit\\b']},
 {'parameter': 'V',
  'clinical_name': 'Vascular disease',
  'scope': 'sentence',
  'excluded_meaning': 'Do not count aneurysm, aortic dissection, MINOCA, carotid disease, mild '
                      'plaques, grade-I aortic arch plaque, normal coronaries, non-significant '
                      'coronary sclerosis, or non-cardiac stents/bypasses.',
  'exclusion_terms': ['Aneurysma',
                      'Aortendissektion',
                      'MINOCA',
                      'Karotisatherosklerose',
                      'Carotisatherosklerose',
                      'Karotisstenose',
                      'leichtgradige Plaques',
                      'thrombogener Aortenbogen Grad I',
                      'unauffällige Koronararterien',
                      'nicht stenosierende Koronarsklerose',
                      'ohne signifikante Stenosen',
                      'Magenbypass',
                      'DJ-Stent',
                      'Harnleiterstent',
                      'Gallengangsstent',
                      'Ösophagusstent',
                      'Stentgraft'],
  'technical_regex_parts': ['\\baneurysma\\w*',
                            '\\baortendissektion\\w*',
                            '\\bminoca\\b',
                            '\\b[ck]arotis(?:atherosklerose|sklerose|stenose|plaque\\w*)\\b',
                            '\\bleichtgradige?\\s+plaques?\\b',
                            '\\bthrombogener?\\s+aortenbogen\\s+grad\\s*(?:i|1)\\b',
                            '\\bunauffällige?\\s+koronar\\w*',
                            '\\bkoronarien\\s+unauffällig\\w*',
                            '\\bnicht\\s+stenosierend\\w*',
                            '\\bohne\\s+(?:signifikante|relevante|höhergradige|stenosierende)\\s+stenose\\w*',
                            '\\bmagen[-\\s]?bypass\\w*',
                            '\\bgastric\\s+bypass\\b',
                            '\\b(?:dj|doppel[-\\s]?j|harnleiter|ureter\\w*|gallen\\w*|ösophagus|tips|aorten)[-\\s]?stent\\w*',
                            '\\bstent[-\\s]?graft\\w*']}]


rule_review_df = pd.DataFrame(
    [
        {
            "parameter": rule["parameter"],
            "clinical_name": rule["clinical_name"],
            "points": rule["points"],
            "positive_clinical_meaning": rule["positive_clinical_meaning"],
            "positive_terms_for_review": "; ".join(rule["positive_terms"]),
        }
        for rule in PHYSICIAN_RULES
    ]
)

exclusion_review_df = pd.DataFrame(
    [
        {
            "parameter": rule["parameter"],
            "clinical_name": rule["clinical_name"],
            "excluded_meaning": rule["excluded_meaning"],
            "exclusion_terms_for_review": "; ".join(rule["exclusion_terms"]),
        }
        for rule in PHYSICIAN_EXCLUSIONS
    ]
)

RULE_PATTERNS = {}
for rule in PHYSICIAN_RULES:
    parameter = rule["parameter"]
    joined_regex = r"\b(?:" + "|".join(rule["technical_regex_parts"]) + r")\b"
    RULE_PATTERNS[parameter] = [re.compile(joined_regex, re.IGNORECASE)]

PARAMETER_EXCLUSION_PATTERNS = {}
for rule in PHYSICIAN_EXCLUSIONS:
    parameter = rule["parameter"]
    joined_regex = r"\b(?:" + "|".join(rule["technical_regex_parts"]) + r")\b"
    PARAMETER_EXCLUSION_PATTERNS[parameter] = re.compile(joined_regex, re.IGNORECASE)

NEGATION_OR_UNCERTAINTY_PATTERN = re.compile(
    r"\b(?:kein(?:e|en|em|er|es)?|ohne|"
    r"nicht(?!\s+(?:insulinpflichtig\w*|obstruktive?\s+kardiomyopathie|dilatiert\w*))|"
    r"verneint|negativ|ausgeschlossen|wurde\s+ausgeschlossen|"
    r"konnte\s+ausgeschlossen\s+werden|"
    r"ließ\s+sich\s+nicht\s+(?:nachweisen|bestätigen|sichern)|"
    r"nicht\s+(?:nachweisbar|bestätigt|nachgewiesen)|"
    r"ausschluss\s+(?:von|einer|eines)|kein\s+hinweis\s+auf|"
    r"keine\s+hinweise\s+auf|kein\s+anhalt\s+für|"
    r"v\.?\s*a\.?|verdacht\s+auf|fraglich\w*|unklar\w*|"
    r"differentialdiagnostisch\w*|(?-i:DD))\b",
    re.IGNORECASE | re.UNICODE,
)

LVEF_PATTERN = re.compile(
    r"\b(?:lvef|lv-ef|ef|ejection\s+fraction|auswurffraktion)\s*"
    r"(?:von|:|=|bei)?\s*(?P<value>\d{1,2}(?:[,.]\d+)?)\s*%",
    re.IGNORECASE,
)

folds = []
for fold_number in range(1, 4):
    fold = load_from_disk(DATASET_PATH / f"cha2ds2_va_fold_{fold_number}")
    folds.append(fold)

print("Test letters per fold:", [len(fold["test"]) for fold in folds])

results = []
total_test_letters = sum(len(fold["test"]) for fold in folds)

with tqdm(total=total_test_letters, desc="Rule-based inference", unit="letter") as progress:
    for fold_number, fold in enumerate(folds, start=1):
        for row in fold["test"]:
            text = row["text"]
            positive_matches = {"C": [], "H": [], "D": [], "S2": [], "V": []}

            for parameter in positive_matches:
                matched_phrases = []
                for pattern in RULE_PATTERNS[parameter]:
                    for match in pattern.finditer(text):
                        left_context = text[max(0, match.start() - 80):match.start()]
                        right_context = text[match.end():min(len(text), match.end() + 50)]
                        full_context = text[max(0, match.start() - 80):min(len(text), match.end() + 80)]

                        if (
                            NEGATION_OR_UNCERTAINTY_PATTERN.search(left_context)
                            or NEGATION_OR_UNCERTAINTY_PATTERN.search(right_context)
                        ):
                            continue

                        if parameter in PARAMETER_EXCLUSION_PATTERNS:
                            if PARAMETER_EXCLUSION_PATTERNS[parameter].search(full_context):
                                continue

                        phrase = match.group(0).strip()
                        if phrase not in matched_phrases:
                            matched_phrases.append(phrase)

                positive_matches[parameter] = matched_phrases

            for match in LVEF_PATTERN.finditer(text):
                lvef = float(match.group("value").replace(",", "."))
                if lvef > 50:
                    continue

                left_context = text[max(0, match.start() - 80):match.start()]
                right_context = text[match.end():min(len(text), match.end() + 50)]
                full_context = text[max(0, match.start() - 80):min(len(text), match.end() + 80)]

                if (
                    NEGATION_OR_UNCERTAINTY_PATTERN.search(left_context)
                    or NEGATION_OR_UNCERTAINTY_PATTERN.search(right_context)
                ):
                    continue

                if PARAMETER_EXCLUSION_PATTERNS["C"].search(full_context):
                    continue

                phrase = match.group(0).strip()
                if phrase not in positive_matches["C"]:
                    positive_matches["C"].append(phrase)

            predicted_criteria = {
                "C": 1 if positive_matches["C"] else 0,
                "H": 1 if positive_matches["H"] else 0,
                "A2": 0,
                "D": 1 if positive_matches["D"] else 0,
                "S2": 2 if positive_matches["S2"] else 0,
                "V": 1 if positive_matches["V"] else 0,
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

            gold_criteria = {"C": 0, "H": 0, "A2": 0, "D": 0, "S2": 0, "V": 0, "A": 0}
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

            results.append(
                {
                    "fold": fold_number,
                    "letter_id": row["letter_id"],
                    "gold_score": sum(gold_criteria.values()),
                    "predicted_score": sum(predicted_criteria.values()),
                    "gold_criteria": gold_criteria,
                    "predicted_criteria": predicted_criteria,
                }
            )

            progress.update(1)

results_df = pd.DataFrame(results)
results_df["score_correct"] = results_df["predicted_score"] == results_df["gold_score"]
results_df["criteria_correct"] = [
    predicted == gold
    for predicted, gold in zip(results_df["predicted_criteria"], results_df["gold_criteria"])
]
results_df["absolute_error"] = (
    pd.to_numeric(results_df["predicted_score"], errors="coerce") - results_df["gold_score"]
).abs()

fold_summary = (
    results_df.groupby("fold")
    .agg(
        n_letters=("letter_id", "count"),
        score_accuracy=("score_correct", "mean"),
        criteria_accuracy=("criteria_correct", "mean"),
        score_mae=("absolute_error", "mean"),
    )
    .reset_index()
)

overall_summary = pd.DataFrame(
    {
        "n_letters": [len(results_df)],
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
micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

print("\nFold summary:")
print(fold_summary.round(4).to_string(index=False))
print("\nOverall summary:")
print(overall_summary.round(4).to_string(index=False))
print(f"\nPooled micro-F1: {micro_f1:.4f}")
print(classification_report_df.to_string())

results_df.to_json(OUTPUT_DIR / "test_results.json", orient="records", force_ascii=False, indent=2)
fold_summary.to_csv(OUTPUT_DIR / "fold_summary.csv", index=False)
overall_summary.to_csv(OUTPUT_DIR / "overall_summary.csv", index=False)
classification_report_df.to_csv(OUTPUT_DIR / "classification_report.csv")
rule_review_df.to_csv(OUTPUT_DIR / "physician_rules.csv", index=False)
exclusion_review_df.to_csv(OUTPUT_DIR / "physician_exclusions.csv", index=False)

with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "pipeline": "physician_curated_rules",
            "task": "score_criteria_extraction",
            "evidence_output": False,
            "dataset_path": str(DATASET_PATH),
            "parameters": PARAMETERS,
        },
        file,
        ensure_ascii=False,
        indent=2,
    )

print("\nSaved outputs to:")
print(OUTPUT_DIR)
