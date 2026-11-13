# Phase 4 — Is student-risk probability reliable?

## Goal

A model may rank higher-risk students correctly while producing poorly
calibrated percentages. A predicted 0.70 should ideally correspond to
roughly 70% of the target adverse outcome among *similar* students.

EngagePulse uses **Platt scaling** (a logistic regression fitted to the
logit of raw model probabilities). Calibration uses held-out student
records, never the model's training records.

## Reproduce

Works with the initial Phase 1 starter, using three student-disjoint splits:

```bash
python -m engagepulse.calibration --mode grouped --data-dir data/raw/oulad
```

For later-presentation validation, first complete Phase 2:

```bash
python -m engagepulse.compare_models --data-dir data/raw/oulad
python -m engagepulse.calibration --mode temporal --data-dir data/raw/oulad
```

Outputs (not committed): `outputs/calibration_report.json` and
`outputs/calibrated_risk_model.joblib`.

## Evaluation

Compare **Brier score** (lower is better), expected calibration error (ECE,
binned, lower is better), and the reliability bins showing predicted versus
observed adverse-outcome frequencies. Average precision, ROC-AUC and
precision-at-top-10% are also included as operational ranking checks.

Calibration is **not guaranteed to improve** Brier score on unseen data.
It can worsen probability estimates when relationships change over time.

## Research limitations

- Grouped mode does not simulate a future cohort. Prefer prospective
  chronological validation when available.
- In temporal mode, Phase 2 already used the intermediate presentation to
  select the model; this same presentation is then used for calibration.
  Its final presentation was previously evaluated in Phase 2. This is
  useful exploratory evidence, **not an untouched confirmatory test**.
- OULAD's `Fail`/`Withdrawn` course outcomes are not a diagnosis of student
  wellbeing, motivation or intent. The first-four-week model is designed
  only to inform possible supportive human follow-up.
- Never infer that model calibration proves a retention intervention works.
