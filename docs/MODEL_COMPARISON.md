# Phase 2: temporal model comparison

## Research question
Can student-engagement signals observed during course days 0–27 identify later
Fail or Withdrawn outcomes on **unseen later course presentations**?

## Run

```bash
python -m engagepulse.compare_models --data-dir data/raw/oulad
```

This builds the same day-28 feature snapshot as Phase 1, then uses the
chronological presentation codes (e.g., 2013B, 2013J, 2014B, 2014J):

- Earlier presentations: model training.
- Penultimate presentation: validation and model selection.
- Latest presentation: final test, used **only once** after selecting a model.

Repeated student IDs are removed from earlier partitions to keep splits
student-disjoint. The report records how many enrolments this removes.
This is a *presentation-held-out* experiment, not continuous longitudinal
validation or evidence of transfer to a different university.

## Models and selection

- Class-weighted Logistic Regression.
- Random Forest.
- Histogram Gradient Boosting.

Select the model with the highest **validation average precision**. Report
ROC-AUC, Brier score and precision/recall at the top 10% risk-ranked learners
as secondary measures. Do not choose a model using test-set metrics.

## Limitations

- `Fail or Withdrawn` is a coarse proxy for an adverse outcome, not directly
  observed disengagement or a student's psychological state.
- VLE click activity is an incomplete measure of participation and can vary
  by module, modality, or accessibility needs.
- Feature engineering uses no data after the end of course day 27, but OULAD
  labels are retrospective and observational.
- Any learner who unregistered by day 27 is deliberately excluded; scores
  therefore apply only to those still in the eligible cohort.
- Missing registration dates are presently treated as eligible in the Phase 1
  snapshot; this assumption should be validated and documented.
- No production use, automated punitive decisions, or causal claims.

## Outputs

- `outputs/temporal_comparison.json`: validation metrics for all candidates,
  held-out test metrics for **only the selected candidate**.
- `outputs/temporal_champion.joblib`: selected fitted model.

The OULAD raw data and generated model artifacts must stay out of Git.
