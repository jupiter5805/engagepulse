# EngagePulse — Research-Led Student Retention Intelligence

Independent portfolio project: an explainable early-warning system to help university staff identify **students who may benefit from support**, using observed learning activity rather than opaque risk labels.

> **Status: Phase 1 baseline prototype.** No production deployment, validated intervention benefit, or real-student risk predictions are claimed.

## Why this project exists

Inspired by the public requirements of the Kortext / University of Stirling KTP Associate — Data Scientist (Engagement Analytics) vacancy. **Not affiliated with Kortext or the University of Stirling.** The aim is to demonstrate behavioural feature engineering, ML, empirical validation, research-to-model translation and adoption-oriented design.

## Data

Uses the **Open University Learning Analytics Dataset (OULAD)**, licensed CC BY 4.0:

- Official page: https://research.stem.open.ac.uk/ouanalyse/dataset/
- UCI download: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
- Citation: Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics Dataset.* Scientific Data 4, 170171. https://doi.org/10.1038/sdata.2017.171

**Download the dataset yourself**. Put `studentInfo.csv`, `studentRegistration.csv`, and `studentVle.csv` in `data/raw/oulad/`. Don't push those CSVs to this repository.

## Quick start (Mac/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
# After downloading and extracting OULAD to data/raw/oulad:
python -m engagepulse.train --data-dir data/raw/oulad
```

The model writes `outputs/baseline.joblib` and `outputs/metrics.json` locally. No metric is invented: actual values depend on the completed training run.

## What Phase 1 does

- Builds an enrolment snapshot at the **end of course day 27** (four completed weeks).
- Excludes enrolments that already ended by day 27 (reduces obvious target leakage).
- Uses only VLE activity from course days 0–27.
- Converts actions into click totals, active days, resource breadth, recency, weekly clicks, and recent engagement trend.
- Predicts future final **Fail or Withdrawn** outcome as an illustrative risk proxy, not as a causal explanation.
- Splits by student identifier to avoid a student's repeated enrolments leaking across train/test.
- Trains a transparent Logistic Regression baseline.
- Reports average precision, ROC-AUC, Brier score, and precision/recall at the highest-risk 10%.

## Critical limitations

- An association with withdrawal/failure does **not** prove that an engagement feature causes the outcome, or that any intervention will work.
- The dataset is from historical courses; results are **not** demonstrated to generalise to current UK institutions.
- A student with few VLE clicks may be studying offline, using accessibility accommodations, or facing technical issues.
- The initial grouped random split is **not** a forward-in-time deployment test. A later phase must train on earlier course presentations and test on future presentations and different modules, without leakage.
- No protected characteristics are used as model inputs in the baseline. A later audit must evaluate performance and potential harms by subgroup, with appropriate governance.
- Predictions should support, never replace, professional judgement and a student-centred support conversation.

See `docs/RESEARCH_PLAN.md` and `docs/BUILD_ROADMAP.md` for the planned research, evaluation and implementation stages.

## Phase 1 — Experimental Results

Dataset: Open University Learning Analytics Dataset (OULAD)
Prediction point: End of the first four course weeks
Target: Subsequent Fail or Withdrawn outcome
Model: Logistic Regression

### Held-out test results
- Test observations: 5,515
- At-risk prevalence: 44.06%
- ROC-AUC: 0.6817
- Average precision: 0.6276
- Brier score: 0.2244
- Precision at top 10%: 75.91%
- Recall at top 10%: 17.24%

These results demonstrate an initial research baseline,
not validated student-retention improvements.

Next: temporal validation, Random Forest comparison,
probability calibration and explainability.

## Reproducing the ML Experiments

EngagePulse uses the Open University Learning Analytics Dataset (OULAD) to study early indicators of student disengagement.

### Run the Project

1. Download OULAD and extract the CSV files into `data/raw/oulad/`.
2. Install dependencies using `pip install -e '.[dev]'`.
3. Run automated tests with `pytest -q`.
4. Train the baseline model:

   `python -m engagepulse.train --data-dir data/raw/oulad`

5. Review the evaluation metrics in `outputs/metrics.json`.

### Research Principles

- Use only information available at the prediction date.
- Separate training and evaluation data to reduce leakage.
- Evaluate both predictive performance and practical usefulness.
- Treat predictions as decision-support evidence, not automatic decisions about students.
- Keep raw student data out of version control.

## Phase 3 — Explainable AI for Student Retention

EngagePulse now includes model explainability to help
understand the factors associated with student disengagement.

### Key Features

- Predict student risk using early learning activity.
- Analyse engagement behaviour using machine learning.
- Evaluate models using ROC-AUC and average precision.
- Identify influential features using permutation importance.
- Generate reproducible feature-importance reports.
- Support transparent, research-led decision-making.

### Generate Model Explanations

Run:

    python -m engagepulse.explain --data-dir data/raw/oulad

Results are saved to:

    outputs/feature_importance.json

### Why Explainability Matters

For universities, identifying potentially at-risk students
is only part of the challenge.

Understanding which observed engagement patterns are
associated with model predictions can help researchers
evaluate risk indicators and support academic staff.

Permutation importance measures overall model reliance
on features. It does not establish causation or explain
an individual student's circumstances.

### Development Roadmap

- [x] Phase 1: Student risk prediction baseline
- [ ] Phase 2: Temporal model comparison and validation
- [x] Phase 3: Global model explainability
- [ ] Phase 4: Probability calibration
- [ ] Phase 5: Student engagement analytics dashboard
- [ ] Phase 6: FastAPI model deployment

This independent research project uses the Open University
Learning Analytics Dataset (OULAD) and is designed to
explore responsible, evidence-based student retention analytics.
