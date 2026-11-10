"""Compare student-risk classifiers on later OULAD presentations.

Run: python -m engagepulse.compare_models --data-dir data/raw/oulad

Selection uses the *validation* presentation only. The held-out latest
presentation is evaluated once after model selection. A student's ID
never occurs in more than one split.
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from engagepulse.features import FEATURES, build_day28_features


def presentation_order(presentation: str) -> tuple[int, int]:
    """OULAD presentation codes, e.g. 2013B precedes 2013J."""
    match = re.fullmatch(r"(\d{4})([BJ])", str(presentation))
    if not match:
        raise ValueError(f"Unexpected course presentation label: {presentation!r}")
    return int(match.group(1)), {"B": 0, "J": 1}[match.group(2)]


def split_by_presentation(frame: pd.DataFrame):
    """Earlier presentations train; penultimate validates; latest tests.

    Repeated learners are allocated exclusively to their *latest* split
    to avoid train/validation/test student identity overlap.
    """
    presentations = sorted(frame["code_presentation"].dropna().unique(), key=presentation_order)
    if len(presentations) < 3:
        raise ValueError("Need at least three distinct course presentations for temporal evaluation")

    val_code, test_code = presentations[-2:]
    train_codes = presentations[:-2]

    train = frame.loc[frame["code_presentation"].isin(train_codes)].copy()
    val = frame.loc[frame["code_presentation"] == val_code].copy()
    test = frame.loc[frame["code_presentation"] == test_code].copy()

    # Explicitly exclude repeat enrollees from earlier splits: the
    # resulting validation and test represent unseen student identities.
    test_students = set(test["id_student"])
    val = val.loc[~val["id_student"].isin(test_students)].copy()
    val_students = set(val["id_student"])
    train = train.loc[~train["id_student"].isin(test_students | val_students)].copy()

    for label, subset in (("training", train), ("validation", val), ("test", test)):
        if subset.empty or subset["at_risk"].nunique() != 2:
            raise ValueError(f"{label} split is empty or lacks both outcome classes")

    assert set(train.id_student).isdisjoint(val.id_student)
    assert set(train.id_student).isdisjoint(test.id_student)
    assert set(val.id_student).isdisjoint(test.id_student)

    metadata = {
        "training_presentations": list(train_codes),
        "validation_presentation": val_code,
        "test_presentation": test_code,
        "n_train": len(train),
        "n_validation": len(val),
        "n_test": len(test),
        "repeat_enrolments_removed_from_earlier_splits": len(frame) - len(train) - len(val) - len(test),
    }
    return train, val, test, metadata


def candidates():
    """Fixed model settings, selected without looking at test labels."""
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=160, min_samples_leaf=10, max_depth=15,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.06, max_leaf_nodes=15,
                l2_regularization=2.0, random_state=42,
            ),
        ),
    }


def evaluate_risk(model, subset: pd.DataFrame) -> dict:
    """Operational evaluation where only 10% of learners can be reviewed."""
    y = subset["at_risk"].to_numpy(dtype=int)
    probabilities = model.predict_proba(subset[FEATURES])[:, 1]
    top_n = max(1, int(np.ceil(len(y) * 0.10)))
    top_idx = np.argsort(-probabilities, kind="stable")[:top_n]

    positives = int(y.sum())
    return {
        "n_examples": len(y),
        "at_risk_prevalence": round(float(y.mean()), 4),
        "average_precision": round(float(average_precision_score(y, probabilities)), 4),
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 4) if len(set(y)) == 2 else None,
        "brier_score": round(float(brier_score_loss(y, probabilities)), 4),
        "precision_at_top_10pct": round(float(y[top_idx].mean()), 4),
        "recall_at_top_10pct": round(float(y[top_idx].sum() / positives), 4) if positives else None,
    }


def run_comparison(frame: pd.DataFrame):
    """Select using validation AP, evaluate the untouched test set once."""
    train, val, test, split = split_by_presentation(frame)

    metrics = {}
    fitted = {}
    scores = {}
    for name, model in candidates().items():
        model.fit(train[FEATURES], train["at_risk"])
        scores[name] = float(average_precision_score(
            val["at_risk"], model.predict_proba(val[FEATURES])[:, 1]
        ))
        metrics[name] = evaluate_risk(model, val)
        fitted[name] = model

    # Defined order breaks exact ties reproducibly, without accessing test labels.
    winner = max(scores, key=scores.get)
    test_metrics = evaluate_risk(fitted[winner], test)
    results = {
        "experiment": "OULAD day-28 student risk: held-out presentations",
        "target": "Fail or Withdrawn (among learners enrolled at day 28)",
        "model_selection_metric": "validation average precision",
        "split": split,
        "validation_metrics": metrics,
        "selected_model": winner,
        "test_metrics_for_selected_model_only": test_metrics,
        "note": "Predictions are for human review, not automatic decisions. No causal retention benefit established.",
    }
    return results, fitted[winner]


def main():
    parser = argparse.ArgumentParser(description="Compare EngagePulse models on later OULAD presentations")
    parser.add_argument("--data-dir", default="data/raw/oulad")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    frame = build_day28_features(args.data_dir)
    results, winner = run_comparison(frame)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "temporal_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    joblib.dump(winner, output / "temporal_champion.joblib")
    print(json.dumps(results, indent=2))
    print(f"Saved results and selected model to {output}/")


if __name__ == "__main__":
    main()
