"""Day-28 student-risk probability calibration.

Quick start (works with Phase 1 baseline):
    python -m engagepulse.calibration --data-dir data/raw/oulad

Stronger, later-presentation evaluation (requires Phase 2):
    python -m engagepulse.compare_models --data-dir data/raw/oulad
    python -m engagepulse.calibration --mode temporal --data-dir data/raw/oulad

Only train-set labels fit the risk model. Only calibration-set labels fit the
sigmoid. The evaluation split is held aside until *both* models are frozen.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from engagepulse.features import FEATURES, build_day28_features
from engagepulse.model import fit_baseline


EPS = 1e-6


def _logits(probabilities):
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected a one-dimensional array of finite probabilities")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must lie between 0 and 1")
    clipped = np.clip(values, EPS, 1 - EPS)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


@dataclass
class SigmoidCalibratedModel:
    """A fitted model with Platt's sigmoid calibration layer."""

    model: object
    calibrator: LogisticRegression

    def predict_proba(self, X):
        raw_probabilities = self.model.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict_proba(_logits(raw_probabilities))[:, 1]
        return np.column_stack((1 - calibrated, calibrated))


def fit_calibrator(model, calibration_frame: pd.DataFrame):
    """Fit a sigmoid to predictions made on never-trained-on learners."""
    if calibration_frame.empty or calibration_frame["at_risk"].nunique() != 2:
        raise ValueError("Calibration data must contain both outcome classes")
    raw = model.predict_proba(calibration_frame[FEATURES])[:, 1]
    calibrator = LogisticRegression(
        C=1000.0, max_iter=1000, solver="lbfgs", random_state=42
    )
    calibrator.fit(_logits(raw), calibration_frame["at_risk"].astype(int))
    return SigmoidCalibratedModel(model=model, calibrator=calibrator)


def split_grouped_three_way(frame: pd.DataFrame):
    """64% train, 16% calibration, 20% test, grouped by student ID."""
    if len(frame) < 15:
        raise ValueError("Not enough student records for three-way evaluation")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    development_idx, test_idx = next(
        splitter.split(frame, frame["at_risk"], groups=frame["id_student"])
    )
    development, test = frame.iloc[development_idx], frame.iloc[test_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=43)
    training_idx, calibration_idx = next(
        inner.split(development, development["at_risk"], groups=development["id_student"])
    )
    training = development.iloc[training_idx].copy()
    calibration = development.iloc[calibration_idx].copy()
    test = test.copy()
    for name, subset in (("training", training), ("calibration", calibration), ("test", test)):
        if subset.empty or subset["at_risk"].nunique() != 2:
            raise ValueError(f"{name} split is empty or contains only one class")
    train_ids, cal_ids, test_ids = (set(s["id_student"]) for s in (training, calibration, test))
    if (train_ids & cal_ids) or (train_ids & test_ids) or (cal_ids & test_ids):
        raise AssertionError("Student identity leakage between evaluation splits")
    return training, calibration, test


def reliability_report(y_true, probabilities, n_bins=10):
    """Compute Brier score and fixed-width risk calibration bins."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels and probabilities must be paired binary observations")
    if n_bins < 2:
        raise ValueError("Use at least two reliability bins")
    _logits(p)  # Also validates ranges, NaNs and dimension.

    bin_ids = np.minimum(np.floor(p * n_bins).astype(int), n_bins - 1)
    bins = []
    ece = 0.0
    for idx in range(n_bins):
        mask = bin_ids == idx
        if not mask.any():
            continue
        predicted = float(p[mask].mean())
        observed = float(y[mask].mean())
        n = int(mask.sum())
        ece += n / len(y) * abs(predicted - observed)
        bins.append({
            "risk_interval": [round(idx / n_bins, 2), round((idx + 1) / n_bins, 2)],
            "n": n,
            "mean_predicted_risk": round(predicted, 4),
            "observed_outcome_rate": round(observed, 4),
        })
    top_n = max(1, int(np.ceil(len(y) * .10)))
    top = np.argsort(-p, kind="stable")[:top_n]
    positives = int(y.sum())
    return {
        "n": len(y),
        "prevalence": round(float(y.mean()), 4),
        "brier_score": round(float(brier_score_loss(y, p)), 4),
        "expected_calibration_error_10_bins": round(float(ece), 4) if n_bins == 10 else None,
        "average_precision": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) == 2 else None,
        "precision_at_top_10pct": round(float(y[top].mean()), 4),
        "recall_at_top_10pct": round(float(y[top].sum() / positives), 4) if positives else None,
        "reliability_bins": bins,
    }


def compare_calibration(model, calibrator, test_frame):
    """Evaluate frozen raw/calibrated predictors on the same test students."""
    X = test_frame[FEATURES]
    y = test_frame["at_risk"].to_numpy(dtype=int)
    raw = model.predict_proba(X)[:, 1]
    adjusted = calibrator.predict_proba(X)[:, 1]
    return {
        "uncalibrated": reliability_report(y, raw),
        "sigmoid_calibrated": reliability_report(y, adjusted),
    }


def main():
    parser = argparse.ArgumentParser(description="EngagePulse model probability calibration")
    parser.add_argument("--data-dir", default="data/raw/oulad")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--mode", choices=("grouped", "temporal"), default="grouped")
    args = parser.parse_args()
    frame = build_day28_features(args.data_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.mode == "temporal":
        try:
            from engagepulse.compare_models import split_by_presentation
        except ImportError as exc:
            raise SystemExit("Phase 2 is required for temporal calibration. Install Phase 2 first.") from exc
        model_file = output / "temporal_champion.joblib"
        if not model_file.is_file():
            raise SystemExit("Run 'python -m engagepulse.compare_models --data-dir data/raw/oulad' first.")
        _, calibration_set, test_set, split_meta = split_by_presentation(frame)
        model = joblib.load(model_file)  # Only load locally generated, trusted artefacts.
        caveat = (
            "Phase 2 used this calibration presentation for model selection, and previously "
            "evaluated the same test presentation. Treat results as exploratory; use a new "
            "untouched cohort for confirmatory evaluation."
        )
    else:
        training, calibration_set, test_set = split_grouped_three_way(frame)
        model = fit_baseline(training)
        split_meta = {
            "n_train": len(training),
            "n_calibration": len(calibration_set),
            "n_test": len(test_set),
            "strategy": "student-grouped, 64/16/20 approximate split",
        }
        caveat = (
            "Grouped-random cohort split; estimates may not generalise to later course "
            "presentations. This is exploratory decision-support research."
        )

    calibrated_model = fit_calibrator(model, calibration_set)
    metrics = compare_calibration(model, calibrated_model, test_set)
    report = {
        "mode": args.mode,
        "dataset": "OULAD student outcome labels at course day 28",
        "target": "Fail or Withdrawn after day 27",
        "calibration_method": "Platt scaling / logistic regression on held-out risk logits",
        "split": split_meta,
        "metrics": metrics,
        "important_caveat": caveat,
        "clinical_or_automatic_decisions": "Not appropriate; human support and review only",
    }
    (output / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    joblib.dump(calibrated_model, output / "calibrated_risk_model.joblib")
    print(json.dumps({
        "mode": args.mode,
        "split": split_meta,
        "uncalibrated_brier": metrics["uncalibrated"]["brier_score"],
        "calibrated_brier": metrics["sigmoid_calibrated"]["brier_score"],
        "uncalibrated_precision_at_top_10pct": metrics["uncalibrated"]["precision_at_top_10pct"],
        "calibrated_precision_at_top_10pct": metrics["sigmoid_calibrated"]["precision_at_top_10pct"],
        "note": "Lower Brier is better; calibration can also make results worse.",
    }, indent=2))
    print(f"Saved full report and calibrated model to {output}/")


if __name__ == "__main__":
    main()
