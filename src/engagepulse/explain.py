"""Global, model-agnostic feature importance for EngagePulse.

Usage:
    python -m engagepulse.explain --data-dir data/raw/oulad
    python -m engagepulse.explain --data-dir data/raw/oulad --mode temporal

This describes predictive associations, not causes of withdrawal.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from engagepulse.features import FEATURES, build_day28_features
from engagepulse.model import split_by_student


def global_feature_importance(model, holdout: pd.DataFrame, repeats: int = 3):
    """Mean fall in average precision when one held-out feature is shuffled."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if holdout.empty or holdout["at_risk"].nunique() != 2:
        raise ValueError("Holdout must contain both outcome classes")

    X = holdout[FEATURES]
    y = holdout["at_risk"].astype(int)
    baseline_ap = average_precision_score(y, model.predict_proba(X)[:, 1])
    result = permutation_importance(
        model, X, y, scoring="average_precision",
        n_repeats=repeats, random_state=42, n_jobs=1,
    )
    ranked = sorted(
        (
            {
                "feature": name,
                "mean_drop_in_average_precision": round(float(result.importances_mean[i]), 5),
                "std_dev": round(float(result.importances_std[i]), 5),
            }
            for i, name in enumerate(FEATURES)
        ),
        key=lambda row: row["mean_drop_in_average_precision"],
        reverse=True,
    )
    return {
        "n_holdout": len(holdout),
        "baseline_average_precision": round(float(baseline_ap), 5),
        "permutation_repeats": repeats,
        "importance": ranked,
        "interpretation": (
            "Higher positive values indicate stronger predictive reliance on a feature "
            "within this holdout. Correlated features can obscure importance. "
            "These are not causal explanations or individual support recommendations."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Explain EngagePulse student-risk model")
    parser.add_argument("--data-dir", default="data/raw/oulad")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--mode", choices=("baseline", "temporal"), default="baseline")
    args = parser.parse_args()

    output = Path(args.output_dir)
    frame = build_day28_features(args.data_dir)

    if args.mode == "temporal":
        from engagepulse.compare_models import split_by_presentation
        _, holdout, _, split_info = split_by_presentation(frame)
        model_path = output / "temporal_champion.joblib"
        population = f"validation presentation {split_info['validation_presentation']}"
    else:
        _, holdout = split_by_student(frame)
        model_path = output / "baseline.joblib"
        population = "grouped random held-out students"

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing trained model: {model_path}. Run training first.")
    model = joblib.load(model_path)  # Only load models you trained locally and trust.
    report = global_feature_importance(model, holdout)
    report["mode"] = args.mode
    report["explanation_population"] = population
    output.mkdir(parents=True, exist_ok=True)
    (output / "feature_importance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output / 'feature_importance.json'}")


if __name__ == "__main__":
    main()
