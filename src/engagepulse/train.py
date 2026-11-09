"""CLI: python -m engagepulse.train --data-dir data/raw/oulad"""
import argparse
import json
from pathlib import Path

import joblib

from engagepulse.features import build_day28_features
from engagepulse.model import evaluate, fit_baseline, split_by_student


def main():
    parser = argparse.ArgumentParser(description="Train EngagePulse day-28 baseline")
    parser.add_argument("--data-dir", default="data/raw/oulad")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    data = build_day28_features(args.data_dir)
    train, test = split_by_student(data)
    if train["at_risk"].nunique() < 2:
        raise ValueError("Training set requires examples from both outcome classes")
    model = fit_baseline(train)
    metrics = evaluate(model, test)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "baseline.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved trained model and metrics to {output}/")


if __name__ == "__main__":
    main()
