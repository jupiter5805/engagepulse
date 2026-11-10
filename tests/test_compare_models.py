import numpy as np
import pandas as pd
import pytest

from engagepulse.compare_models import (
    evaluate_risk, presentation_order, run_comparison, split_by_presentation,
)
from engagepulse.features import FEATURES


def demo_data(n=40):
    rng = np.random.default_rng(42)
    rows = []
    for cycle, label in enumerate(("2013B", "2013J", "2014B", "2014J")):
        for idx in range(n):
            # Reuse some learner IDs across presentations to test isolation.
            student_id = idx if idx < 4 else cycle * 1000 + idx
            risk = idx % 2
            row = {
                "code_presentation": label, "id_student": student_id,
                "at_risk": risk,
            }
            for feature in FEATURES:
                row[feature] = float(risk + rng.normal(scale=0.5))
            rows.append(row)
    return pd.DataFrame(rows)


def test_presentation_order_is_chronological():
    assert sorted(["2014J", "2013J", "2014B", "2013B"], key=presentation_order) == [
        "2013B", "2013J", "2014B", "2014J"
    ]


def test_bad_presentation_code_fails():
    with pytest.raises(ValueError, match="Unexpected"):
        presentation_order("latest")


def test_temporal_split_keeps_students_disjoint():
    training, validation, test, info = split_by_presentation(demo_data())
    assert info["validation_presentation"] == "2014B"
    assert info["test_presentation"] == "2014J"
    assert set(training.id_student).isdisjoint(validation.id_student)
    assert set(training.id_student).isdisjoint(test.id_student)
    assert set(validation.id_student).isdisjoint(test.id_student)
    assert info["repeat_enrolments_removed_from_earlier_splits"] > 0


def test_requires_three_presentations():
    with pytest.raises(ValueError, match="three distinct"):
        split_by_presentation(demo_data().query("code_presentation in ['2013B', '2014J']"))


def test_model_comparison_returns_metrics_and_champion():
    results, champion = run_comparison(demo_data(n=30))
    assert set(results["validation_metrics"]) == {
        "logistic_regression", "random_forest", "hist_gradient_boosting"
    }
    assert results["selected_model"] in results["validation_metrics"]
    assert results["test_metrics_for_selected_model_only"]["n_examples"] == 30
    assert 0 <= results["test_metrics_for_selected_model_only"]["average_precision"] <= 1
    assert callable(champion.predict_proba)


def test_evaluation_uses_top_ten_percent():
    class FixedModel:
        def predict_proba(self, x):
            p = np.linspace(0.01, 0.99, len(x))
            return np.column_stack([1 - p, p])

    df = demo_data(n=30).query("code_presentation == '2014J'")
    metrics = evaluate_risk(FixedModel(), df)
    assert metrics["n_examples"] == 30
    assert 0 <= metrics["precision_at_top_10pct"] <= 1
