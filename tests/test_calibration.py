import numpy as np
import pandas as pd
import pytest

from engagepulse.calibration import (
    _logits, compare_calibration, fit_calibrator,
    reliability_report, split_grouped_three_way,
)
from engagepulse.features import FEATURES
from engagepulse.model import fit_baseline


def sample_data(n=240):
    rng = np.random.default_rng(42)
    outcomes = rng.integers(0, 2, size=n)
    data = {feature: rng.normal(size=n) for feature in FEATURES}
    data["total_clicks"] = outcomes * 2 + rng.normal(0, 1, size=n)
    return pd.DataFrame({**data, "id_student": np.arange(n), "at_risk": outcomes})


def test_splits_keep_student_ids_disjoint():
    frame = sample_data()
    repeated = frame.iloc[:20].copy()
    frame = pd.concat([frame, repeated], ignore_index=True)
    train, cal, test = split_grouped_three_way(frame)
    assert set(train.id_student).isdisjoint(cal.id_student)
    assert set(train.id_student).isdisjoint(test.id_student)
    assert set(cal.id_student).isdisjoint(test.id_student)
    assert len(train) + len(cal) + len(test) == len(frame)


def test_calibrated_model_produces_valid_probabilities():
    train, cal, test = split_grouped_three_way(sample_data())
    model = fit_baseline(train)
    adjusted = fit_calibrator(model, cal)
    probabilities = adjusted.predict_proba(test[FEATURES])
    assert probabilities.shape == (len(test), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_reliability_report_counts_observations_and_bins():
    result = reliability_report([0, 1, 0, 1], [0.1, 0.9, 0.3, 1.0])
    assert result["n"] == 4
    assert sum(bin_["n"] for bin_ in result["reliability_bins"]) == 4
    assert result["brier_score"] >= 0
    assert 0 <= result["expected_calibration_error_10_bins"] <= 1


def test_comparison_reports_both_versions():
    train, cal, test = split_grouped_three_way(sample_data())
    model = fit_baseline(train)
    adjusted = fit_calibrator(model, cal)
    report = compare_calibration(model, adjusted, test)
    assert set(report) == {"uncalibrated", "sigmoid_calibrated"}
    assert report["uncalibrated"]["n"] == len(test)
    assert report["sigmoid_calibrated"]["n"] == len(test)


def test_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="between 0 and 1"):
        _logits([0.2, 1.2])
    with pytest.raises(ValueError, match="finite"):
        _logits([0.1, float("nan")])


def test_calibration_requires_two_classes():
    train, cal, _ = split_grouped_three_way(sample_data())
    model = fit_baseline(train)
    only_positive = cal.loc[cal["at_risk"] == 1]
    with pytest.raises(ValueError, match="both outcome classes"):
        fit_calibrator(model, only_positive)
