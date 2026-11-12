import numpy as np
import pandas as pd
import pytest

from engagepulse.explain import global_feature_importance
from engagepulse.features import FEATURES
from engagepulse.model import fit_baseline


def example_frame():
    rng = np.random.default_rng(42)
    n = 160
    labels = rng.integers(0, 2, size=n)
    data = {feature: rng.normal(size=n) for feature in FEATURES}
    data["total_clicks"] = labels * 4 + rng.normal(0, 0.3, size=n)
    return pd.DataFrame({**data, "at_risk": labels})


def test_explanation_identifies_predictive_feature():
    frame = example_frame()
    model = fit_baseline(frame.iloc[:110])
    explanation = global_feature_importance(model, frame.iloc[110:], repeats=2)
    assert explanation["n_holdout"] == 50
    assert explanation["importance"][0]["feature"] == "total_clicks"
    assert explanation["importance"][0]["mean_drop_in_average_precision"] > 0
    assert len(explanation["importance"]) == len(FEATURES)


def test_explanation_rejects_single_class_holdout():
    frame = example_frame()
    model = fit_baseline(frame)
    with pytest.raises(ValueError, match="both outcome classes"):
        global_feature_importance(model, frame[frame.at_risk == 1])


def test_explanation_rejects_zero_repeats():
    frame = example_frame()
    model = fit_baseline(frame)
    with pytest.raises(ValueError, match="at least 1"):
        global_feature_importance(model, frame, repeats=0)
