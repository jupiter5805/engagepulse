import numpy as np
import pandas as pd

from engagepulse.features import FEATURES
from engagepulse.model import split_by_student, fit_baseline, evaluate


def test_grouped_split_has_no_student_overlap():
    frame = pd.DataFrame({"id_student": [i // 2 for i in range(60)],
                          "at_risk": [i % 2 for i in range(60)]})
    train, test = split_by_student(frame)
    assert set(train.id_student).isdisjoint(set(test.id_student))


def test_baseline_trains_on_numeric_features():
    size = 100
    frame = pd.DataFrame({"id_student": np.arange(size),
                          "at_risk": [i % 2 for i in range(size)]})
    for feature in FEATURES:
        frame[feature] = np.arange(size) / 10 + np.array([i % 2 for i in range(size)])
    train, test = split_by_student(frame)
    model = fit_baseline(train)
    metrics = evaluate(model, test)
    assert metrics["n_test"] == len(test)
    assert 0 <= metrics["average_precision"] <= 1
    assert 0 <= metrics["precision_at_top_10pct"] <= 1
