"""Grouped, reproducible baseline with decision-useful metrics."""
import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from engagepulse.features import FEATURES


def split_by_student(frame: pd.DataFrame, test_size: float = 0.2):
    """Prevent one student's multiple enrolments appearing in train and test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(splitter.split(frame, frame["at_risk"], frame["id_student"]))
    return frame.iloc[train_idx], frame.iloc[test_idx]


def fit_baseline(train: pd.DataFrame):
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )
    model.fit(train[FEATURES], train["at_risk"])
    return model


def evaluate(model, test: pd.DataFrame):
    labels = test["at_risk"].to_numpy()
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    top_n = max(1, math.ceil(len(test) * 0.10))
    top_indices = np.argsort(probabilities)[-top_n:]
    positives = int(labels.sum())
    return {
        "n_test": len(test),
        "at_risk_prevalence": round(float(labels.mean()), 4),
        "average_precision": round(float(average_precision_score(labels, probabilities)), 4),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4) if len(set(labels)) == 2 else None,
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 4),
        "precision_at_top_10pct": round(float(labels[top_indices].mean()), 4),
        "recall_at_top_10pct": round(float(labels[top_indices].sum() / positives), 4) if positives else None,
    }
