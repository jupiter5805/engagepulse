"""Leakage-aware OULAD student engagement features at day 27.

Only events from course days 0..27 can become predictive features.
Students already unregistered on/before day 27 are excluded: this is
an *early warning* model, not a detector of past withdrawal.
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]
FEATURES = [
    "total_clicks", "active_days", "distinct_resources", "days_since_last_activity",
    "week_1_clicks", "week_2_clicks", "week_3_clicks", "week_4_clicks",
    "recent_click_trend",
]


def build_day28_features(data_dir: str | Path, chunksize: int = 250_000) -> pd.DataFrame:
    root = Path(data_dir)
    required = ["studentInfo.csv", "studentRegistration.csv", "studentVle.csv"]
    absent = [name for name in required if not (root / name).is_file()]
    if absent:
        raise FileNotFoundError(f"Missing OULAD files in {root}: {', '.join(absent)}")

    info = pd.read_csv(root / "studentInfo.csv", usecols=KEY + ["final_result"])
    registration = pd.read_csv(
        root / "studentRegistration.csv",
        usecols=KEY + ["date_registration", "date_unregistration"],
    )
    # Convert OULAD registration dates to numeric values.
    # Missing or invalid date markers become NaN.
    for column in ("date_registration", "date_unregistration"):
        registration[column] = pd.to_numeric(
            registration[column],
            errors="coerce",
        )

    if info.duplicated(KEY).any() or registration.duplicated(KEY).any():
        raise ValueError("Unexpected duplicate enrolment keys in OULAD source tables")

    base = info.merge(registration, on=KEY, how="inner", validate="one_to_one")
    base = base.loc[
        (base["date_registration"].isna() | (base["date_registration"] <= 27))
        & (base["date_unregistration"].isna() | (base["date_unregistration"] > 27))
    ].copy()
    base["at_risk"] = base["final_result"].isin(["Withdrawn", "Fail"]).astype(int)

    # Keyed accumulation avoids loading ~10 million VLE records in memory.
    def blank():
        return {"weeks": [0, 0, 0, 0], "days": set(), "resources": set(), "last": -1}

    stats = defaultdict(blank)
    valid_keys = set(base[KEY].itertuples(index=False, name=None))
    for chunk in pd.read_csv(
        root / "studentVle.csv", usecols=KEY + ["id_site", "date", "sum_click"],
        chunksize=chunksize,
    ):
        chunk = chunk.loc[chunk["date"].between(0, 27)]
        # Restrict early to avoid storing records for students who already withdrew.
        for row in chunk.itertuples(index=False):
            key = (row.code_module, row.code_presentation, row.id_student)
            if key not in valid_keys:
                continue
            day = int(row.date)
            state = stats[key]
            state["weeks"][day // 7] += int(row.sum_click)
            state["days"].add(day)
            state["resources"].add(int(row.id_site))
            state["last"] = max(state["last"], day)

    records = []
    for key in valid_keys:
        state = stats.get(key)
        weeks = state["weeks"] if state else [0, 0, 0, 0]
        last = state["last"] if state else -1
        records.append((*key, sum(weeks), len(state["days"]) if state else 0,
                        len(state["resources"]) if state else 0,
                        27 - last if last >= 0 else 28,
                        *weeks, weeks[3] - weeks[0]))

    values = pd.DataFrame.from_records(records, columns=KEY + FEATURES)
    result = base.merge(values, on=KEY, validate="one_to_one")
    return result.sort_values(KEY).reset_index(drop=True)
