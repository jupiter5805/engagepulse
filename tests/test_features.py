import pandas as pd

from engagepulse.features import build_day28_features


def test_day_28_features_exclude_future_and_already_withdrawn(tmp_path):
    keys = ["code_module", "code_presentation", "id_student"]
    pd.DataFrame([
        ["AAA", "2013J", 1, "Pass"],
        ["AAA", "2013J", 2, "Withdrawn"],
        ["AAA", "2013J", 3, "Fail"],
    ], columns=keys + ["final_result"]).to_csv(tmp_path / "studentInfo.csv", index=False)
    pd.DataFrame([
        ["AAA", "2013J", 1, -3, None],
        ["AAA", "2013J", 2, -3, 12],
        ["AAA", "2013J", 3, -3, None],
    ], columns=keys + ["date_registration", "date_unregistration"]).to_csv(
        tmp_path / "studentRegistration.csv", index=False,
    )
    pd.DataFrame([
        ["AAA", "2013J", 1, 101, 0, 2],
        ["AAA", "2013J", 1, 101, 7, 3],
        ["AAA", "2013J", 1, 102, 27, 5],
        ["AAA", "2013J", 1, 103, 28, 900],  # future info: must not leak
        ["AAA", "2013J", 2, 101, 0, 7],  # withdrew before snapshot
        ["AAA", "2013J", 3, 101, 40, 9], # future info: must not leak
    ], columns=keys + ["id_site", "date", "sum_click"]).to_csv(
        tmp_path / "studentVle.csv", index=False,
    )
    result = build_day28_features(tmp_path, chunksize=2)
    assert set(result["id_student"]) == {1, 3}
    first = result.set_index("id_student").loc[1]
    assert first["total_clicks"] == 10
    assert first["active_days"] == 3
    assert first["distinct_resources"] == 2
    assert first["week_4_clicks"] == 5
    assert first["days_since_last_activity"] == 0
    assert result.set_index("id_student").loc[3, "total_clicks"] == 0
    assert result.set_index("id_student").loc[3, "days_since_last_activity"] == 28
    assert result.set_index("id_student").loc[3, "at_risk"] == 1
