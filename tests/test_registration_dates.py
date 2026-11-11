import pandas as pd

from engagepulse.features import build_day28_features


def test_registration_dates_handle_missing_values(tmp_path):
    keys = ["code_module", "code_presentation", "id_student"]

    pd.DataFrame([
        ["AAA", "2013J", 1, "Pass"],
        ["AAA", "2013J", 2, "Fail"],
        ["AAA", "2013J", 3, "Fail"],
        ["AAA", "2013J", 4, "Withdrawn"],
        ["AAA", "2013J", 5, "Pass"],
    ], columns=keys + ["final_result"]).to_csv(
        tmp_path / "studentInfo.csv", index=False
    )

    pd.DataFrame([
        ["AAA", "2013J", 1, "-3", "?"],
        ["AAA", "2013J", 2, "?", "?"],
        ["AAA", "2013J", 3, "35", "?"],
        ["AAA", "2013J", 4, "-5", "12"],
        ["AAA", "2013J", 5, "-10", "28"],
    ], columns=keys + [
        "date_registration",
        "date_unregistration",
    ]).to_csv(
        tmp_path / "studentRegistration.csv", index=False
    )

    pd.DataFrame([
        ["AAA", "2013J", 1, 101, 10, 4],
        ["AAA", "2013J", 2, 102, 7, 3],
        ["AAA", "2013J", 5, 103, 27, 2],
    ], columns=keys + [
        "id_site", "date", "sum_click"
    ]).to_csv(
        tmp_path / "studentVle.csv", index=False
    )

    result = build_day28_features(
        tmp_path, chunksize=2
    )

    assert set(result["id_student"]) == {1, 2, 5}
    assert result.set_index("id_student").loc[1, "total_clicks"] == 4
    assert result.set_index("id_student").loc[2, "at_risk"] == 1
