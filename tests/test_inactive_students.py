import pandas as pd

from engagepulse.features import build_day28_features


def test_students_without_activity_are_retained(tmp_path):
    keys = ["code_module", "code_presentation", "id_student"]

    pd.DataFrame([
        ["AAA", "2013J", 1, "Pass"],
        ["AAA", "2013J", 2, "Fail"],
    ], columns=keys + ["final_result"]).to_csv(
        tmp_path / "studentInfo.csv", index=False
    )

    pd.DataFrame([
        ["AAA", "2013J", 1, -5, None],
        ["AAA", "2013J", 2, -5, None],
    ], columns=keys + [
        "date_registration", "date_unregistration"
    ]).to_csv(
        tmp_path / "studentRegistration.csv", index=False
    )

    pd.DataFrame([
        ["AAA", "2013J", 1, 101, 7, 10],
    ], columns=keys + [
        "id_site", "date", "sum_click"
    ]).to_csv(
        tmp_path / "studentVle.csv", index=False
    )

    result = build_day28_features(tmp_path)
    students = result.set_index("id_student")

    assert set(students.index) == {1, 2}
    assert students.loc[2, "total_clicks"] == 0
