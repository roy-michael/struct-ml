import argparse
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Presentation start dates in OULAD
PRESENTATION_STARTS = {
    "2013B": "2013-02-01",
    "2013J": "2013-10-01",
    "2014B": "2014-02-01",
    "2014J": "2014-10-01",
}


def prepare_data(raw_dir: str | Path, out_dir: str | Path) -> None:
    raw_path = Path(raw_dir).resolve()
    out_path = Path(out_dir).resolve()

    db_dir = out_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw OULAD CSV files from: {raw_path}")
    print(f"Output RelBench dataset directory: {out_path}")

    # 1. Courses
    print("\n[1/7] Processing 'courses'...")
    courses_df = pd.read_csv(raw_path / "courses.csv")
    courses_df["raw_course_id"] = courses_df["code_module"] + "_" + courses_df["code_presentation"]
    courses_df.sort_values("raw_course_id", inplace=True)
    courses_df.reset_index(drop=True, inplace=True)
    courses_df["course_id"] = np.arange(len(courses_df), dtype=np.int64)
    courses_df["course_start_date"] = pd.to_datetime(courses_df["code_presentation"].map(PRESENTATION_STARTS))
    courses_df["course_end_date"] = courses_df["course_start_date"] + pd.to_timedelta(
        courses_df["module_presentation_length"], unit="D"
    )

    course_to_pkey = dict(zip(courses_df["raw_course_id"], courses_df["course_id"]))

    # 2. Assessments
    print("[2/7] Processing 'assessments'...")
    assessments_df = pd.read_csv(raw_path / "assessments.csv")
    assessments_df["raw_course_id"] = assessments_df["code_module"] + "_" + assessments_df["code_presentation"]
    assessments_df["course_id"] = assessments_df["raw_course_id"].map(course_to_pkey).astype(np.int64)
    assessments_df.sort_values("id_assessment", inplace=True)
    assessments_df.reset_index(drop=True, inplace=True)
    assessments_df["raw_assessment_id"] = assessments_df["id_assessment"]
    assessments_df["id_assessment"] = np.arange(len(assessments_df), dtype=np.int64)

    assess_start = pd.to_datetime(assessments_df["code_presentation"].map(PRESENTATION_STARTS))
    assessments_df["date_days"] = assessments_df["date"]
    assessments_df["assessment_date"] = assess_start + pd.to_timedelta(
        assessments_df["date"].fillna(0).astype(int), unit="D"
    )

    assess_raw_to_pkey = dict(zip(assessments_df["raw_assessment_id"], assessments_df["id_assessment"]))
    assess_to_pres = dict(zip(assessments_df["raw_assessment_id"], assessments_df["code_presentation"]))
    assess_to_raw_course = dict(zip(assessments_df["raw_assessment_id"], assessments_df["raw_course_id"]))

    # 3. VLE
    print("[3/7] Processing 'vle'...")
    vle_df = pd.read_csv(raw_path / "vle.csv")
    vle_df["raw_course_id"] = vle_df["code_module"] + "_" + vle_df["code_presentation"]
    vle_df["course_id"] = vle_df["raw_course_id"].map(course_to_pkey).astype(np.int64)
    vle_df.sort_values("id_site", inplace=True)
    vle_df.reset_index(drop=True, inplace=True)
    vle_df["raw_site_id"] = vle_df["id_site"]
    vle_df["id_site"] = np.arange(len(vle_df), dtype=np.int64)

    site_raw_to_pkey = dict(zip(vle_df["raw_site_id"], vle_df["id_site"]))

    # 4. StudentInfo
    print("[4/7] Processing 'studentInfo'...")
    student_info_df = pd.read_csv(raw_path / "studentInfo.csv")
    student_info_df["raw_course_id"] = student_info_df["code_module"] + "_" + student_info_df["code_presentation"]
    student_info_df["raw_student_reg_id"] = (
        student_info_df["raw_course_id"] + "_" + student_info_df["id_student"].astype(str)
    )
    student_info_df.sort_values("raw_student_reg_id", inplace=True)
    student_info_df.reset_index(drop=True, inplace=True)
    student_info_df["student_reg_id"] = np.arange(len(student_info_df), dtype=np.int64)
    student_info_df["course_id"] = student_info_df["raw_course_id"].map(course_to_pkey).astype(np.int64)

    student_reg_to_pkey = dict(zip(student_info_df["raw_student_reg_id"], student_info_df["student_reg_id"]))

    # 5. StudentRegistration
    print("[5/7] Processing 'studentRegistration'...")
    student_reg_df = pd.read_csv(raw_path / "studentRegistration.csv")
    student_reg_df["raw_course_id"] = student_reg_df["code_module"] + "_" + student_reg_df["code_presentation"]
    student_reg_df["raw_student_reg_id"] = (
        student_reg_df["raw_course_id"] + "_" + student_reg_df["id_student"].astype(str)
    )
    student_reg_df["student_reg_id"] = student_reg_df["raw_student_reg_id"].map(student_reg_to_pkey).astype(np.int64)
    student_reg_df["course_id"] = student_reg_df["raw_course_id"].map(course_to_pkey).astype(np.int64)
    student_reg_df.sort_values("student_reg_id", inplace=True)
    student_reg_df.reset_index(drop=True, inplace=True)

    reg_start = pd.to_datetime(student_reg_df["code_presentation"].map(PRESENTATION_STARTS))
    student_reg_df["registration_date"] = reg_start + pd.to_timedelta(
        student_reg_df["date_registration"].fillna(0).astype(int), unit="D"
    )
    unreg_mask = student_reg_df["date_unregistration"].notna()
    student_reg_df["unregistration_date"] = pd.NaT
    student_reg_df.loc[unreg_mask, "unregistration_date"] = reg_start[unreg_mask] + pd.to_timedelta(
        student_reg_df.loc[unreg_mask, "date_unregistration"].astype(int), unit="D"
    )

    # 6. StudentAssessment
    print("[6/7] Processing 'studentAssessment'...")
    student_assess_df = pd.read_csv(raw_path / "studentAssessment.csv")
    raw_course = student_assess_df["id_assessment"].map(assess_to_raw_course)
    raw_sreg = raw_course + "_" + student_assess_df["id_student"].astype(str)
    student_assess_df["student_reg_id"] = raw_sreg.map(student_reg_to_pkey).astype(np.int64)
    student_assess_df["id_assessment"] = student_assess_df["id_assessment"].map(assess_raw_to_pkey).astype(np.int64)

    sa_pres = student_assess_df["id_assessment"].map(dict(zip(assessments_df["id_assessment"], assessments_df["code_presentation"])))
    sa_start = pd.to_datetime(sa_pres.map(PRESENTATION_STARTS))
    student_assess_df["date"] = sa_start + pd.to_timedelta(
        student_assess_df["date_submitted"].fillna(0).astype(int), unit="D"
    )
    student_assess_df.sort_values("date", inplace=True)
    student_assess_df.reset_index(drop=True, inplace=True)

    # 7. StudentVle
    print("[7/7] Processing 'studentVle' (10M rows)...")
    student_vle_df = pd.read_csv(raw_path / "studentVle.csv")
    raw_vle_course = student_vle_df["code_module"] + "_" + student_vle_df["code_presentation"]
    raw_vle_sreg = raw_vle_course + "_" + student_vle_df["id_student"].astype(str)
    student_vle_df["student_reg_id"] = raw_vle_sreg.map(student_reg_to_pkey).astype(np.int64)
    student_vle_df["id_site"] = student_vle_df["id_site"].map(site_raw_to_pkey).astype(np.int64)

    vle_start = pd.to_datetime(student_vle_df["code_presentation"].map(PRESENTATION_STARTS))
    student_vle_df["date_offset"] = student_vle_df["date"]
    student_vle_df["date"] = vle_start + pd.to_timedelta(
        student_vle_df["date_offset"].fillna(0).astype(int), unit="D"
    )
    student_vle_df.drop(columns=["code_module", "code_presentation", "id_student"], inplace=True)
    student_vle_df.sort_values("date", inplace=True)
    student_vle_df.reset_index(drop=True, inplace=True)

    # Write tables to parquet in db/ and out_dir/
    table_dfs = {
        "courses": courses_df,
        "assessments": assessments_df,
        "vle": vle_df,
        "studentInfo": student_info_df,
        "studentRegistration": student_reg_df,
        "studentAssessment": student_assess_df,
        "studentVle": student_vle_df,
    }

    print("\nSaving database tables to parquet...")
    for tname, df in table_dfs.items():
        pq_path_db = db_dir / f"{tname}.parquet"
        pq_path_root = out_path / f"{tname}.parquet"
        df.to_parquet(pq_path_db, index=False)
        df.to_parquet(pq_path_root, index=False)
        print(f"  - {tname:20s}: {len(df):>10,} rows -> {pq_path_db.name}")

    # Dataset Manifest
    val_timestamp = "2014-02-01"
    test_timestamp = "2014-10-01"

    dataset_manifest = {
        "name": "oulad",
        "manifest_version": 1,
        "val_timestamp": val_timestamp,
        "test_timestamp": test_timestamp,
        "description": "Open University Learning Analytics Dataset (OULAD) formatted for RelBench.",
        "tables": {
            "courses": {
                "pkey": "course_id",
                "time_col": None,
                "fkeys": {},
            },
            "assessments": {
                "pkey": "id_assessment",
                "time_col": None,
                "fkeys": {"course_id": "courses"},
            },
            "vle": {
                "pkey": "id_site",
                "time_col": None,
                "fkeys": {"course_id": "courses"},
            },
            "studentInfo": {
                "pkey": "student_reg_id",
                "time_col": None,
                "fkeys": {"course_id": "courses"},
            },
            "studentRegistration": {
                "pkey": "student_reg_id",
                "time_col": None,
                "fkeys": {
                    "student_reg_id": "studentInfo",
                    "course_id": "courses",
                },
            },
            "studentAssessment": {
                "pkey": None,
                "time_col": "date",
                "fkeys": {
                    "id_assessment": "assessments",
                    "student_reg_id": "studentInfo",
                },
            },
            "studentVle": {
                "pkey": None,
                "time_col": "date",
                "fkeys": {
                    "id_site": "vle",
                    "student_reg_id": "studentInfo",
                },
            },
        },
    }

    with open(out_path / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_manifest, f, sort_keys=False)
    print(f"Created dataset manifest at: {out_path / 'manifest.yaml'}")

    # Build Tasks: student-fail and student-failure
    print("\nConstructing prediction tasks...")
    si = student_info_df[["student_reg_id", "code_presentation", "final_result"]].copy()
    si["timestamp"] = pd.to_datetime(si["code_presentation"].map(PRESENTATION_STARTS))
    si["failed"] = si["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    # Train / Val / Test splits
    train_mask = si["code_presentation"].isin(["2013B", "2013J"])
    val_mask = si["code_presentation"] == "2014B"
    test_mask = si["code_presentation"] == "2014J"

    train_df = si.loc[train_mask, ["timestamp", "student_reg_id", "failed"]].reset_index(drop=True)
    val_df = si.loc[val_mask, ["timestamp", "student_reg_id", "failed"]].reset_index(drop=True)
    test_df = si.loc[test_mask, ["timestamp", "student_reg_id", "failed"]].reset_index(drop=True)

    print(f"Task splits:")
    print(f"  Train: {len(train_df):>6,} rows | Fail rate: {train_df['failed'].mean():.1%}")
    print(f"  Val  : {len(val_df):>6,} rows | Fail rate: {val_df['failed'].mean():.1%}")
    print(f"  Test : {len(test_df):>6,} rows | Fail rate: {test_df['failed'].mean():.1%}")

    task_sql = """
SELECT
    t.timestamp,
    si.student_reg_id,
    CASE WHEN si.final_result IN ('Fail', 'Withdrawn') THEN 1 ELSE 0 END AS failed
FROM timestamps t
JOIN studentInfo si
    ON t.timestamp = CASE
        WHEN si.code_presentation = '2013B' THEN TIMESTAMP '2013-02-01'
        WHEN si.code_presentation = '2013J' THEN TIMESTAMP '2013-10-01'
        WHEN si.code_presentation = '2014B' THEN TIMESTAMP '2014-02-01'
        WHEN si.code_presentation = '2014J' THEN TIMESTAMP '2014-10-01'
    END
""".strip()

    task_manifest = {
        "name": "student-fail",
        "kind": "forecast",
        "task_type": "binary_classification",
        "description": "Predict whether a student will fail or withdraw from a course presentation.",
        "entity_table": "studentInfo",
        "entity_col": "student_reg_id",
        "target_col": "failed",
        "time_col": "timestamp",
        "timedelta": "30 days",
        "num_eval_timestamps": 1,
        "remove_columns": [
            ["studentInfo", "final_result"],
            ["studentRegistration", "date_unregistration"],
            ["studentRegistration", "unregistration_date"],
        ],
        "sql": task_sql,
        "manifest_version": 1,
    }

    task_dirs = [
        out_path / "tasks" / "student-fail",
        out_path / "tasks" / "student-failure",
        out_path / "student-fail",
        out_path / "student-failure",
    ]

    for tdir in task_dirs:
        tdir.mkdir(parents=True, exist_ok=True)
        manifest_copy = dict(task_manifest)
        manifest_copy["name"] = tdir.name
        with open(tdir / "manifest.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest_copy, f, sort_keys=False)
        train_df.to_parquet(tdir / "train.parquet", index=False)
        val_df.to_parquet(tdir / "val.parquet", index=False)
        test_df.to_parquet(tdir / "test.parquet", index=False)

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare OULAD dataset for RelBench.")
    parser.add_argument("--raw_dir", type=str, default="oulad", help="Path to raw CSV files.")
    parser.add_argument("--out_dir", type=str, default="output", help="Path to output directory.")
    args = parser.parse_args()

    prepare_data(raw_dir=args.raw_dir, out_dir=args.out_dir)
