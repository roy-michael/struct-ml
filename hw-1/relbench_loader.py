import os
from pathlib import Path
import pandas as pd
import yaml
from relbench.base import Table, Database


PRESENTATION_STARTS = {
    "2013B": "2013-02-01",
    "2013J": "2013-10-01",
    "2014B": "2014-02-01",
    "2014J": "2014-10-01",
}


def _offset_to_datetime(presentations: pd.Series, offsets: pd.Series) -> pd.Series:
    base_dates = pd.to_datetime(presentations.map(PRESENTATION_STARTS))
    valid = offsets.notna()
    result = pd.Series(pd.NaT, index=offsets.index, dtype="datetime64[ns]")
    result.loc[valid] = base_dates.loc[valid] + pd.to_timedelta(
        offsets.loc[valid].astype(int), unit="D"
    )
    return result


def load_oulad_relbench(raw_dir: str | Path | None = None) -> Database:
    """Load the OULAD dataset from raw CSV files into a normalized RelBench Database."""
    import numpy as np

    if raw_dir is None:
        raw_path = Path(__file__).resolve().parent.parent / "datasets" / "OULAD"
    else:
        raw_path = Path(raw_dir)
    print(f"Loading CSV files from {raw_path.resolve()}...")

    # 1. Courses (PK: course_id)
    raw_courses = pd.read_csv(raw_path / "courses.csv")
    raw_courses["_raw_course_id"] = (
        raw_courses["code_module"] + "_" + raw_courses["code_presentation"]
    )
    raw_courses = raw_courses.sort_values("_raw_course_id").reset_index(drop=True)
    raw_courses["course_id"] = np.arange(len(raw_courses), dtype=np.int64)
    course_to_pkey = dict(zip(raw_courses["_raw_course_id"], raw_courses["course_id"]))
    courses_df = raw_courses[
        ["course_id", "code_module", "code_presentation", "module_presentation_length"]
    ].copy()

    # 2. Assessments (PK: assessment_id, FK: course_id -> courses)
    raw_assessments = pd.read_csv(raw_path / "assessments.csv")
    raw_assessments["_raw_course_id"] = (
        raw_assessments["code_module"] + "_" + raw_assessments["code_presentation"]
    )
    raw_assessments = raw_assessments.sort_values("id_assessment").reset_index(drop=True)
    raw_assessments["_raw_assessment_id"] = raw_assessments["id_assessment"]
    raw_assessments["assessment_id"] = np.arange(len(raw_assessments), dtype=np.int64)
    raw_assessments["course_id"] = (
        raw_assessments["_raw_course_id"].map(course_to_pkey).astype(np.int64)
    )
    raw_assessments["date"] = _offset_to_datetime(
        raw_assessments["code_presentation"], raw_assessments["date"]
    )
    assess_raw_to_pkey = dict(
        zip(raw_assessments["_raw_assessment_id"], raw_assessments["assessment_id"])
    )
    assess_to_pres = dict(
        zip(raw_assessments["_raw_assessment_id"], raw_assessments["code_presentation"])
    )
    assess_to_raw_course = dict(
        zip(raw_assessments["_raw_assessment_id"], raw_assessments["_raw_course_id"])
    )
    assessments_df = raw_assessments[
        ["assessment_id", "course_id", "assessment_type", "date", "weight"]
    ].copy()

    # 3. VLE (PK: site_id, FK: course_id -> courses)
    raw_vle = pd.read_csv(raw_path / "vle.csv")
    raw_vle["_raw_course_id"] = (
        raw_vle["code_module"] + "_" + raw_vle["code_presentation"]
    )
    raw_vle = raw_vle.sort_values("id_site").reset_index(drop=True)
    raw_vle["_raw_site_id"] = raw_vle["id_site"]
    raw_vle["site_id"] = np.arange(len(raw_vle), dtype=np.int64)
    raw_vle["course_id"] = raw_vle["_raw_course_id"].map(course_to_pkey).astype(np.int64)
    site_raw_to_pkey = dict(zip(raw_vle["_raw_site_id"], raw_vle["site_id"]))
    vle_df = raw_vle[
        ["site_id", "course_id", "activity_type", "week_from", "week_to"]
    ].copy()

    # 4. StudentInfo (28,785 unique students, PK: student_id)
    raw_student_info = pd.read_csv(raw_path / "studentInfo.csv")
    student_info_df = (
        raw_student_info[
            [
                "id_student",
                "gender",
                "region",
                "highest_education",
                "imd_band",
                "disability",
            ]
        ]
        .drop_duplicates()
        .sort_values("id_student")
        .reset_index(drop=True)
    )
    student_info_df["_raw_student_id"] = student_info_df["id_student"]
    student_info_df["student_id"] = np.arange(len(student_info_df), dtype=np.int64)
    student_to_pkey = dict(
        zip(student_info_df["_raw_student_id"], student_info_df["student_id"])
    )
    student_info_df = student_info_df[
        ["student_id", "gender", "region", "highest_education", "imd_band", "disability"]
    ].copy()

    # 5. StudentRegistration (32,593 enrollments, PK: student_reg_id)
    raw_student_reg = pd.read_csv(raw_path / "studentRegistration.csv")
    join_cols = ["code_module", "code_presentation", "id_student"]
    raw_student_reg = raw_student_reg.merge(
        raw_student_info[
            join_cols
            + ["age_band", "num_of_prev_attempts", "studied_credits", "final_result"]
        ],
        on=join_cols,
        how="inner",
    )
    raw_student_reg["_raw_course_id"] = (
        raw_student_reg["code_module"] + "_" + raw_student_reg["code_presentation"]
    )
    raw_student_reg["_raw_student_reg_id"] = (
        raw_student_reg["_raw_course_id"]
        + "_"
        + raw_student_reg["id_student"].astype(str)
    )
    raw_student_reg = raw_student_reg.sort_values("_raw_student_reg_id").reset_index(
        drop=True
    )
    raw_student_reg["student_reg_id"] = np.arange(len(raw_student_reg), dtype=np.int64)
    raw_student_reg["student_id"] = (
        raw_student_reg["id_student"].map(student_to_pkey).astype(np.int64)
    )
    raw_student_reg["course_id"] = (
        raw_student_reg["_raw_course_id"].map(course_to_pkey).astype(np.int64)
    )
    raw_student_reg["date_registration"] = _offset_to_datetime(
        raw_student_reg["code_presentation"], raw_student_reg["date_registration"]
    )
    raw_student_reg["date_unregistration"] = _offset_to_datetime(
        raw_student_reg["code_presentation"], raw_student_reg["date_unregistration"]
    )
    student_reg_to_pkey = dict(
        zip(raw_student_reg["_raw_student_reg_id"], raw_student_reg["student_reg_id"])
    )
    student_reg_df = raw_student_reg[
        [
            "student_reg_id",
            "student_id",
            "course_id",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "date_registration",
            "date_unregistration",
            "final_result",
        ]
    ].copy()

    # 6. StudentAssessment
    raw_student_assess = pd.read_csv(raw_path / "studentAssessment.csv")
    raw_sa_course = raw_student_assess["id_assessment"].map(assess_to_raw_course)
    raw_sa_sreg = raw_sa_course + "_" + raw_student_assess["id_student"].astype(str)
    sa_pres = raw_student_assess["id_assessment"].map(assess_to_pres)
    raw_student_assess["student_reg_id"] = (
        raw_sa_sreg.map(student_reg_to_pkey).astype(np.int64)
    )
    raw_student_assess["assessment_id"] = (
        raw_student_assess["id_assessment"].map(assess_raw_to_pkey).astype(np.int64)
    )
    raw_student_assess["date_submitted"] = _offset_to_datetime(
        sa_pres, raw_student_assess["date_submitted"]
    )
    raw_student_assess = raw_student_assess.sort_values("date_submitted").reset_index(
        drop=True
    )
    raw_student_assess["student_assessment_id"] = np.arange(
        len(raw_student_assess), dtype=np.int64
    )
    student_assess_df = raw_student_assess[
        [
            "student_assessment_id",
            "assessment_id",
            "student_reg_id",
            "date_submitted",
            "is_banked",
            "score",
        ]
    ].copy()

    # 7. StudentVle (optional)
    student_vle_path = raw_path / "studentVle.csv"
    student_vle_df = None
    if student_vle_path.exists():
        print("Loading studentVle.csv (this may take a few seconds)...")
        raw_student_vle = pd.read_csv(student_vle_path)
        raw_vle_sreg = (
            raw_student_vle["code_module"]
            + "_"
            + raw_student_vle["code_presentation"]
            + "_"
            + raw_student_vle["id_student"].astype(str)
        )
        raw_student_vle["student_reg_id"] = (
            raw_vle_sreg.map(student_reg_to_pkey).astype(np.int64)
        )
        raw_student_vle["site_id"] = (
            raw_student_vle["id_site"].map(site_raw_to_pkey).astype(np.int64)
        )
        raw_student_vle["date"] = _offset_to_datetime(
            raw_student_vle["code_presentation"], raw_student_vle["date"]
        )
        raw_student_vle = raw_student_vle.sort_values("date").reset_index(drop=True)
        student_vle_df = raw_student_vle[
            ["site_id", "student_reg_id", "date", "sum_click"]
        ].copy()

    table_dict = {
        "courses": Table(
            df=courses_df,
            fkey_col_to_pkey_table={},
            pkey_col="course_id",
            time_col=None,
        ),
        "assessments": Table(
            df=assessments_df,
            fkey_col_to_pkey_table={"course_id": "courses"},
            pkey_col="assessment_id",
            time_col=None,
        ),
        "vle": Table(
            df=vle_df,
            fkey_col_to_pkey_table={"course_id": "courses"},
            pkey_col="site_id",
            time_col=None,
        ),
        "studentInfo": Table(
            df=student_info_df,
            fkey_col_to_pkey_table={},
            pkey_col="student_id",
            time_col=None,
        ),
        "studentRegistration": Table(
            df=student_reg_df,
            fkey_col_to_pkey_table={
                "student_id": "studentInfo",
                "course_id": "courses",
            },
            pkey_col="student_reg_id",
            time_col=None,
        ),
        "studentAssessment": Table(
            df=student_assess_df,
            fkey_col_to_pkey_table={
                "assessment_id": "assessments",
                "student_reg_id": "studentRegistration",
            },
            pkey_col="student_assessment_id",
            time_col="date_submitted",
        ),
    }

    if student_vle_df is not None:
        table_dict["studentVle"] = Table(
            df=student_vle_df,
            fkey_col_to_pkey_table={
                "site_id": "vle",
                "student_reg_id": "studentRegistration",
            },
            pkey_col=None,
            time_col="date",
        )

    return Database(table_dict=table_dict)


def export_relbench_dataset(
    db: Database,
    out_dir: str | Path,
    dataset_name: str = "oulad",
    val_timestamp: str = "2014-02-01",
    test_timestamp: str = "2014-10-01",
) -> None:
    """Exports a RelBench Database to disk with manifest.yaml and parquet files (BYOD format)."""
    out_path = Path(out_dir)
    db_dir = out_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    manifest_tables = {}
    for table_name, table in db.table_dict.items():
        print(f"Saving table '{table_name}' ({len(table):,} rows) to parquet...")
        parquet_file = db_dir / f"{table_name}.parquet"
        table.df.to_parquet(parquet_file, index=False)

        manifest_tables[table_name] = {
            "pkey": table.pkey_col,
            "time_col": table.time_col,
            "fkeys": table.fkey_col_to_pkey_table,
        }

    manifest = {
        "name": dataset_name,
        "manifest_version": 1,
        "val_timestamp": val_timestamp,
        "test_timestamp": test_timestamp,
        "tables": manifest_tables,
    }

    manifest_file = out_path / "manifest.yaml"
    with open(manifest_file, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"RelBench dataset exported to: {out_path.resolve()}")
    print(f"Manifest created at: {manifest_file.resolve()}")


if __name__ == "__main__":
    database = load_oulad_relbench()
    print("\nDatabase loaded successfully:")
    print(database)
    for name, tbl in database.table_dict.items():
        print(f"  {name:20s} rows={len(tbl):>8,}  pkey={tbl.pkey_col}  fkeys={tbl.fkey_col_to_pkey_table}")
