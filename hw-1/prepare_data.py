import argparse
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


def offset_to_datetime(
    presentations: pd.Series, offsets: pd.Series
) -> pd.Series:
    """Convert OULAD relative day offsets to datetime64 timestamps (preserving NaN as NaT)."""
    base_dates = pd.to_datetime(presentations.map(PRESENTATION_STARTS))
    valid = offsets.notna()
    result = pd.Series(pd.NaT, index=offsets.index, dtype="datetime64[ns]")
    result.loc[valid] = base_dates.loc[valid] + pd.to_timedelta(
        offsets.loc[valid].astype(int), unit="D"
    )
    return result


def prepare_courses(raw_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build normalized courses table (PK: course_id) and raw-to-pkey mapping."""
    print("\n[1/7] Processing 'courses'...")
    raw_courses = pd.read_csv(raw_path / "courses.csv")
    raw_courses["_raw_course_id"] = (
        raw_courses["code_module"] + "_" + raw_courses["code_presentation"]
    )
    raw_courses = raw_courses.sort_values("_raw_course_id").reset_index(drop=True)
    raw_courses["course_id"] = np.arange(len(raw_courses), dtype=np.int64)

    course_to_pkey = dict(
        zip(raw_courses["_raw_course_id"], raw_courses["course_id"])
    )
    courses_df = raw_courses[
        ["course_id", "code_module", "code_presentation", "module_presentation_length"]
    ].copy()
    return courses_df, course_to_pkey


def prepare_assessments(
    raw_path: Path, course_to_pkey: dict[str, int]
) -> tuple[pd.DataFrame, dict[int, int], dict[int, str], dict[int, str]]:
    """Build normalized assessments table (PK: id_assessment, FK: course_id -> courses)."""
    print("[2/7] Processing 'assessments'...")
    raw_assessments = pd.read_csv(raw_path / "assessments.csv")
    raw_assessments["_raw_course_id"] = (
        raw_assessments["code_module"] + "_" + raw_assessments["code_presentation"]
    )
    raw_assessments = raw_assessments.sort_values("id_assessment").reset_index(drop=True)

    raw_assessments["_raw_assessment_id"] = raw_assessments["id_assessment"]
    raw_assessments["id_assessment"] = np.arange(len(raw_assessments), dtype=np.int64)
    raw_assessments["course_id"] = (
        raw_assessments["_raw_course_id"].map(course_to_pkey).astype(np.int64)
    )
    raw_assessments["date"] = offset_to_datetime(
        raw_assessments["code_presentation"], raw_assessments["date"]
    )

    assess_raw_to_pkey = dict(
        zip(raw_assessments["_raw_assessment_id"], raw_assessments["id_assessment"])
    )
    assess_to_pres = dict(
        zip(raw_assessments["_raw_assessment_id"], raw_assessments["code_presentation"])
    )
    assess_to_raw_course = dict(
        zip(raw_assessments["_raw_assessment_id"], raw_assessments["_raw_course_id"])
    )

    assessments_df = raw_assessments[
        ["id_assessment", "course_id", "assessment_type", "date", "weight"]
    ].copy()
    return assessments_df, assess_raw_to_pkey, assess_to_pres, assess_to_raw_course


def prepare_vle(
    raw_path: Path, course_to_pkey: dict[str, int]
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Build normalized vle table (PK: id_site, FK: course_id -> courses)."""
    print("[3/7] Processing 'vle'...")
    raw_vle = pd.read_csv(raw_path / "vle.csv")
    raw_vle["_raw_course_id"] = (
        raw_vle["code_module"] + "_" + raw_vle["code_presentation"]
    )
    raw_vle = raw_vle.sort_values("id_site").reset_index(drop=True)

    raw_vle["_raw_site_id"] = raw_vle["id_site"]
    raw_vle["id_site"] = np.arange(len(raw_vle), dtype=np.int64)
    raw_vle["course_id"] = raw_vle["_raw_course_id"].map(course_to_pkey).astype(np.int64)

    site_raw_to_pkey = dict(zip(raw_vle["_raw_site_id"], raw_vle["id_site"]))
    vle_df = raw_vle[
        ["id_site", "course_id", "activity_type", "week_from", "week_to"]
    ].copy()
    return vle_df, site_raw_to_pkey


def prepare_student_info(
    raw_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    """Build deduplicated studentInfo table (PK: id_student) and return raw df + mapping."""
    print("[4/7] Processing 'studentInfo'...")
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
    student_info_df["_raw_id_student"] = student_info_df["id_student"]
    student_info_df["id_student"] = np.arange(len(student_info_df), dtype=np.int64)

    student_to_pkey = dict(
        zip(student_info_df["_raw_id_student"], student_info_df["id_student"])
    )
    student_info_df = student_info_df[
        [
            "id_student",
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "disability",
        ]
    ].copy()
    return student_info_df, raw_student_info, student_to_pkey


def prepare_student_registration(
    raw_path: Path,
    raw_student_info: pd.DataFrame,
    course_to_pkey: dict[str, int],
    student_to_pkey: dict[int, int],
) -> tuple[pd.DataFrame, dict[str, int], pd.Series]:
    """Build studentRegistration table (PK: student_reg_id) with per-enrollment attributes."""
    print("[5/7] Processing 'studentRegistration'...")
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

    raw_student_reg["student_reg_id"] = np.arange(
        len(raw_student_reg), dtype=np.int64
    )
    raw_student_reg["id_student"] = (
        raw_student_reg["id_student"].map(student_to_pkey).astype(np.int64)
    )
    raw_student_reg["course_id"] = (
        raw_student_reg["_raw_course_id"].map(course_to_pkey).astype(np.int64)
    )
    raw_student_reg["date_registration"] = offset_to_datetime(
        raw_student_reg["code_presentation"], raw_student_reg["date_registration"]
    )
    raw_student_reg["date_unregistration"] = offset_to_datetime(
        raw_student_reg["code_presentation"], raw_student_reg["date_unregistration"]
    )

    student_reg_to_pkey = dict(
        zip(raw_student_reg["_raw_student_reg_id"], raw_student_reg["student_reg_id"])
    )
    reg_pres_series = raw_student_reg["code_presentation"].copy()

    student_reg_df = raw_student_reg[
        [
            "student_reg_id",
            "id_student",
            "course_id",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "date_registration",
            "date_unregistration",
            "final_result",
        ]
    ].copy()
    return student_reg_df, student_reg_to_pkey, reg_pres_series


def prepare_student_assessment(
    raw_path: Path,
    assess_raw_to_pkey: dict[int, int],
    assess_to_pres: dict[int, str],
    assess_to_raw_course: dict[int, str],
    student_reg_to_pkey: dict[str, int],
) -> pd.DataFrame:
    """Build chronologically sorted studentAssessment table (PK: student_assessment_id)."""
    print("[6/7] Processing 'studentAssessment'...")
    raw_student_assess = pd.read_csv(raw_path / "studentAssessment.csv")
    raw_sa_course = raw_student_assess["id_assessment"].map(assess_to_raw_course)
    raw_sa_sreg = raw_sa_course + "_" + raw_student_assess["id_student"].astype(str)
    sa_pres = raw_student_assess["id_assessment"].map(assess_to_pres)

    raw_student_assess["student_reg_id"] = (
        raw_sa_sreg.map(student_reg_to_pkey).astype(np.int64)
    )
    raw_student_assess["id_assessment"] = (
        raw_student_assess["id_assessment"].map(assess_raw_to_pkey).astype(np.int64)
    )
    raw_student_assess["date_submitted"] = offset_to_datetime(
        sa_pres, raw_student_assess["date_submitted"]
    )
    raw_student_assess = raw_student_assess.sort_values("date_submitted").reset_index(
        drop=True
    )
    raw_student_assess["student_assessment_id"] = np.arange(
        len(raw_student_assess), dtype=np.int64
    )

    return raw_student_assess[
        [
            "student_assessment_id",
            "id_assessment",
            "student_reg_id",
            "date_submitted",
            "is_banked",
            "score",
        ]
    ].copy()


def prepare_student_vle(
    raw_path: Path,
    site_raw_to_pkey: dict[int, int],
    student_reg_to_pkey: dict[str, int],
) -> pd.DataFrame:
    """Build chronologically sorted studentVle table (PK: None, TIME: date)."""
    print("[7/7] Processing 'studentVle' (10M+ rows)...")
    raw_student_vle = pd.read_csv(raw_path / "studentVle.csv")
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
    raw_student_vle["id_site"] = (
        raw_student_vle["id_site"].map(site_raw_to_pkey).astype(np.int64)
    )
    raw_student_vle["date"] = offset_to_datetime(
        raw_student_vle["code_presentation"], raw_student_vle["date"]
    )
    raw_student_vle = raw_student_vle.sort_values("date").reset_index(drop=True)

    return raw_student_vle[
        ["id_site", "student_reg_id", "date", "sum_click"]
    ].copy()


def write_dataset(out_path: Path, table_dfs: dict[str, pd.DataFrame]) -> dict:
    """Write normalized tables to db/*.parquet and write dataset manifest.yaml."""
    db_dir = out_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving database tables to parquet...")
    for tname, df in table_dfs.items():
        pq_path_db = db_dir / f"{tname}.parquet"
        df.to_parquet(pq_path_db, index=False)
        print(f"  - {tname:20s}: {len(df):>10,} rows -> {pq_path_db}")

    dataset_manifest = {
        "name": "oulad",
        "manifest_version": 1,
        "val_timestamp": "2014-02-01",
        "test_timestamp": "2014-10-01",
        "description": "Normalized OULAD relational database formatted for RelBench.",
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
                "pkey": "id_student",
                "time_col": None,
                "fkeys": {},
            },
            "studentRegistration": {
                "pkey": "student_reg_id",
                "time_col": None,
                "fkeys": {
                    "id_student": "studentInfo",
                    "course_id": "courses",
                },
            },
            "studentAssessment": {
                "pkey": "student_assessment_id",
                "time_col": "date_submitted",
                "fkeys": {
                    "id_assessment": "assessments",
                    "student_reg_id": "studentRegistration",
                },
            },
            "studentVle": {
                "pkey": None,
                "time_col": "date",
                "fkeys": {
                    "id_site": "vle",
                    "student_reg_id": "studentRegistration",
                },
            },
        },
    }

    with open(out_path / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_manifest, f, sort_keys=False)
    print(f"\nCreated dataset manifest at: {out_path / 'manifest.yaml'}")
    return dataset_manifest


def build_student_fail_task(
    out_path: Path, student_reg_df: pd.DataFrame, reg_pres_series: pd.Series
) -> None:
    """Create train/val/test parquet splits and manifest.yaml for student-fail."""
    print("\nConstructing prediction task 'student-fail'...")
    task_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(reg_pres_series.map(PRESENTATION_STARTS)),
            "student_reg_id": student_reg_df["student_reg_id"],
            "failed": student_reg_df["final_result"]
            .isin(["Fail", "Withdrawn"])
            .astype(np.int64),
        }
    )

    train_mask = reg_pres_series.isin(["2013B", "2013J"])
    val_mask = reg_pres_series == "2014B"
    test_mask = reg_pres_series == "2014J"

    train_df = task_df.loc[train_mask].reset_index(drop=True)
    val_df = task_df.loc[val_mask].reset_index(drop=True)
    test_df = task_df.loc[test_mask].reset_index(drop=True)

    print("Task splits:")
    print(f"  Train: {len(train_df):>6,} rows | Fail rate: {train_df['failed'].mean():.1%}")
    print(f"  Val  : {len(val_df):>6,} rows | Fail rate: {val_df['failed'].mean():.1%}")
    print(f"  Test : {len(test_df):>6,} rows | Fail rate: {test_df['failed'].mean():.1%}")

    task_manifest = {
        "name": "student-fail",
        "kind": "external",
        "task_type": "binary_classification",
        "description": "Predict whether a student will fail or withdraw from a course presentation.",
        "entity_table": "studentRegistration",
        "entity_col": "student_reg_id",
        "target_col": "failed",
        "time_col": "timestamp",
        "timedelta": "30 days",
        "remove_columns": [
            ["studentRegistration", "final_result"],
            ["studentRegistration", "date_unregistration"],
        ],
        "manifest_version": 1,
    }

    task_dir = out_path / "tasks" / "student-fail"
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(task_manifest, f, sort_keys=False)
    train_df.to_parquet(task_dir / "train.parquet", index=False)
    val_df.to_parquet(task_dir / "val.parquet", index=False)
    test_df.to_parquet(task_dir / "test.parquet", index=False)


def print_schema_summary(
    table_dfs: dict[str, pd.DataFrame], dataset_manifest: dict
) -> None:
    """Print normalized RelBench schema overview."""
    print("\nNORMALIZED DB SCHEMA")
    for idx, (tname, df) in enumerate(table_dfs.items(), start=1):
        tmeta = dataset_manifest["tables"][tname]
        header = [f"{idx}. {tname} ({len(df):,})", f"PK: {tmeta['pkey']}"]
        if tmeta["fkeys"]:
            fk_str = ", ".join(f"{k} -> {v}" for k, v in tmeta["fkeys"].items())
            header.append(f"FK: {fk_str}")
        if tmeta["time_col"]:
            header.append(f"TIME: {tmeta['time_col']}")
        cols_str = ", ".join(f"{c} ({dt})" for c, dt in df.dtypes.items())
        print(" | ".join(header))
        print(f"   cols: {cols_str}")


def prepare_data(raw_dir: str | Path, out_dir: str | Path) -> None:
    raw_path = Path(raw_dir).resolve()
    out_path = Path(out_dir).resolve()

    print(f"Reading raw OULAD CSV files from: {raw_path}")
    print(f"Output RelBench dataset directory: {out_path}")

    courses_df, course_to_pkey = prepare_courses(raw_path)
    assessments_df, assess_raw_to_pkey, assess_to_pres, assess_to_raw_course = (
        prepare_assessments(raw_path, course_to_pkey)
    )
    vle_df, site_raw_to_pkey = prepare_vle(raw_path, course_to_pkey)
    student_info_df, raw_student_info, student_to_pkey = prepare_student_info(raw_path)
    student_reg_df, student_reg_to_pkey, reg_pres_series = (
        prepare_student_registration(
            raw_path, raw_student_info, course_to_pkey, student_to_pkey
        )
    )
    student_assess_df = prepare_student_assessment(
        raw_path,
        assess_raw_to_pkey,
        assess_to_pres,
        assess_to_raw_course,
        student_reg_to_pkey,
    )
    student_vle_df = prepare_student_vle(
        raw_path, site_raw_to_pkey, student_reg_to_pkey
    )

    table_dfs = {
        "courses": courses_df,
        "assessments": assessments_df,
        "vle": vle_df,
        "studentInfo": student_info_df,
        "studentRegistration": student_reg_df,
        "studentAssessment": student_assess_df,
        "studentVle": student_vle_df,
    }

    dataset_manifest = write_dataset(out_path, table_dfs)
    build_student_fail_task(out_path, student_reg_df, reg_pres_series)
    print_schema_summary(table_dfs, dataset_manifest)

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare OULAD dataset for RelBench.")
    parser.add_argument(
        "--raw_dir", type=str, default="../datasets/OULAD", help="Path to raw CSV files."
    )
    parser.add_argument(
        "--out_dir", type=str, default="output", help="Path to output directory."
    )
    args = parser.parse_args()

    prepare_data(raw_dir=args.raw_dir, out_dir=args.out_dir)
