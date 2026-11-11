import os
from pathlib import Path
import pandas as pd
import yaml
from relbench.base import Table, Database


def load_oulad_relbench(raw_dir: str | Path | None = None) -> Database:
    """Load the OULAD dataset from raw CSV files into an in-memory RelBench Database object.

    Handles combining composite keys into single surrogate columns as required by RelBench.
    """
    if raw_dir is None:
        raw_path = Path(__file__).resolve().parent.parent / "oulad"
    else:
        raw_path = Path(raw_dir)
    print(f"Loading CSV files from {raw_path.resolve()}...")

    # 1. Load CSVs
    courses_df = pd.read_csv(raw_path / "courses.csv")
    assessments_df = pd.read_csv(raw_path / "assessments.csv")
    vle_df = pd.read_csv(raw_path / "vle.csv")
    student_info_df = pd.read_csv(raw_path / "studentInfo.csv")
    student_reg_df = pd.read_csv(raw_path / "studentRegistration.csv")
    student_assess_df = pd.read_csv(raw_path / "studentAssessment.csv")

    # Optional / large file: load studentVle if present
    student_vle_path = raw_path / "studentVle.csv"
    has_student_vle = student_vle_path.exists()
    student_vle_df = None
    if has_student_vle:
        print("Loading studentVle.csv (this may take a few seconds)...")
        student_vle_df = pd.read_csv(student_vle_path)

    # 2. RelBench requires single-column primary and foreign keys.
    # We construct unified key columns for composite relationships:
    # course_id: code_module + '_' + code_presentation
    # student_reg_id: code_module + '_' + code_presentation + '_' + id_student

    courses_df["course_id"] = (
        courses_df["code_module"] + "_" + courses_df["code_presentation"]
    )

    assessments_df["course_id"] = (
        assessments_df["code_module"] + "_" + assessments_df["code_presentation"]
    )

    vle_df["course_id"] = (
        vle_df["code_module"] + "_" + vle_df["code_presentation"]
    )

    student_info_df["course_id"] = (
        student_info_df["code_module"] + "_" + student_info_df["code_presentation"]
    )
    student_info_df["student_reg_id"] = (
        student_info_df["course_id"] + "_" + student_info_df["id_student"].astype(str)
    )

    student_reg_df["course_id"] = (
        student_reg_df["code_module"] + "_" + student_reg_df["code_presentation"]
    )
    student_reg_df["student_reg_id"] = (
        student_reg_df["course_id"] + "_" + student_reg_df["id_student"].astype(str)
    )

    if student_vle_df is not None:
        student_vle_df["course_id"] = (
            student_vle_df["code_module"] + "_" + student_vle_df["code_presentation"]
        )
        student_vle_df["student_reg_id"] = (
            student_vle_df["course_id"] + "_" + student_vle_df["id_student"].astype(str)
        )

    # 3. Build RelBench Table objects
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
            pkey_col="id_assessment",
            time_col=None,
        ),
        "vle": Table(
            df=vle_df,
            fkey_col_to_pkey_table={"course_id": "courses"},
            pkey_col="id_site",
            time_col=None,
        ),
        "studentInfo": Table(
            df=student_info_df,
            fkey_col_to_pkey_table={"course_id": "courses"},
            pkey_col="student_reg_id",
            time_col=None,
        ),
        "studentRegistration": Table(
            df=student_reg_df,
            fkey_col_to_pkey_table={"student_reg_id": "studentInfo"},
            pkey_col="student_reg_id",
            time_col=None,
        ),
        "studentAssessment": Table(
            df=student_assess_df,
            fkey_col_to_pkey_table={"id_assessment": "assessments"},
            pkey_col=None,
            time_col=None,
        ),
    }

    if student_vle_df is not None:
        table_dict["studentVle"] = Table(
            df=student_vle_df,
            fkey_col_to_pkey_table={
                "id_site": "vle",
                "student_reg_id": "studentInfo",
            },
            pkey_col=None,
            time_col=None,
        )

    # 4. Construct and return Database
    db = Database(table_dict=table_dict)
    return db


def export_relbench_dataset(
    db: Database,
    out_dir: str | Path,
    dataset_name: str = "oulad",
    val_timestamp: str = "2014-06-01",
    test_timestamp: str = "2014-09-01",
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
