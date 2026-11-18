"""Evaluates the 3 homework baselines on the student-score regression task:

1. Global & Historical Mean
2. XGBoost on target table only (entity + timestamp)
3. XGBoost on join-aggregate features (DFS across multiple tables)
"""

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import relbench
from sklearn.metrics import mean_absolute_error
import xgboost as xgb


def main():
    print("=" * 70)
    print("EVALUATING BASELINES ON RELBENCH 'student-score' REGRESSION TASK")
    print("=" * 70)

    # 1. Load dataset & task
    dataset_dir = Path("output")
    dataset = relbench.load_dataset(dataset_dir)
    task = dataset.load_task("student-score")
    db = dataset.get_db()  # cut at test_timestamp

    train_table = task.get_table("train")
    val_table = task.get_table("val")

    train_df = train_table.df
    val_df = val_table.df

    print(f"Loaded Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")

    results = {}

    def report(name: str, y_pred: np.ndarray):
        y_pred = np.asarray(y_pred, dtype=float)
        mae = mean_absolute_error(val_df["score"], y_pred)
        nmae = task.evaluate(y_pred, val_table)["nmae"]
        results[name] = {"Val MAE": mae, "Val NMAE": nmae}
        print(f"  {name:35s} | MAE: {mae:.3f} | nMAE: {nmae:.3f}")

    # -------------------------------------------------------------
    # BASELINE 1: Global Mean & Historical Per-Student Mean
    # -------------------------------------------------------------
    print("\n--- BASELINE 1: Historical Mean ---")
    # 1a. Global Mean
    global_mean = train_df["score"].mean()
    pred_global = np.full(len(val_df), global_mean)
    report("1a. Global Mean", pred_global)

    # 1b. Per-student historical mean
    # Note: student_reg_id connects to studentInfo, which has id_student
    student_info = db.table_dict["studentInfo"].df
    train_with_student = train_df.merge(
        student_info[["student_reg_id", "id_student"]], on="student_reg_id", how="left"
    )
    val_with_student = val_df.merge(
        student_info[["student_reg_id", "id_student"]], on="student_reg_id", how="left"
    )

    student_means = train_with_student.groupby("id_student")["score"].mean()
    pred_hist = val_with_student["id_student"].map(student_means).fillna(global_mean).values
    report("1b. Per-Student Historical Mean", pred_hist)

    # -------------------------------------------------------------
    # BASELINE 2: XGBoost on Target Table Only (entity + timestamp)
    # -------------------------------------------------------------
    print("\n--- BASELINE 2: XGBoost on Target Table Only ---")
    # Target table features: only student_reg_id and temporal encoding of timestamp
    def extract_target_table_features(df):
        X = pd.DataFrame(index=df.index)
        X["student_reg_id"] = df["student_reg_id"].astype(int)
        X["year"] = pd.to_datetime(df["timestamp"]).dt.year
        X["month"] = pd.to_datetime(df["timestamp"]).dt.month
        return X

    X_train_t = extract_target_table_features(train_df)
    y_train = train_df["score"].values
    X_val_t = extract_target_table_features(val_df)

    model_b2 = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model_b2.fit(X_train_t, y_train)
    pred_b2 = model_b2.predict(X_val_t)
    report("2. XGBoost (Target Table Only)", pred_b2)

    # -------------------------------------------------------------
    # BASELINE 3: Deep Feature Synthesis (Multi-Table Aggregation)
    # -------------------------------------------------------------
    print("\n--- BASELINE 3: Deep Feature Synthesis / Relational Features ---")

    # We extract features using DuckDB across courses, studentInfo, studentVle, and past assessments
    FEATURE_SQL = """
    SELECT
        t.student_reg_id,
        t.timestamp,
        -- Student Demographics
        ANY_VALUE(si.gender)               AS gender,
        ANY_VALUE(si.highest_education)    AS highest_education,
        ANY_VALUE(si.imd_band)             AS imd_band,
        ANY_VALUE(si.age_band)             AS age_band,
        ANY_VALUE(si.num_of_prev_attempts) AS num_of_prev_attempts,
        ANY_VALUE(si.studied_credits)      AS studied_credits,
        ANY_VALUE(si.disability)           AS disability,
        
        -- Course Information
        ANY_VALUE(c.module_presentation_length) AS presentation_length,
        
        -- Registration Timing
        ANY_VALUE(sr.date_registration)    AS days_before_start_registered,
        
        -- VLE Engagement (Strictly before prediction timestamp t)
        COUNT(v.date)                      AS num_vle_interactions,
        COALESCE(SUM(v.sum_click), 0)      AS total_clicks,
        COALESCE(AVG(v.sum_click), 0)      AS avg_clicks_per_interaction,
        COALESCE(COUNT(DISTINCT v.date), 0) AS active_vle_days,
        COALESCE(COUNT(DISTINCT v.id_site), 0) AS distinct_sites_visited
        
    FROM task_rows t
    JOIN studentInfo si ON si.student_reg_id = t.student_reg_id
    JOIN courses c ON c.course_id = si.course_id
    LEFT JOIN studentRegistration sr ON sr.student_reg_id = t.student_reg_id
    LEFT JOIN studentVle v
           ON v.student_reg_id = t.student_reg_id
          AND v.date < t.timestamp
    GROUP BY t.student_reg_id, t.timestamp
    """

    con = duckdb.connect()
    con.register("courses", db.table_dict["courses"].df)
    con.register("studentInfo", db.table_dict["studentInfo"].df)
    con.register("studentRegistration", db.table_dict["studentRegistration"].df)
    con.register("studentVle", db.table_dict["studentVle"].df)

    con.register("task_rows", train_df[["student_reg_id", "timestamp"]])
    train_features = con.sql(FEATURE_SQL).df()

    con.register("task_rows", val_df[["student_reg_id", "timestamp"]])
    val_features = con.sql(FEATURE_SQL).df()
    con.close()

    # Align with task table order
    X_train_full = train_df[["student_reg_id", "timestamp"]].merge(
        train_features, on=["student_reg_id", "timestamp"], how="left"
    )
    X_val_full = val_df[["student_reg_id", "timestamp"]].merge(
        val_features, on=["student_reg_id", "timestamp"], how="left"
    )

    feature_cols = [c for c in X_train_full.columns if c not in ["student_reg_id", "timestamp"]]

    # Convert categoricals to category dtype for XGBoost
    cat_cols = ["gender", "highest_education", "imd_band", "age_band", "disability"]
    for c in cat_cols:
        X_train_full[c] = X_train_full[c].astype("category")
        X_val_full[c] = X_val_full[c].astype("category")

    model_b3 = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        enable_categorical=True,
        random_state=42,
    )
    model_b3.fit(X_train_full[feature_cols], y_train)
    pred_b3 = model_b3.predict(X_val_full[feature_cols])
    report("3. XGBoost (Multi-Table DFS Features)", pred_b3)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY COMPARISON TABLE")
    print("=" * 70)
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())


if __name__ == "__main__":
    main()
