"""Minimal example: reading an exercise 1 submission with RelBench.

    python load_example.py output/output_dir task_name
"""

import sys
from pathlib import Path

import relbench
import yaml

dataset_dir, task_name = Path(sys.argv[1]), sys.argv[2]

# Reads <dataset_dir>/manifest.yaml; tables are loaded lazily by get_db().
dataset = relbench.load_dataset(dataset_dir)
print(dataset.get_task_names(), dataset.val_timestamp, dataset.test_timestamp)

# The database, cut at test_timestamp (the default), so test-period rows are never input.
db = dataset.get_db()
for name, table in db.table_dict.items():
    print(f"{name:15s} rows={len(table):>8,}  pkey={table.pkey_col}  "
          f"time={table.time_col}  fkeys={table.fkey_col_to_pkey_table}")

# The task: reads tasks/<task>/manifest.yaml and {train,val,test}.parquet.
task = dataset.load_task(task_name)
train = task.get_table("train")  # columns: timestamp, entity id, label
val = task.get_table("val")
test = task.get_table("test")    # label column hidden by default
print(train.df.head())
print("test columns:", list(test.df.columns))