from pathlib import Path
import sqlite3
import pandas as pd

# 1. Define the folder path and database name
csv_folder = Path("./../oulad")  # Replace with your folder path
db_path = "oulad_database.db"

# 2. Connect to the SQLite database
conn = sqlite3.connect(db_path)

# 3. Loop through all CSV files in the directory
for file_path in csv_folder.glob("*.csv"):
    # Use the file name (without .csv) as the SQL table name
    table_name = file_path.stem
    print(f"Importing {file_path.name} into table '{table_name}'...")

    try:
        # Load CSV into DataFrame
        df = pd.read_csv(file_path)

        # Write to SQLite (creates a new table for each file)
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    except Exception as e:
        print(f"Failed to import {file_path.name}: {e}")

# 4. Close the connection
conn.close()
print("All files processed successfully!")
