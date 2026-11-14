from pathlib import Path
import duckdb
import yaml

dataset_dir = Path("hw-1/output").resolve()
db_path = dataset_dir / "oulad_views.duckdb"
db_path.unlink(missing_ok=True)  # remove the version-999 file first

manifest = yaml.safe_load((dataset_dir / "manifest.yaml").read_text(encoding="utf-8"))

con = duckdb.connect(":memory:")
con.execute(f"ATTACH '{db_path.as_posix()}' AS oulad (STORAGE_VERSION 'v1.2.0');")

for tname in manifest["tables"]:
    pq_path = (dataset_dir / "db" / f"{tname}.parquet").as_posix()
    con.execute(
        f"CREATE OR REPLACE VIEW oulad.main.{tname} AS SELECT * FROM read_parquet('{pq_path}');"
    )

con.close()
print(f"Created compatible DuckDB file at: {db_path}")