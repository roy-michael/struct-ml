import argparse
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Dataset & task configuration
KAGGLE_DATASET = "darkmatternet/us-severe-weather-losses-1950-2026"
START_YEAR = 2006  # Consistent episode_id, narratives, and all 48 NOAA event types
END_YEAR = 2025    # Last complete calendar year (avoids partial 2026 backfill lag)
VAL_TIMESTAMP = "2022-01-01"
TEST_TIMESTAMP = "2024-01-01"
HORIZON_DAYS = 90  # Quarterly non-overlapping prediction horizon
DAMAGE_THRESHOLD_USD = 0.0

SCHEMA = {
    "states":          {"pkey": "state_id",        "time_col": None,              "fkeys": {}},
    "zones":           {"pkey": "zone_id",         "time_col": None,              "fkeys": {"state_id": "states"}},
    "places":          {"pkey": "place_id",        "time_col": None,              "fkeys": {"zone_id": "zones"}},
    "episodes":        {"pkey": "episode_id",      "time_col": "begin_date_time", "fkeys": {"state_id": "states"}},
    "episode_zones":   {"pkey": "episode_zone_id", "time_col": "begin_date_time", "fkeys": {"episode_id": "episodes", "zone_id": "zones"}},
    "events":          {"pkey": "event_id",        "time_col": "begin_date_time", "fkeys": {"episode_zone_id": "episode_zones"}},
    "event_locations": {"pkey": "location_id",     "time_col": "begin_date_time", "fkeys": {"event_id": "events", "place_id": "places"}},
    "fatalities":      {"pkey": "fatality_id",     "time_col": "fatality_date",   "fkeys": {"event_id": "events"}},
}

def read_raw_table(raw_path: Path, stem: str) -> pd.DataFrame:
    """Load `<stem>.parquet` (or fallback `<stem>.csv`) with lowercase column names."""
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", lambda p: pd.read_csv(p, low_memory=False))):
        path = raw_path / f"{stem}{ext}"
        if path.exists():
            df = reader(path)
            df.columns = [c.strip().lower() for c in df.columns]
            print(f"  Loaded {path.name} ({len(df):,} rows)")
            return df
    raise FileNotFoundError(f"Could not find {stem}.parquet or {stem}.csv in {raw_path}")


def load_raw_events(raw_path: Path) -> pd.DataFrame:
    """Load and clean raw storm events within [START_YEAR, END_YEAR]."""
    df = read_raw_table(raw_path, "storm_events")
    df["begin_date_time"] = pd.to_datetime(df["begin_datetime"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["begin_date_time", "event_id", "episode_id"])
    df = df[df["begin_date_time"].dt.year.between(START_YEAR, END_YEAR)].copy()

    df["raw_event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype(np.int64)
    df["raw_episode_id"] = pd.to_numeric(df["episode_id"], errors="coerce").astype(np.int64)
    df["state"] = df["state"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["state_fips"] = pd.to_numeric(df["state_fips"], errors="coerce").fillna(-1).astype(np.int64)
    df["cz_type"] = df["cz_type"].fillna("U").astype(str).str.strip().str.upper()
    df["cz_name"] = df["cz_name"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["cz_fips"] = pd.to_numeric(df["cz_fips"], errors="coerce").fillna(-1).astype(np.int64)
    df["raw_zone_key"] = df["state"] + "_" + df["cz_type"] + "_" + df["cz_name"]

    cpi = pd.to_numeric(df["cpi_adjustment_factor"], errors="coerce").fillna(1.0).astype(np.float64)
    df["damage_property_usd"] = df["property_damage_usd_nominal"].fillna(0.0).astype(np.float64) * cpi
    df["damage_crops_usd"] = df["crop_damage_usd_nominal"].fillna(0.0).astype(np.float64) * cpi
    df["total_damage_usd"] = df["total_damage_usd_adjusted"].fillna(0.0).astype(np.float64)
    return df


def build_database_tables(raw_path: Path) -> dict[str, pd.DataFrame]:
    """Normalize raw NOAA files into the 8-table RelBench schema."""
    raw = load_raw_events(raw_path)

    # 1. states (static)
    states = (
        raw[["state", "state_fips"]]
        .drop_duplicates(subset=["state"])
        .sort_values("state", ignore_index=True)
    )
    states.insert(0, "state_id", np.arange(len(states), dtype=np.int64))
    state_map = dict(zip(states["state"], states["state_id"]))

    # 2. zones (static entity table)
    zones = (
        raw[["raw_zone_key", "state", "cz_type", "cz_name", "cz_fips"]]
        .drop_duplicates(subset=["raw_zone_key"])
        .sort_values("raw_zone_key", ignore_index=True)
    )
    zones.insert(0, "zone_id", np.arange(len(zones), dtype=np.int64))
    zones["state_id"] = zones["state"].map(state_map).astype(np.int64)
    zone_map = dict(zip(zones["raw_zone_key"], zones["zone_id"]))
    zones = zones[["zone_id", "state_id", "cz_type", "cz_name", "cz_fips"]]

    # 3. episodes (temporal)
    episodes = (
        raw.groupby("raw_episode_id", as_index=False)
        .agg(
            begin_date_time=("begin_date_time", "min"),
            state=("state", "first"),
            episode_narrative=("episode_narrative", "first"),
        )
        .sort_values(["begin_date_time", "raw_episode_id"], ignore_index=True)
    )
    episodes["episode_id"] = np.arange(len(episodes), dtype=np.int64)
    episodes["state_id"] = episodes["state"].map(state_map).astype(np.int64)
    episodes["episode_narrative"] = episodes["episode_narrative"].fillna("").astype(str)
    ep_map = dict(zip(episodes["raw_episode_id"], episodes["episode_id"]))
    episodes = episodes[["episode_id", "state_id", "begin_date_time", "episode_narrative"]]

    # 4. episode_zones (many-to-many temporal bridge)
    raw["episode_id"] = raw["raw_episode_id"].map(ep_map).astype(np.int64)
    raw["zone_id"] = raw["raw_zone_key"].map(zone_map).astype(np.int64)
    episode_zones = (
        raw.groupby(["episode_id", "zone_id"], as_index=False)["begin_date_time"]
        .min()
        .sort_values(["begin_date_time", "episode_id", "zone_id"], ignore_index=True)
    )
    episode_zones.insert(0, "episode_zone_id", np.arange(len(episode_zones), dtype=np.int64))

    # 5. events (temporal)
    ev = (
        raw.drop_duplicates(subset=["raw_event_id"])
        .merge(episode_zones[["episode_id", "zone_id", "episode_zone_id"]], on=["episode_id", "zone_id"], how="left")
        .sort_values(["begin_date_time", "raw_event_id"], ignore_index=True)
    )
    ev["event_id"] = np.arange(len(ev), dtype=np.int64)
    for col, fill in [("event_type", "UNKNOWN"), ("source", "UNKNOWN"), ("magnitude_type", "NONE"), ("event_narrative", "")]:
        ev[col] = ev[col].fillna(fill).astype(str)
    ev["magnitude"] = pd.to_numeric(ev["magnitude"], errors="coerce")
    for col in ("injuries_direct", "injuries_indirect", "deaths_direct", "deaths_indirect"):
        ev[col] = pd.to_numeric(ev[col], errors="coerce").fillna(0).astype(np.int64)

    event_keys = ev[["raw_event_id", "event_id", "zone_id", "begin_date_time"]]
    event_cols = [
        "event_id", "episode_zone_id", "begin_date_time", "event_type", "source",
        "magnitude", "magnitude_type", "damage_property_usd", "damage_crops_usd",
        "total_damage_usd", "injuries_direct", "injuries_indirect",
        "deaths_direct", "deaths_indirect", "event_narrative",
    ]
    events = ev[event_cols]

    # 6. places (static) & 7. event_locations (temporal)
    loc = read_raw_table(raw_path, "storm_locations").rename(columns={"event_id": "raw_event_id"})
    loc["raw_event_id"] = pd.to_numeric(loc["raw_event_id"], errors="coerce")
    loc = loc.merge(event_keys, on="raw_event_id", how="inner")
    loc["place_name"] = loc["location"].fillna("").astype(str).str.strip().str.upper()
    loc = loc[loc["place_name"] != ""].copy()

    places = (
        loc[["zone_id", "place_name"]]
        .drop_duplicates()
        .sort_values(["zone_id", "place_name"], ignore_index=True)
    )
    places.insert(0, "place_id", np.arange(len(places), dtype=np.int64))

    event_locations = loc.merge(places, on=["zone_id", "place_name"], how="left")
    event_locations["latitude"] = pd.to_numeric(event_locations["latitude"], errors="coerce")
    event_locations["longitude"] = pd.to_numeric(event_locations["longitude"], errors="coerce")
    event_locations["location_index"] = (
        pd.to_numeric(event_locations["location_index"], errors="coerce").fillna(-1).astype(np.int64)
    )
    event_locations = event_locations[
        ["event_id", "place_id", "begin_date_time", "latitude", "longitude", "location_index"]
    ].sort_values(["begin_date_time", "event_id", "place_id"], ignore_index=True)
    event_locations.insert(0, "location_id", np.arange(len(event_locations), dtype=np.int64))

    # 8. fatalities (temporal; fatality_date falls back to event begin_date_time when null)
    fat = read_raw_table(raw_path, "storm_fatalities").rename(columns={"event_id": "raw_event_id"})
    fat["raw_event_id"] = pd.to_numeric(fat["raw_event_id"], errors="coerce")
    fat = fat.merge(event_keys[["raw_event_id", "event_id", "begin_date_time"]], on="raw_event_id", how="inner")
    fat_dt = pd.to_datetime(fat["fatality_date"], errors="coerce").dt.tz_localize(None)
    fat["fatality_date"] = fat_dt.astype(fat["begin_date_time"].dtype).fillna(fat["begin_date_time"])
    fatalities = fat[["event_id", "fatality_date"]].sort_values(["fatality_date", "event_id"], ignore_index=True)
    fatalities.insert(0, "fatality_id", np.arange(len(fatalities), dtype=np.int64))

    return {
        "states": states,
        "zones": zones,
        "places": places,
        "episodes": episodes,
        "episode_zones": episode_zones,
        "events": events,
        "event_locations": event_locations,
        "fatalities": fatalities,
    }


def write_task(out_path: Path, name: str, manifest: dict, splits: dict[str, pd.DataFrame]) -> None:
    task_dir = out_path / "tasks" / name
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    for split_name, df in splits.items():
        df.to_parquet(task_dir / f"{split_name}.parquet", index=False)
    print(f"Created task at: {task_dir}")


def build_zone_tasks(
    events_df: pd.DataFrame,
    episode_zones_df: pd.DataFrame,
    out_path: Path,
    val_timestamp: str = VAL_TIMESTAMP,
    test_timestamp: str = TEST_TIMESTAMP,
    horizon_days: int = HORIZON_DAYS,
    damage_threshold: float = DAMAGE_THRESHOLD_USD,
) -> None:
    """Build quarterly 90-day zone-level classification ('zone-damage') and regression ('zone-loss') tasks."""
    val_ts, test_ts = pd.Timestamp(val_timestamp), pd.Timestamp(test_timestamp)
    max_ts = events_df["begin_date_time"].max()
    horizon = pd.Timedelta(days=horizon_days)

    ev_zone = events_df[["episode_zone_id", "begin_date_time", "total_damage_usd"]].merge(
        episode_zones_df[["episode_zone_id", "zone_id"]], on="episode_zone_id", how="left"
    )
    train_counts = ev_zone.loc[ev_zone["begin_date_time"] < val_ts, "zone_id"].value_counts()
    active_zones = np.sort(train_counts[train_counts >= 5].index.to_numpy(dtype=np.int64))

    # Per-zone sorted positive-damage timestamps and prefix sums for O(log N) window sums
    dmg_df = ev_zone.loc[ev_zone["total_damage_usd"] > 0.0, ["zone_id", "begin_date_time", "total_damage_usd"]]
    dmg_df = dmg_df.sort_values(["zone_id", "begin_date_time"])
    dmg_by_zone: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for zid, grp in dmg_df.groupby("zone_id", sort=False):
        times = grp["begin_date_time"].to_numpy(dtype="datetime64[ns]")
        vals = grp["total_damage_usd"].to_numpy(dtype=np.float64)
        csum = np.concatenate(([0.0], np.cumsum(vals)))
        dmg_by_zone[int(zid)] = (times, csum)

    anchors_by_split = {
        "train": pd.date_range("2007-01-01", val_ts - horizon, freq="QS"),
        "val": pd.date_range(val_ts, test_ts - horizon, freq="QS"),
        "test": pd.date_range(test_ts, max_ts - horizon, freq="QS"),
    }

    damage_splits: dict[str, pd.DataFrame] = {}
    loss_splits: dict[str, pd.DataFrame] = {}
    n_zones = len(active_zones)

    for split_name, anchors in anchors_by_split.items():
        starts = anchors.to_numpy(dtype="datetime64[ns]")
        ends = (anchors + horizon).to_numpy(dtype="datetime64[ns]")
        rows_ts = pd.to_datetime(np.repeat(starts, n_zones))
        rows_zid = np.tile(active_zones, len(starts)).astype(np.int64)
        window_dmg = np.zeros(len(rows_ts), dtype=np.float64)

        for z_idx, zid in enumerate(active_zones):
            entry = dmg_by_zone.get(int(zid))
            if entry is None:
                continue
            ev_times, csum = entry
            left = np.searchsorted(ev_times, starts, side="left")
            right = np.searchsorted(ev_times, ends, side="left")
            window_dmg[z_idx::n_zones] = np.maximum(csum[right] - csum[left], 0.0)

        damage_splits[split_name] = pd.DataFrame(
            {"timestamp": rows_ts, "zone_id": rows_zid, "has_damage": (window_dmg > damage_threshold).astype(np.int64)}
        )
        loss_splits[split_name] = pd.DataFrame(
            {"timestamp": rows_ts, "zone_id": rows_zid, "loss": np.log1p(window_dmg)}
        )

    base_manifest = {
        "manifest_version": 1,
        "kind": "external",
        "entity_table": "zones",
        "entity_col": "zone_id",
        "time_col": "timestamp",
        "timedelta": f"{horizon_days} days",
    }

    print("\nTask 'zone-damage' (binary classification):")
    for s, df in damage_splits.items():
        print(f"  {s:5s}: {len(df):>8,} rows | Positive rate: {df['has_damage'].mean() * 100:5.2f}%")
    write_task(
        out_path,
        "zone-damage",
        {
            "name": "zone-damage",
            **base_manifest,
            "task_type": "binary_classification",
            "description": (
                f"Predict whether an NWS county/forecast zone will experience a damaging severe weather "
                f"event (total_damage_usd > {damage_threshold}) within the next {horizon_days} days."
            ),
            "target_col": "has_damage",
            "metrics": ["roc_auc"],
        },
        damage_splits,
    )


def prepare_data(raw_dir: str | Path, out_dir: str | Path = "output") -> None:
    raw_path = Path(raw_dir).resolve()
    out_path = Path(out_dir).resolve()
    if out_path.exists():
        shutil.rmtree(out_path)
    db_dir = out_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw NOAA Storm Events files from: {raw_path}")
    print(f"Output RelBench dataset directory: {out_path}\n")

    tables = build_database_tables(raw_path)

    print("\nSaving database tables to db/*.parquet...")
    for name, df in tables.items():
        df.to_parquet(db_dir / f"{name}.parquet", index=False)
        print(f"  - {name:16s}: {len(df):>10,} rows")
    print(f"  Total tuples: {sum(len(df) for df in tables.values()):,}")

    manifest = {
        "name": "rel-noaa-storms",
        "manifest_version": 1,
        "val_timestamp": VAL_TIMESTAMP,
        "test_timestamp": TEST_TIMESTAMP,
        "description": (
            f"NOAA Storm Events and severe weather losses ({START_YEAR}-{END_YEAR}) formatted for RelBench: "
            "states, county/forecast zones, named places, storm episodes, episode-zone footprints, "
            "hazard events, geo-referenced event locations, and fatalities."
        ),
        "tables": SCHEMA,
    }
    with open(out_path / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    build_zone_tasks(tables["events"], tables["episode_zones"], out_path)
    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NOAA Storm Events dataset for RelBench.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=None,
        required=True,
        help="Path to raw NOAA CSV/Parquet files. If omitted, downloads automatically via kagglehub.",
    )
    parser.add_argument("--out_dir", type=str, default="output", help="Path to output RelBench directory.")
    args = parser.parse_args()
    prepare_data(raw_dir=args.raw_dir, out_dir=args.out_dir)