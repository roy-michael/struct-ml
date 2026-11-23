"""Evaluates the 3 homework baselines on the NOAA Storm Events 'zone-damage' classification task:

1. Historical Majority / Historical Rate (per-zone past target table records, fallback to global)
2. XGBoost on Target Table Only (entity + timestamp)
3. Deep Feature Synthesis (both automatic multi-hop schema-driven DFS and manual multi-table join-aggregate features)
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import relbench
from sklearn.metrics import roc_auc_score
import xgboost as xgb


def auto_dfs_features(
    db: relbench.base.Database,
    all_task_rows: pd.DataFrame,
    entity_table: str = "zones",
    entity_col: str = "zone_id",
    time_col: str = "timestamp",
    max_depth: int = 3,
) -> pd.DataFrame:
    """Automatic multi-hop schema-driven Deep Feature Synthesis (DFS) across RelBench tables.

    Traverses foreign-key paths in ``db.table_dict`` programmatically via BFS up to ``max_depth``:
    - 1-hop parent tables (e.g., zones -> states): attach static parent attributes
    - Static child tables (e.g., places -> zones): aggregate entity child COUNT
    - 1-, 2-, and 3-hop temporal child chains (e.g., episode_zones -> zones,
      events -> episode_zones -> zones, event_locations -> places -> zones,
      fatalities -> events -> episode_zones -> zones): COUNT, SUM, MEAN, MAX of numeric columns
    - 2-hop sibling temporal tables via parent (e.g., episodes -> states <- zones): parent COUNT

    All temporal aggregations are strictly leak-free (only records with ``tbl.time_col < t.timestamp``).
    """
    con = duckdb.connect()
    for tname, tbl in db.table_dict.items():
        con.register(tname, tbl.df)

    anchors = all_task_rows[[entity_col, time_col]].drop_duplicates().sort_values([entity_col, time_col])
    con.register("task_anchors", anchors)
    con.execute(
        f"""
        CREATE TEMP TABLE anchor_grid AS
        SELECT {entity_col}, {time_col}, DATE_TRUNC('quarter', {time_col}) AS q_ts
        FROM task_anchors
        """
    )

    result = anchors.reset_index(drop=True)
    ent_tbl = db.table_dict[entity_table]

    # 1. Entity table static columns + 1-hop parent static columns
    ent_non_key = [
        c
        for c in ent_tbl.df.columns
        if c != ent_tbl.pkey_col and c not in ent_tbl.fkey_col_to_pkey_table
    ]
    select_parts = [f"e.{entity_col}"] + [f"e.{c} AS auto_{entity_table}_{c}" for c in ent_non_key]
    joins = []
    for fk_col, parent_name in ent_tbl.fkey_col_to_pkey_table.items():
        parent_tbl = db.table_dict[parent_name]
        joins.append(f"LEFT JOIN {parent_name} p_{parent_name} ON p_{parent_name}.{parent_tbl.pkey_col} = e.{fk_col}")
        select_parts.append(f"e.{fk_col} AS auto_{entity_table}_{fk_col}")
        for pc in parent_tbl.df.columns:
            if pc != parent_tbl.pkey_col and pc not in parent_tbl.fkey_col_to_pkey_table:
                select_parts.append(f"p_{parent_name}.{pc} AS auto_{parent_name}_{pc}")

    static_sql = f"SELECT {', '.join(select_parts)} FROM {entity_table} e {' '.join(joins)}"
    result = result.merge(con.sql(static_sql).df(), on=entity_col, how="left")

    # Extended anchor grid including parent FKs (e.g., state_id)
    parent_fk_cols = list(ent_tbl.fkey_col_to_pkey_table.keys())
    ext_cols = ", ".join([f"a.{entity_col}", f"a.{time_col}", "a.q_ts"] + [f"e.{fk}" for fk in parent_fk_cols])
    con.execute(
        f"""
        CREATE TEMP TABLE anchor_grid_ext AS
        SELECT {ext_cols}
        FROM anchor_grid a
        JOIN {entity_table} e ON e.{ent_tbl.pkey_col} = a.{entity_col}
        """
    )

    result_with_q = result.copy()
    result_with_q["q_ts"] = result_with_q[time_col].dt.to_period("Q").dt.to_timestamp()
    for fk in parent_fk_cols:
        if fk not in result_with_q.columns:
            result_with_q[fk] = result_with_q[f"auto_{entity_table}_{fk}"]

    # Helper: exact strictly-before-timestamp cumulative stats via quarterly buckets + window functions
    def _cum_temporal_agg(
        source_sql: str,
        group_key: str,
        source_time_col: str,
        num_cols: list[str],
        prefix: str,
        anchor_join_key: str,
    ) -> pd.DataFrame:
        bucket_aggs = ["COUNT(*) AS _cnt"]
        win_selects = [f"COALESCE(SUM(_cnt) OVER w, 0) AS {prefix}__count"]
        for col in num_cols:
            bucket_aggs.extend(
                [
                    f"SUM(COALESCE(CAST({col} AS DOUBLE), 0.0)) AS _sum_{col}",
                    f"COUNT({col}) AS _nn_{col}",
                    f"MAX(CAST({col} AS DOUBLE)) AS _max_{col}",
                ]
            )
            win_selects.extend(
                [
                    f"COALESCE(SUM(_sum_{col}) OVER w, 0.0) AS {prefix}__{col}_sum",
                    f"CASE WHEN COALESCE(SUM(_nn_{col}) OVER w, 0) > 0 "
                    f"THEN SUM(_sum_{col}) OVER w / SUM(_nn_{col}) OVER w ELSE NULL END AS {prefix}__{col}_mean",
                    f"MAX(_max_{col}) OVER w AS {prefix}__{col}_max",
                ]
            )

        sql = f"""
        WITH src AS ({source_sql}),
        buckets AS (
            SELECT
                {group_key} AS _gk,
                DATE_TRUNC('quarter', {source_time_col}) AS q_ts,
                {', '.join(bucket_aggs)}
            FROM src
            GROUP BY 1, 2
        ),
        grid AS (
            SELECT DISTINCT a.{anchor_join_key} AS _gk, a.q_ts
            FROM anchor_grid_ext a
        ),
        merged AS (
            SELECT g._gk, g.q_ts, b.* EXCLUDE (_gk, q_ts)
            FROM grid g
            LEFT JOIN buckets b ON b._gk = g._gk AND b.q_ts = g.q_ts
        )
        SELECT
            _gk AS {anchor_join_key},
            q_ts,
            {', '.join(win_selects)}
        FROM merged
        WINDOW w AS (PARTITION BY _gk ORDER BY q_ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        """
        return con.sql(sql).df()

    # 2. BFS backward along FK edges (child -> parent) starting from entity_table up to max_depth
    # Each queue entry: (table_name, depth, join_chain_from_table_to_entity)
    # join_chain is a list of (child_tbl, child_fk, parent_tbl, parent_pk) from current table to entity_table
    visited = {entity_table}
    queue: deque[tuple[str, int, list[tuple[str, str, str, str]]]] = deque([(entity_table, 0, [])])

    while queue:
        cur_table, depth, chain = queue.popleft()
        if depth >= max_depth:
            continue
        cur_pk = db.table_dict[cur_table].pkey_col
        if cur_pk is None:
            continue

        for child_name, child_tbl in db.table_dict.items():
            if child_name in visited:
                continue
            for fk_col, target in child_tbl.fkey_col_to_pkey_table.items():
                if target != cur_table:
                    continue
                visited.add(child_name)
                new_chain = [(child_name, fk_col, cur_table, cur_pk)] + chain
                queue.append((child_name, depth + 1, new_chain))

                # Build SQL joining child_name along new_chain to entity_col
                # Alias step 0 as t0 (child_name), step k as tk
                join_clauses = []
                for idx, (c_name, c_fk, p_name, p_pk) in enumerate(new_chain[:-1]):
                    join_clauses.append(f"JOIN {p_name} t{idx + 1} ON t{idx + 1}.{p_pk} = t{idx}.{c_fk}")
                last_idx = len(new_chain) - 1
                last_fk = new_chain[-1][1]
                entity_expr = f"t{last_idx}.{last_fk}"

                num_cols = [
                    c
                    for c in child_tbl.df.columns
                    if c != child_tbl.pkey_col
                    and c not in child_tbl.fkey_col_to_pkey_table
                    and pd.api.types.is_numeric_dtype(child_tbl.df[c])
                ]
                prefix = "auto_" + "_".join(step[0] for step in new_chain)

                if child_tbl.time_col is None:
                    # Static child table (e.g., places -> zones): aggregate static count
                    static_child_sql = f"""
                    SELECT {entity_expr} AS {entity_col}, COUNT(*) AS {prefix}__count
                    FROM {child_name} t0
                    {' '.join(join_clauses)}
                    GROUP BY 1
                    """
                    sc_df = con.sql(static_child_sql).df()
                    result_with_q = result_with_q.merge(sc_df, on=entity_col, how="left")
                    result_with_q[f"{prefix}__count"] = result_with_q[f"{prefix}__count"].fillna(0)
                else:
                    col_selects = ", ".join(
                        [f"{entity_expr} AS {entity_col}", f"t0.{child_tbl.time_col}"]
                        + [f"t0.{c}" for c in num_cols]
                    )
                    src_sql = f"SELECT {col_selects} FROM {child_name} t0 {' '.join(join_clauses)}"
                    feats = _cum_temporal_agg(
                        source_sql=src_sql,
                        group_key=entity_col,
                        source_time_col=child_tbl.time_col,
                        num_cols=num_cols,
                        prefix=prefix,
                        anchor_join_key=entity_col,
                    )
                    result_with_q = result_with_q.merge(feats, on=[entity_col, "q_ts"], how="left")
                break

    # 3. 2-hop sibling temporal tables pointing to entity's parent (e.g., episodes -> states <- zones)
    for parent_fk, parent_name in ent_tbl.fkey_col_to_pkey_table.items():
        for tname, tbl in db.table_dict.items():
            if tbl.time_col is None or tname in visited:
                continue
            for fk_col, target in tbl.fkey_col_to_pkey_table.items():
                if target == parent_name:
                    feats = _cum_temporal_agg(
                        source_sql=f"SELECT {fk_col} AS {parent_fk}, {tbl.time_col} FROM {tname}",
                        group_key=parent_fk,
                        source_time_col=tbl.time_col,
                        num_cols=[],
                        prefix=f"auto_{parent_name}_{tname}",
                        anchor_join_key=parent_fk,
                    )
                    result_with_q = result_with_q.merge(feats, on=[parent_fk, "q_ts"], how="left")

    con.close()
    drop_cols = ["q_ts"] + [fk for fk in parent_fk_cols if fk != entity_col]
    result_with_q = result_with_q.drop(columns=[c for c in drop_cols if c in result_with_q.columns])
    return all_task_rows[[entity_col, time_col]].merge(result_with_q, on=[entity_col, time_col], how="left")


def manual_dfs_features(
    db: relbench.base.Database,
    all_task_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Domain-specific multi-table join-aggregate features across all 8 tables, strictly before t.timestamp.

    Traverses:
    - zones -> states (parent geography & state-level seasonal activity)
    - zones <- episode_zones <- events (recency 1Q/4Q/12Q, same-quarter seasonality, log-damage, severity)
    - episode_zones -> episodes (episode multi-zone breadth & narrative length)
    - zones <- places <- event_locations (distinct recurring places hit, spatial footprint)
    - events <- fatalities (3-hop historical fatality count)
    """
    con = duckdb.connect()
    for tname, tbl in db.table_dict.items():
        con.register(tname, tbl.df)

    anchors = all_task_rows[["zone_id", "timestamp"]].drop_duplicates().sort_values(["zone_id", "timestamp"])
    con.register("task_anchors", anchors)

    sql = """
    WITH grid AS (
        SELECT
            a.zone_id,
            a.timestamp,
            DATE_TRUNC('quarter', a.timestamp) AS q_ts,
            EXTRACT(quarter FROM a.timestamp)::INT AS q_num,
            z.state_id
        FROM task_anchors a
        JOIN zones z ON z.zone_id = a.zone_id
    ),
    -- Zone quarterly event aggregates via episode_zones bridge
    ev_q AS (
        SELECT
            ez.zone_id,
            DATE_TRUNC('quarter', e.begin_date_time) AS q_ts,
            COUNT(*) AS n_events,
            SUM(CASE WHEN e.total_damage_usd > 0 THEN 1 ELSE 0 END) AS n_dmg_events,
            MAX(CASE WHEN e.total_damage_usd > 0 THEN 1 ELSE 0 END) AS had_dmg_quarter,
            SUM(LN(1.0 + COALESCE(e.total_damage_usd, 0.0))) AS sum_log_dmg,
            COUNT(DISTINCT ez.episode_id) AS n_episodes,
            COUNT(DISTINCT e.event_type) AS n_event_types,
            SUM(CASE WHEN e.event_type IN ('Thunderstorm Wind', 'Hail', 'Tornado', 'Flash Flood', 'Flood', 'High Wind', 'Winter Storm') THEN 1 ELSE 0 END) AS n_severe_events,
            SUM(e.injuries_direct + e.injuries_indirect) AS n_injuries,
            SUM(e.deaths_direct + e.deaths_indirect) AS n_deaths,
            MAX(e.begin_date_time) AS last_event_ts,
            MAX(CASE WHEN e.total_damage_usd > 0 THEN e.begin_date_time END) AS last_dmg_ts,
            AVG(LENGTH(COALESCE(e.event_narrative, ''))) AS avg_ev_narr_len
        FROM events e
        JOIN episode_zones ez ON ez.episode_zone_id = e.episode_zone_id
        GROUP BY 1, 2
    ),
    -- Episode multi-zone breadth & narrative length (strictly per quarter)
    ep_zone_counts AS (
        SELECT
            episode_id,
            DATE_TRUNC('quarter', begin_date_time) AS q_ts,
            COUNT(DISTINCT zone_id) AS ep_n_zones
        FROM episode_zones
        GROUP BY 1, 2
    ),
    ep_q AS (
        SELECT
            ez.zone_id,
            DATE_TRUNC('quarter', ez.begin_date_time) AS q_ts,
            AVG(ec.ep_n_zones) AS avg_ep_zones,
            AVG(LENGTH(COALESCE(ep.episode_narrative, ''))) AS avg_ep_narr_len
        FROM episode_zones ez
        JOIN episodes ep ON ep.episode_id = ez.episode_id
        JOIN ep_zone_counts ec
          ON ec.episode_id = ez.episode_id
         AND ec.q_ts = DATE_TRUNC('quarter', ez.begin_date_time)
        GROUP BY 1, 2
    ),
    -- Hyperlocal place activity via places <- event_locations
    place_q AS (
        SELECT
            p.zone_id,
            DATE_TRUNC('quarter', el.begin_date_time) AS q_ts,
            COUNT(DISTINCT el.place_id) AS n_distinct_places_hit,
            COUNT(*) AS n_loc_sightings
        FROM event_locations el
        JOIN places p ON p.place_id = el.place_id
        GROUP BY 1, 2
    ),
    -- State quarterly activity (via zones -> states)
    state_q AS (
        SELECT
            z.state_id,
            DATE_TRUNC('quarter', e.begin_date_time) AS q_ts,
            COUNT(*) AS st_n_events,
            SUM(CASE WHEN e.total_damage_usd > 0 THEN 1 ELSE 0 END) AS st_n_dmg_events,
            COUNT(DISTINCT ez.zone_id) AS st_active_zones,
            COUNT(DISTINCT CASE WHEN e.total_damage_usd > 0 THEN ez.zone_id END) AS st_dmg_zones
        FROM events e
        JOIN episode_zones ez ON ez.episode_zone_id = e.episode_zone_id
        JOIN zones z ON z.zone_id = ez.zone_id
        GROUP BY 1, 2
    ),
    all_q AS (
        SELECT DISTINCT q_ts FROM (
            SELECT q_ts FROM grid
            UNION
            SELECT q_ts FROM ev_q
        )
    ),
    zone_timeline AS (
        SELECT
            z.zone_id,
            z.state_id,
            q.q_ts,
            EXTRACT(quarter FROM q.q_ts)::INT AS q_num,
            COALESCE(ev.n_events, 0) AS n_events,
            COALESCE(ev.n_dmg_events, 0) AS n_dmg_events,
            COALESCE(ev.had_dmg_quarter, 0) AS had_dmg_quarter,
            COALESCE(ev.sum_log_dmg, 0.0) AS sum_log_dmg,
            COALESCE(ev.n_episodes, 0) AS n_episodes,
            COALESCE(ev.n_severe_events, 0) AS n_severe_events,
            ev.last_event_ts,
            ev.last_dmg_ts,
            ev.avg_ev_narr_len,
            ep.avg_ep_zones,
            ep.avg_ep_narr_len,
            COALESCE(pq.n_distinct_places_hit, 0) AS n_distinct_places_hit,
            COALESCE(pq.n_loc_sightings, 0) AS n_loc_sightings
        FROM (SELECT DISTINCT zone_id, state_id FROM grid) z
        CROSS JOIN all_q q
        LEFT JOIN ev_q ev ON ev.zone_id = z.zone_id AND ev.q_ts = q.q_ts
        LEFT JOIN ep_q ep ON ep.zone_id = z.zone_id AND ep.q_ts = q.q_ts
        LEFT JOIN place_q pq ON pq.zone_id = z.zone_id AND pq.q_ts = q.q_ts
    ),
    zone_win AS (
        SELECT
            zone_id,
            q_ts,
            -- Prior-quarter (last 90d)
            COALESCE(SUM(n_events) OVER w1, 0) AS man_z_events_1q,
            COALESCE(SUM(n_dmg_events) OVER w1, 0) AS man_z_dmg_events_1q,
            COALESCE(SUM(had_dmg_quarter) OVER w1, 0) AS man_z_had_dmg_1q,
            COALESCE(SUM(sum_log_dmg) OVER w1, 0.0) AS man_z_log_dmg_1q,
            COALESCE(SUM(n_distinct_places_hit) OVER w1, 0) AS man_z_places_hit_1q,
            -- Prior-year (last 4 quarters)
            COALESCE(SUM(n_events) OVER w4, 0) AS man_z_events_4q,
            COALESCE(SUM(n_dmg_events) OVER w4, 0) AS man_z_dmg_events_4q,
            COALESCE(AVG(had_dmg_quarter) OVER w4, 0.0) AS man_z_dmg_q_rate_4q,
            COALESCE(SUM(sum_log_dmg) OVER w4, 0.0) AS man_z_log_dmg_4q,
            COALESCE(SUM(n_severe_events) OVER w4, 0) AS man_z_severe_4q,
            COALESCE(SUM(n_episodes) OVER w4, 0) AS man_z_episodes_4q,
            COALESCE(SUM(n_distinct_places_hit) OVER w4, 0) AS man_z_places_hit_4q,
            COALESCE(SUM(n_loc_sightings) OVER w4, 0) AS man_z_loc_sightings_4q,
            -- Prior-3-years (last 12 quarters)
            COALESCE(AVG(had_dmg_quarter) OVER w12, 0.0) AS man_z_dmg_q_rate_12q,
            COALESCE(SUM(n_dmg_events) OVER w12, 0) AS man_z_dmg_events_12q,
            -- All-time prior
            COALESCE(AVG(had_dmg_quarter) OVER wall, 0.0) AS man_z_dmg_q_rate_all,
            CASE WHEN COALESCE(SUM(n_events) OVER wall, 0) > 0
                 THEN SUM(n_dmg_events) OVER wall * 1.0 / SUM(n_events) OVER wall
                 ELSE 0.0 END AS man_z_dmg_event_ratio_all,
            COALESCE(AVG(avg_ep_zones) OVER wall, 0.0) AS man_z_avg_ep_zones_all,
            COALESCE(AVG(avg_ev_narr_len) OVER wall, 0.0) AS man_z_avg_ev_narr_len_all,
            COALESCE(AVG(avg_ep_narr_len) OVER wall, 0.0) AS man_z_avg_ep_narr_len_all,
            -- Same-season (same calendar quarter in prior years)
            COALESCE(AVG(had_dmg_quarter) OVER wseason, 0.0) AS man_z_same_q_dmg_rate,
            COALESCE(SUM(n_dmg_events) OVER wseason, 0) AS man_z_same_q_dmg_events,
            COALESCE(SUM(n_events) OVER wseason, 0) AS man_z_same_q_events,
            -- Exact recency in days
            DATE_DIFF('day', MAX(last_event_ts) OVER wall, q_ts) AS man_z_days_since_event,
            DATE_DIFF('day', MAX(last_dmg_ts) OVER wall, q_ts) AS man_z_days_since_dmg
        FROM zone_timeline
        WINDOW
            w1 AS (PARTITION BY zone_id ORDER BY q_ts ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING),
            w4 AS (PARTITION BY zone_id ORDER BY q_ts ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING),
            w12 AS (PARTITION BY zone_id ORDER BY q_ts ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING),
            wall AS (PARTITION BY zone_id ORDER BY q_ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
            wseason AS (PARTITION BY zone_id, q_num ORDER BY q_ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ),
    state_timeline AS (
        SELECT
            s.state_id,
            q.q_ts,
            EXTRACT(quarter FROM q.q_ts)::INT AS q_num,
            COALESCE(sq.st_n_events, 0) AS st_n_events,
            COALESCE(sq.st_n_dmg_events, 0) AS st_n_dmg_events,
            COALESCE(sq.st_active_zones, 0) AS st_active_zones,
            COALESCE(sq.st_dmg_zones, 0) AS st_dmg_zones
        FROM (SELECT DISTINCT state_id FROM grid) s
        CROSS JOIN all_q q
        LEFT JOIN state_q sq ON sq.state_id = s.state_id AND sq.q_ts = q.q_ts
    ),
    state_win AS (
        SELECT
            state_id,
            q_ts,
            COALESCE(SUM(st_n_dmg_events) OVER w1, 0) AS man_st_dmg_events_1q,
            COALESCE(SUM(st_dmg_zones) OVER w1, 0) AS man_st_dmg_zones_1q,
            COALESCE(SUM(st_n_dmg_events) OVER w4, 0) AS man_st_dmg_events_4q,
            COALESCE(AVG(st_dmg_zones) OVER wseason, 0.0) AS man_st_same_q_avg_dmg_zones,
            CASE WHEN COALESCE(SUM(st_n_events) OVER wseason, 0) > 0
                 THEN SUM(st_n_dmg_events) OVER wseason * 1.0 / SUM(st_n_events) OVER wseason
                 ELSE 0.0 END AS man_st_same_q_dmg_ratio
        FROM state_timeline
        WINDOW
            w1 AS (PARTITION BY state_id ORDER BY q_ts ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING),
            w4 AS (PARTITION BY state_id ORDER BY q_ts ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING),
            wseason AS (PARTITION BY state_id, q_num ORDER BY q_ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    )
    SELECT
        g.zone_id,
        g.timestamp,
        EXTRACT(year FROM g.timestamp)::INT AS man_year,
        EXTRACT(quarter FROM g.timestamp)::INT AS man_quarter,
        zw.* EXCLUDE (zone_id, q_ts),
        sw.* EXCLUDE (state_id, q_ts)
    FROM grid g
    LEFT JOIN zone_win zw ON zw.zone_id = g.zone_id AND zw.q_ts = g.q_ts
    LEFT JOIN state_win sw ON sw.state_id = g.state_id AND sw.q_ts = g.q_ts
    """
    feats = con.sql(sql).df()
    con.close()
    return all_task_rows[["zone_id", "timestamp"]].merge(feats, on=["zone_id", "timestamp"], how="left")


def prepare_xgb_matrix(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return X


def main() -> None:
    print("=" * 75)
    print("EVALUATING BASELINES ON RELBENCH 'rel-noaa-storms' / 'zone-damage' TASK")
    print("=" * 75)

    dataset_dir = Path("output")
    dataset = relbench.load_dataset(dataset_dir)
    task = dataset.load_task("zone-damage")
    db = dataset.get_db()  # cut at test_timestamp

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    train_df = train_table.df
    val_df = val_table.df
    test_df = test_table.df

    y_train = train_df["has_damage"].to_numpy(dtype=int)
    y_val = val_df["has_damage"].to_numpy(dtype=int)
    y_test = test_df["has_damage"].to_numpy(dtype=int)

    print(
        f"Loaded Train: {len(train_df):,} ({y_train.mean()*100:.2f}% pos) | "
        f"Val: {len(val_df):,} ({y_val.mean()*100:.2f}% pos) | "
        f"Test: {len(test_df):,} ({y_test.mean()*100:.2f}% pos)"
    )

    results: dict[str, dict[str, float]] = {}

    def report(name: str, val_pred: np.ndarray, test_pred: np.ndarray) -> None:
        val_pred = np.asarray(val_pred, dtype=float)
        test_pred = np.asarray(test_pred, dtype=float)
        val_auc = task.evaluate(val_pred, val_table)["roc_auc"]
        test_auc = roc_auc_score(y_test, test_pred)
        results[name] = {"Val ROC-AUC": val_auc, "Test ROC-AUC": test_auc}
        print(f"  {name:42s} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

    # -----------------------------------------------------------------
    # BASELINE 1: Historical Majority & Historical Positive Rate
    # -----------------------------------------------------------------
    print("\n--- BASELINE 1: Historical Majority / Rate (Target Table Only) ---")
    global_rate = float(y_train.mean())
    global_maj = int(global_rate >= 0.5)

    report(
        "1a. Global Historical Majority",
        np.full(len(val_df), global_maj),
        np.full(len(test_df), global_maj),
    )

    zone_rates_train = train_df.groupby("zone_id")["has_damage"].mean()
    zone_maj_train = (zone_rates_train >= 0.5).astype(int)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    zone_rates_trainval = train_val_df.groupby("zone_id")["has_damage"].mean()
    zone_maj_trainval = (zone_rates_trainval >= 0.5).astype(int)

    report(
        "1b. Per-Zone Historical Majority",
        val_df["zone_id"].map(zone_maj_train).fillna(global_maj).to_numpy(),
        test_df["zone_id"].map(zone_maj_trainval).fillna(global_maj).to_numpy(),
    )

    report(
        "1c. Per-Zone Historical Rate",
        val_df["zone_id"].map(zone_rates_train).fillna(global_rate).to_numpy(),
        test_df["zone_id"].map(zone_rates_trainval).fillna(global_rate).to_numpy(),
    )

    # -----------------------------------------------------------------
    # BASELINE 2: XGBoost on Target Table Only (entity + timestamp)
    # -----------------------------------------------------------------
    print("\n--- BASELINE 2: XGBoost on Target Table Only ---")

    def target_table_features(df: pd.DataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(df["timestamp"])
        return pd.DataFrame(
            {
                "zone_id": df["zone_id"].astype(int),
                "year": ts.dt.year,
                "month": ts.dt.month,
                "quarter": ts.dt.quarter,
            },
            index=df.index,
        )

    model_b2 = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    model_b2.fit(target_table_features(train_df), y_train)
    report(
        "2. XGBoost (Target Table Only)",
        model_b2.predict_proba(target_table_features(val_df))[:, 1],
        model_b2.predict_proba(target_table_features(test_df))[:, 1],
    )

    # -----------------------------------------------------------------
    # BASELINE 3: Deep Feature Synthesis (Automatic + Manual Multi-Table)
    # -----------------------------------------------------------------
    print("\n--- BASELINE 3: Deep Feature Synthesis (Multi-Table Relational Features) ---")
    all_rows = pd.concat(
        [
            train_df[["zone_id", "timestamp"]],
            val_df[["zone_id", "timestamp"]],
            test_df[["zone_id", "timestamp"]],
        ],
        ignore_index=True,
    )

    print("  Synthesizing automatic multi-hop schema-driven DFS features...")
    auto_all = auto_dfs_features(db, all_rows)
    n_tr, n_va = len(train_df), len(val_df)
    auto_tr = auto_all.iloc[:n_tr].reset_index(drop=True)
    auto_va = auto_all.iloc[n_tr : n_tr + n_va].reset_index(drop=True)
    auto_te = auto_all.iloc[n_tr + n_va :].reset_index(drop=True)

    auto_feat_cols = [c for c in auto_all.columns if c not in ("zone_id", "timestamp")]
    auto_cat_cols = [
        c for c in auto_feat_cols if not pd.api.types.is_numeric_dtype(auto_all[c])
    ]
    print(f"  Automatic DFS generated {len(auto_feat_cols)} features across all {len(db.table_dict)} tables.")

    model_b3_auto = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    model_b3_auto.fit(prepare_xgb_matrix(auto_tr, auto_feat_cols, auto_cat_cols), y_train)
    report(
        "3a. XGBoost (Automatic DFS)",
        model_b3_auto.predict_proba(prepare_xgb_matrix(auto_va, auto_feat_cols, auto_cat_cols))[:, 1],
        model_b3_auto.predict_proba(prepare_xgb_matrix(auto_te, auto_feat_cols, auto_cat_cols))[:, 1],
    )

    print("  Synthesizing manual domain-specific multi-table features...")
    man_all = manual_dfs_features(db, all_rows)
    man_feat_cols = [c for c in man_all.columns if c not in ("zone_id", "timestamp")]
    print(f"  Manual DFS generated {len(man_feat_cols)} features.")

    comb_all = pd.concat([auto_all, man_all[man_feat_cols]], axis=1)
    comb_feat_cols = auto_feat_cols + man_feat_cols
    comb_tr = comb_all.iloc[:n_tr].reset_index(drop=True)
    comb_va = comb_all.iloc[n_tr : n_tr + n_va].reset_index(drop=True)
    comb_te = comb_all.iloc[n_tr + n_va :].reset_index(drop=True)

    model_b3_comb = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    model_b3_comb.fit(prepare_xgb_matrix(comb_tr, comb_feat_cols, auto_cat_cols), y_train)
    report(
        "3b. XGBoost (Manual + Auto DFS)",
        model_b3_comb.predict_proba(prepare_xgb_matrix(comb_va, comb_feat_cols, auto_cat_cols))[:, 1],
        model_b3_comb.predict_proba(prepare_xgb_matrix(comb_te, comb_feat_cols, auto_cat_cols))[:, 1],
    )

    importances = pd.Series(model_b3_comb.feature_importances_, index=comb_feat_cols).sort_values(ascending=False)
    print("\nTop 10 Most Important Features (3b):")
    for fname, imp in importances.head(10).items():
        print(f"  {fname:45s}: {imp:.4f}")

    print("\n" + "=" * 75)
    print("FINAL SUMMARY COMPARISON TABLE")
    print("=" * 75)
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
