"""
src/datamerge.py
----------------
Joins the four /data/ CSVs into a single data/merged_final.csv.
Run from backend/: python src/datamerge.py

Output schema: one row per (geo_place_name × time_period), wide format.
Each source contributes its columns; sources with no shared key are
appended as additional rows. To add a new source: write one _join_*
or _append_* function and call it in main().
"""

import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "merged_final.csv"

FINAL_COLS = [
    "borough", "geo_place_name", "zip_code", "time_period",
    "pm25", "no2", "ozone", "aqi", "purpleair_pm25", "truck_vmt",
    "asthma_er_rate", "cardiovascular_hosp_rate",
    "respiratory_hosp_rate", "pm25_deaths",
    # Issue #12 — neighborhood ED visit rates from Epiquery / SPARCS
    "asthma_ed_rate", "cardiovascular_ed_rate",
    "respiratory_ed_rate", "population",
]

# (name column value, measure column value) → output column name.
# To add a new indicator: one new entry here, nothing else to change.
INDICATOR_MAP: dict[tuple[str, str], str] = {
    ("Fine particles (PM 2.5)",                              "Annual mean"):                           "pm25",
    ("Nitrogen dioxide (NO2)",                               "Annual mean"):                           "no2",
    ("Ozone (O3)",                                           "Summer mean"):                           "ozone",
    ("Traffic density (trucks)",                             "Million miles"):                         "truck_vmt",
    ("Deaths due to PM2.5",                                  "Estimated annual rate (age 30+)"):       "pm25_deaths",
    ("Cardiovascular hospitalizations due to PM2.5 (age 40+)", "Estimated annual rate"):              "cardiovascular_hosp_rate",
    ("Respiratory hospitalizations due to PM2.5 (age 20+)", "Estimated annual rate"):                 "respiratory_hosp_rate",
    ("Asthma emergency department visits due to PM2.5",      "Estimated annual rate (under age 18)"): "asthma_er_rate",
}

# geo_join_id hundreds-digit → borough for each geo type.
# UHF42/UHF34 and CD use opposite conventions in DOHMH open data.
_UHF_PREFIX = {"1": "Bronx", "2": "Brooklyn", "3": "Manhattan", "4": "Queens", "5": "Staten Island"}
_CD_PREFIX  = {"1": "Manhattan", "2": "Bronx",  "3": "Brooklyn", "4": "Queens", "5": "Staten Island"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names; replace whitespace/hyphens/dots/parens with _."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(r"[\s\-\.\(\)]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def _load(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [skip] {label}: file not found")
        return None
    df = _normalize_cols(pd.read_csv(path))
    print(f"  [ok]   {label}: {len(df):,} rows, {len(df.columns)} cols")
    return df


def _infer_borough(geo_type: str, geo_place_name: str, geo_join_id) -> str | None:
    """
    Return borough name for a row.
    UHF42/UHF34 and CD use different hundreds-digit borough encodings.
    """
    if geo_type == "Borough":
        return geo_place_name
    if geo_type == "Citywide":
        return "Citywide"
    with contextlib.suppress(ValueError, TypeError):
        prefix = str(int(float(geo_join_id)))[0]
        if geo_type in {"UHF42", "UHF34"}:
            return _UHF_PREFIX.get(prefix)
        if geo_type == "CD":
            return _CD_PREFIX.get(prefix)
    return None


# ── source transforms ─────────────────────────────────────────────────────────

def _pivot_aq_health(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot air_quality_health from long → wide format.
    Filters to the 8 indicators in INDICATOR_MAP; all other rows are dropped.
    Output: one row per (geo × time_period), each indicator is a column.
    """
    df = df.copy()
    df["_col"] = df.apply(
        lambda r: INDICATOR_MAP.get((r["name"], r["measure"])), axis=1
    )
    df = df[df["_col"].notna()]

    # UHF42 and UHF34 share geo_join_ids and often the same geo_place_name.
    # UHF42 is authoritative (it has truck VMT data; UHF34 doesn't). Sort by
    # priority then deduplicate so each (geo × time_period × indicator) has
    # exactly one source row before the pivot.
    _GEO_PRIORITY = {"UHF42": 0, "UHF34": 1, "CD": 2, "Borough": 3, "Citywide": 4}
    df["_geo_sort"] = df["geo_type_name"].map(_GEO_PRIORITY).fillna(99)
    df = (
        df.sort_values("_geo_sort")
        .drop_duplicates(
            subset=["geo_join_id", "geo_place_name", "time_period", "_col"],
            keep="first",
        )
        .drop(columns=["_geo_sort"])
    )

    wide = (
        df.pivot_table(
            index=["geo_join_id", "geo_place_name", "geo_type_name", "time_period"],
            columns="_col",
            values="data_value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None

    wide["borough"] = wide.apply(
        lambda r: _infer_borough(r["geo_type_name"], r["geo_place_name"], r["geo_join_id"]),
        axis=1,
    )
    return wide


def _join_asthma(merged: pd.DataFrame, asthma: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-join asthma_ed_pm25 on geo_join_id.

    asthma_ed_pm25 has no time_period column — its 5 rows per neighborhood
    are different years without labels, so we average them as a fallback.
    The main pivot already carries asthma_er_rate (with time info) from
    air_quality_health; this only fills gaps for geo areas not yet covered.
    """
    agg = (
        asthma.groupby("geo_join_id")["data_value"]
        .mean()
        .reset_index()
        .rename(columns={"data_value": "_asthma_supp"})
    )
    merged = merged.merge(agg, on="geo_join_id", how="left")

    if "asthma_er_rate" not in merged.columns:
        merged["asthma_er_rate"] = merged["_asthma_supp"]
    else:
        merged["asthma_er_rate"] = merged["asthma_er_rate"].fillna(merged["_asthma_supp"])

    return merged.drop(columns=["_asthma_supp"])


def _append_airnow(merged: pd.DataFrame, airnow: pd.DataFrame) -> pd.DataFrame:
    """
    AirNow returns one row per (parameter × zip × date).
    Pivot to one row per (zip × date) with max AQI across parameters,
    then append as new rows (no shared geo key with UHF42 neighborhoods).
    """
    group_keys = [c for c in ("query_zip", "dateobserved") if c in airnow.columns]
    if not group_keys or "aqi" not in airnow.columns:
        print("  [warn] AirNow: expected columns missing, skipping")
        return merged

    agg_spec: dict = {"aqi": ("aqi", "max")}
    if "reportingarea" in airnow.columns:
        agg_spec["geo_place_name"] = ("reportingarea", "first")

    an = (
        airnow.groupby(group_keys)
        .agg(**agg_spec)
        .reset_index()
        .rename(columns={"query_zip": "zip_code", "dateobserved": "time_period"})
    )
    return pd.concat([merged, an], ignore_index=True, sort=False)


def _append_purpleair(merged: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    """
    Each PurpleAir row is one outdoor sensor.
    After _normalize_cols, 'pm2.5' → 'pm2_5'.
    Appends sensor name + PM2.5 reading as new rows.
    """
    pm_col = next(
        (c for c in ("pm2_5", "pm2.5") if c in pa.columns),
        next((c for c in pa.columns if c.startswith("pm") and "minute" not in c), None),
    )
    if pm_col is None:
        print("  [warn] PurpleAir: PM2.5 column not found, skipping")
        return merged

    rows = pa[["name", pm_col]].rename(
        columns={"name": "geo_place_name", pm_col: "purpleair_pm25"}
    )
    return pd.concat([merged, rows], ignore_index=True, sort=False)


# ── ED visit join (Issue #12) ─────────────────────────────────────────────────

def _join_er_visits(merged: pd.DataFrame, er: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join neighborhood ED visit rates onto merged on (geo_place_name, year).

    time_period formats handled:
      "2019"      → year 2019
      "2017-2019" → year 2019 (end year of the range)

    A left join preserves every pollution row; ED columns are NaN where
    there is no ED data for that neighborhood/year combination.
    """
    er = er.copy()
    er["geo_place_name"] = er["neighborhood_name"].str.strip()
    er["year"] = pd.to_numeric(er["year"], errors="coerce").astype("Int64")

    rate_cols = [
        c for c in ("asthma_ed_rate", "cardiovascular_ed_rate",
                    "respiratory_ed_rate", "population")
        if c in er.columns
    ]
    # Deduplicate: if the same (neighborhood, year) appears more than once, average
    er_slim = (
        er[["geo_place_name", "year"] + rate_cols]
        .groupby(["geo_place_name", "year"])[rate_cols]
        .mean()
        .reset_index()
    )

    # Extract end year from time_period so ranges map to a single integer
    merged = merged.copy()
    merged["_year"] = (
        merged["time_period"]
        .str.split("-")
        .str[-1]
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )

    n_before = len(merged)
    merged = merged.merge(
        er_slim,
        left_on=["geo_place_name", "_year"],
        right_on=["geo_place_name", "year"],
        how="left",
    )
    # Drop the helper columns introduced by the join
    merged = merged.drop(columns=["_year", "year"], errors="ignore")

    # ── Join quality report ───────────────────────────────────────────────────
    n_matched = (
        merged["asthma_ed_rate"].notna().sum()
        if "asthma_ed_rate" in merged.columns else 0
    )
    pct = n_matched / n_before * 100 if n_before else 0
    n_unmatched = n_before - n_matched
    print(f"  Rows matched  : {n_matched:,}")
    print(f"  Rows unmatched: {n_unmatched:,}")
    print(f"  Match rate    : {pct:.1f}%")
    if pct < 70:
        print("  [warn] Match rate below 70% — ED data may not span all years/neighborhoods")

    return merged


# ── summary ───────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame) -> None:
    print(f"\n  Rows   : {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print("\n  Null % per column:")
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        bar = "#" * int(pct / 5)
        print(f"    {col:<32} {pct:5.1f}%  {bar}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    print("Data merge\n")

    aq_health = _load(DATA_DIR / "air_quality_health.csv",      "Air Quality & Health")
    asthma_ed = _load(DATA_DIR / "asthma_ed_pm25.csv",          "Asthma ED PM2.5")
    airnow    = _load(DATA_DIR / "airnow_aqi.csv",              "AirNow AQI")
    purpleair = _load(DATA_DIR / "purpleair_pm25.csv",          "PurpleAir PM2.5")
    er_visits = _load(DATA_DIR / "er_visits_neighborhood.csv",  "ED visits (neighborhood)")

    if aq_health is None:
        print("\n[error] air_quality_health.csv is required — run src/dataingestion.py first")
        sys.exit(1)

    print("\nPivoting air quality indicators to wide format...")
    merged = _pivot_aq_health(aq_health)
    print(f"  Shape after pivot: {merged.shape}")

    if asthma_ed is not None:
        merged = _join_asthma(merged, asthma_ed)

    if airnow is not None:
        print("Appending AirNow observations...")
        merged = _append_airnow(merged, airnow)

    if purpleair is not None:
        print("Appending PurpleAir sensors...")
        merged = _append_purpleair(merged, purpleair)

    if er_visits is not None:
        print("Joining ED visit rates (neighborhood × year)...")
        merged = _join_er_visits(merged, er_visits)

    # guarantee every target column exists (NaN where source had no data)
    for col in FINAL_COLS:
        if col not in merged.columns:
            merged[col] = np.nan

    # UHF34 and UHF42 share geo_join_ids, so some neighborhood-year pairs appear
    # in both geo types with identical values. Drop these after FINAL_COLS removes
    # geo_type_name.
    merged = merged[FINAL_COLS].drop_duplicates()
    merged.to_csv(OUT_PATH, index=False)

    print(f"\nSaved -> {OUT_PATH}")
    _print_summary(merged)
    print()


if __name__ == "__main__":
    main()
