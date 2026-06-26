"""
datamerge.py
Reads the 4 CSVs produced by dataingestion.py and merges them into
one clean analysis-ready file: Data/merged_final.csv

Run this after dataingestion.py:
    python src/datamerge.py
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
NYC_BOROUGHS = ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]


# ── Helper ─────────────────────────────────────────────────────────────────────
def load_csv(filename, label):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  ⚠ {label} not found — skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  ✓ Loaded {label}: {len(df)} rows, {df.shape[1]} columns")
    return df


# ── Step 1: Load all CSVs ──────────────────────────────────────────────────────
def load_all():
    print("\n[1/5] Loading CSVs...")
    df_air     = load_csv("nyc_air_quality_health.csv", "NYC Air Quality")
    if df_air.empty:
        df_air = load_csv("Air_Quality_and_Health_Impacts.csv", "NYC Air Quality (fallback)")
    df_asthma  = load_csv("asthma_ed_pm25.csv",         "Asthma ED Visits")
    df_airnow  = load_csv("airnow_realtime_aqi.csv",    "AirNow AQI")
    df_purple  = load_csv("purpleair_pm25.csv",         "PurpleAir Sensors")
    return df_air, df_asthma, df_airnow, df_purple


# ── Step 2: Clean air quality data ────────────────────────────────────────────
def clean_air_quality(df):
    print("\n[2/5] Cleaning NYC Air Quality data...")

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Handle both API and downloaded CSV column names
    col_map = {
        "geo_type_name":  "geo_type_name",
        "geo_place_name": "geo_place_name",
        "name":           "name",
        "data_value":     "data_value",
        "time_period":    "time_period",
        "start_date":     "start_date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Filter to 5 boroughs
    if "geo_type_name" in df.columns:
        df = df[df["geo_type_name"] == "Borough"].copy()
    if "geo_place_name" in df.columns:
        df = df[df["geo_place_name"].isin(NYC_BOROUGHS)].copy()

    df["data_value"] = pd.to_numeric(df.get("data_value", pd.Series()), errors="coerce")
    df["start_date"] = pd.to_datetime(df.get("start_date", pd.Series()), errors="coerce")
    df["year"]       = df["start_date"].dt.year
    df               = df.dropna(subset=["data_value"])

    print(f"  ✓ {len(df)} rows after cleaning")
    return df


# ── Step 3: Pivot air quality to wide format ───────────────────────────────────
def pivot_air_quality(df):
    print("\n[3/5] Pivoting to wide format (one row per borough per time period)...")

    if "name" not in df.columns or "geo_place_name" not in df.columns:
        print("  ⚠ Cannot pivot — missing required columns")
        return df

    pivot = df.pivot_table(
        index=["geo_place_name", "time_period", "year"],
        columns="name",
        values="data_value",
        aggfunc="mean"
    ).reset_index()

    pivot.columns.name = None
    pivot = pivot.rename(columns={"geo_place_name": "borough"})

    # Clean column names for Python use
    pivot.columns = [
        c.lower()
         .replace(" ", "_")
         .replace("(", "")
         .replace(")", "")
         .replace(".", "")
         .replace(",", "")
         .replace("+", "plus")
        for c in pivot.columns
    ]

    print(f"  ✓ Wide format: {pivot.shape[0]} rows x {pivot.shape[1]} columns")
    return pivot


# ── Step 4: Add real-time data from AirNow and PurpleAir ──────────────────────
def add_realtime(df_wide, df_airnow, df_purple):
    print("\n[4/5] Adding real-time sensor data...")

    merged = df_wide.copy()

    # ── AirNow — average AQI per borough ──────────────────────────────────────
    if not df_airnow.empty:
        df_airnow.columns = [c.lower() for c in df_airnow.columns]
        aqi_col = next((c for c in df_airnow.columns
                        if "aqi" in c.lower()), None)
        if aqi_col and "borough" in df_airnow.columns:
            airnow_avg = (
                df_airnow.groupby("borough")[aqi_col]
                .mean()
                .reset_index()
                .rename(columns={aqi_col: "airnow_avg_aqi"})
            )
            merged = merged.merge(airnow_avg, on="borough", how="left")
            print(f"  ✓ Added AirNow average AQI per borough")

    # ── PurpleAir — average PM2.5 per borough ─────────────────────────────────
    if not df_purple.empty:
        df_purple.columns = [c.lower() for c in df_purple.columns]
        pm_col = next((c for c in df_purple.columns
                       if "pm2.5" in c.lower() or "pm25" in c.lower()), None)
        if pm_col and "borough" in df_purple.columns:
            purple_avg = (
                df_purple[df_purple["borough"].isin(NYC_BOROUGHS)]
                .groupby("borough")[pm_col]
                .mean()
                .reset_index()
                .rename(columns={pm_col: "purpleair_avg_pm25"})
            )
            merged = merged.merge(purple_avg, on="borough", how="left")
            print(f"  ✓ Added PurpleAir average PM2.5 per borough")

    return merged


# ── Step 5: Save merged file ───────────────────────────────────────────────────
def save_merged(df):
    print("\n[5/5] Saving merged_final.csv...")

    df["borough"] = df["borough"].astype("category")

    out_path = os.path.join(DATA_DIR, "merged_final.csv")
    df.to_csv(out_path, index=False)

    print(f"  ✓ Saved: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  ✓ Path: {out_path}")
    return df


# ── Summary ────────────────────────────────────────────────────────────────────
def print_summary(df):
    print("\n" + "=" * 60)
    print("  MERGED DATASET SUMMARY")
    print("=" * 60)
    print(f"  Shape         : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Boroughs      : {sorted(df['borough'].unique().tolist())}")
    yr = df.get("year", pd.Series())
    if not yr.dropna().empty:
        print(f"  Years covered : {int(yr.min())} - {int(yr.max())}")
    print(f"\n  Columns ({df.shape[1]} total):")
    for col in df.columns:
        nulls = df[col].isna().sum()
        pct   = round(nulls / len(df) * 100, 1)
        print(f"    {col:<55} nulls: {nulls} ({pct}%)")
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  NYC Pollution & Disease — Data Merge")
    print("=" * 60)

    df_air, df_asthma, df_airnow, df_purple = load_all()

    if df_air.empty:
        print("\n✗ Cannot merge — no air quality data found.")
        print("  Run dataingestion.py first.")
        exit(1)

    df_clean  = clean_air_quality(df_air)
    df_wide   = pivot_air_quality(df_clean)
    df_merged = add_realtime(df_wide, df_airnow, df_purple)
    df_final  = save_merged(df_merged)

    print_summary(df_final)
