"""
datamerge.py
Reads the 4 CSVs produced by dataingestion.py, enriches them with
Census poverty/income data and asthma ED visit rates, then saves
one clean analysis-ready file: Data/merged_final.csv

Run this after dataingestion.py:
    python src/datamerge.py
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "Data")
ENV_PATH  = os.path.join(BASE_DIR, "..", ".env")
load_dotenv(dotenv_path=ENV_PATH)

NYC_BOROUGHS = ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load CSVs
# ══════════════════════════════════════════════════════════════════════════════
def load_csv(filename, label):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  ⚠ {label} not found — skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  ✓ Loaded {label}: {len(df)} rows, {df.shape[1]} columns")
    return df


def load_all():
    print("\n[1/6] Loading CSVs...")
    df_air    = load_csv("nyc_air_quality_health.csv", "NYC Air Quality")
    if df_air.empty:
        df_air = load_csv("Air_Quality_and_Health_Impacts.csv", "NYC Air Quality (fallback)")
    df_asthma = load_csv("asthma_ed_pm25.csv",      "Asthma ED Visits")
    df_airnow = load_csv("airnow_realtime_aqi.csv", "AirNow AQI")
    df_purple = load_csv("purpleair_pm25.csv",      "PurpleAir Sensors")
    return df_air, df_asthma, df_airnow, df_purple


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Clean each source
# ══════════════════════════════════════════════════════════════════════════════
def clean_air_quality(df):
    print("\n[2/6] Cleaning NYC Air Quality data...")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "geo_type_name" in df.columns:
        df = df[df["geo_type_name"] == "Borough"].copy()
    if "geo_place_name" in df.columns:
        df = df[df["geo_place_name"].isin(NYC_BOROUGHS)].copy()

    df["data_value"] = pd.to_numeric(df.get("data_value", pd.Series()), errors="coerce")
    df["start_date"] = pd.to_datetime(df.get("start_date", pd.Series()), errors="coerce")
    df["year"]       = df["start_date"].dt.year
    df               = df.dropna(subset=["data_value"])
    df               = df.rename(columns={"geo_place_name": "borough"})

    print(f"  ✓ {len(df)} rows after cleaning")
    return df


def clean_asthma(df):
    print("\n[3/6] Cleaning Asthma ED data...")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["source"] = "asthma_ed_pm25"
    print(f"  ✓ {len(df)} rows")
    return df


def clean_realtime(df_airnow, df_purple):
    print("\n[4/6] Cleaning real-time sensor data...")

    if not df_airnow.empty:
        df_airnow.columns = [c.lower() for c in df_airnow.columns]
        df_airnow["source"] = "airnow"
        print(f"  ✓ AirNow: {len(df_airnow)} rows")

    if not df_purple.empty:
        df_purple.columns = [c.lower() for c in df_purple.columns]
        df_purple = df_purple[df_purple["borough"].isin(NYC_BOROUGHS)].copy()
        df_purple["source"] = "purpleair"
        print(f"  ✓ PurpleAir: {len(df_purple)} rows")

    return df_airnow, df_purple


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Pull Census poverty + income data and join by zip code
# ══════════════════════════════════════════════════════════════════════════════
def pull_census():
    print("\n[5/6] Pulling Census poverty & income data...")
    api_key = os.getenv("CENSUS_API_KEY")
    if not api_key:
        print("  ⚠ CENSUS_API_KEY not found in .env — skipping")
        return pd.DataFrame()

    url = (
        "https://api.census.gov/data/2022/acs/acs5"
        "?get=NAME,B17001_001E,B17001_002E,B19013_001E"
        "&for=zip%20code%20tabulation%20area:*"
        f"&key={api_key}"
    )
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"  ⚠ Census API returned {r.status_code} — skipping")
            return pd.DataFrame()

        data = r.json()
        poverty_df = pd.DataFrame(data[1:], columns=data[0])
        poverty_df = poverty_df.rename(columns={
            "B17001_001E": "total_pop",
            "B17001_002E": "pop_below_poverty",
            "B19013_001E": "median_income",
            "zip code tabulation area": "zip_code"
        })
        poverty_df["poverty_rate"] = (
            poverty_df["pop_below_poverty"].astype(float) /
            poverty_df["total_pop"].astype(float) * 100
        ).round(1)

        # Filter to NYC zip codes only
        poverty_df = poverty_df[
            poverty_df["zip_code"].str.startswith((
                "100", "101", "102", "103", "104",  # Manhattan + Bronx
                "110", "111", "112", "113", "114",  # Queens + Brooklyn
                "116",                               # Far Rockaway
                "117",                               # Staten Island
            ))
        ]
        poverty_df["zip_code"] = poverty_df["zip_code"].astype(str)
        poverty_df["median_income"] = pd.to_numeric(poverty_df["median_income"], errors="coerce")

        print(f"  ✓ Census data: {len(poverty_df)} NYC zip codes")
        return poverty_df[["zip_code", "poverty_rate", "median_income"]]

    except Exception as e:
        print(f"  ⚠ Census API error: {e} — skipping")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Pull asthma ED rates and join by borough
# ══════════════════════════════════════════════════════════════════════════════
def pull_asthma_ed_rates():
    print("\n[6/6] Pulling asthma ED visit rates by borough...")

    def uhf_to_borough(geo_id):
        prefix = str(geo_id)[0]
        mapping = {
            "1": "Bronx",
            "2": "Brooklyn",
            "3": "Manhattan",
            "4": "Queens",
            "5": "Staten Island"
        }
        return mapping.get(prefix, "Unknown")

    try:
        r = requests.get(
            "https://data.cityofnewyork.us/resource/ebe7-6eah.json",
            params={"$limit": 5000},
            timeout=30
        )
        if r.status_code != 200:
            print(f"  ⚠ Asthma ED API returned {r.status_code} — skipping")
            return pd.DataFrame()

        asthma_df = pd.DataFrame(r.json())
        asthma_df["borough"] = asthma_df["geo_join_id"].astype(str).apply(uhf_to_borough)
        asthma_df["data_value"] = pd.to_numeric(asthma_df["data_value"], errors="coerce")

        asthma_borough = (
            asthma_df.groupby("borough")["data_value"]
            .mean()
            .reset_index()
            .rename(columns={"data_value": "asthma_ed_rate_under18"})
        )
        print(f"  ✓ Asthma ED rates pulled for {len(asthma_borough)} boroughs")
        print(asthma_borough.to_string(index=False))
        return asthma_borough

    except Exception as e:
        print(f"  ⚠ Asthma ED API error: {e} — skipping")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Combine, enrich, and save
# ══════════════════════════════════════════════════════════════════════════════
def save_all(df_air, df_asthma, df_airnow, df_purple, poverty_df, asthma_rates):
    print("\n[Saving] Building merged_final.csv...")

    # Stack all 4 source CSVs
    combined = pd.concat(
        [df for df in [df_air, df_asthma, df_airnow, df_purple] if not df.empty],
        ignore_index=True, sort=False
    )

    # ── Join Census poverty data by zip_code ──────────────────────────────────
    if not poverty_df.empty:
        combined["zip_code"] = combined["zip_code"].astype(str).str.replace(".0", "", regex=False)
        combined = combined.merge(poverty_df, on="zip_code", how="left")
        filled = combined["poverty_rate"].notna().sum()
        print(f"  ✓ poverty_rate & median_income joined — {filled} rows filled")
    else:
        print("  ⚠ poverty_df empty — skipping Census join")

    # ── Join asthma ED rates by borough ───────────────────────────────────────
    if not asthma_rates.empty:
        combined = combined.merge(asthma_rates, on="borough", how="left")
        filled = combined["asthma_ed_rate_under18"].notna().sum()
        print(f"  ✓ asthma_ed_rate_under18 joined — {filled} rows filled")
    else:
        print("  ⚠ asthma_rates empty — skipping asthma ED join")

    # ── Save individual source CSVs ───────────────────────────────────────────
    df_air.to_csv(os.path.join(DATA_DIR, "merged_air_quality.csv"), index=False)
    if not df_asthma.empty:
        df_asthma.to_csv(os.path.join(DATA_DIR, "merged_asthma_ed.csv"), index=False)
    if not df_airnow.empty:
        df_airnow.to_csv(os.path.join(DATA_DIR, "merged_airnow.csv"), index=False)
    if not df_purple.empty:
        df_purple.to_csv(os.path.join(DATA_DIR, "merged_purpleair.csv"), index=False)

    # ── Save combined file ────────────────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, "merged_final.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n  ✓ merged_final.csv: {combined.shape[0]} rows x {combined.shape[1]} columns")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  NYC Pollution & Disease — Data Merge")
    print("=" * 60)

    df_air, df_asthma, df_airnow, df_purple = load_all()

    if df_air.empty:
        print("\n✗ Cannot merge — no air quality data found.")
        exit(1)

    df_air               = clean_air_quality(df_air)
    df_asthma            = clean_asthma(df_asthma)
    df_airnow, df_purple = clean_realtime(df_airnow, df_purple)
    poverty_df           = pull_census()
    asthma_rates         = pull_asthma_ed_rates()
    combined             = save_all(df_air, df_asthma, df_airnow, df_purple,
                                    poverty_df, asthma_rates)

    print("\n" + "=" * 60)
    print("  MERGED DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total rows      : {combined.shape[0]}")
    print(f"  Total columns   : {combined.shape[1]}")
    print(f"  poverty_rate    : {combined['poverty_rate'].notna().sum()} rows filled")
    print(f"  median_income   : {combined['median_income'].notna().sum()} rows filled")
    print(f"  asthma_ed_rate  : {combined['asthma_ed_rate_under18'].notna().sum()} rows filled")
    if "source" in combined.columns:
        print(f"  Sources         : {combined['source'].value_counts().to_dict()}")
    print("=" * 60)
