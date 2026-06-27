"""
src/dataingestion.py
--------------------
Pulls raw air quality data from four sources and saves each CSV to /data/.
Run from backend/: python src/dataingestion.py
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

AIRNOW_KEY    = os.getenv("AIRNOW_API_KEY", "")
PURPLEAIR_KEY = os.getenv("PURPLEAIR_API_KEY", "")
DATA_DIR      = Path("data")

# One representative ZIP per borough — gives AirNow coverage across the city
NYC_ZIPS = ["10001", "10451", "11201", "11354", "10301"]

# NYC bounding box used by PurpleAir (nw-corner → se-corner)
NYC_BBOX = {
    "nwlng": -74.259, "nwlat": 40.918,
    "selng": -73.700, "selat": 40.477,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(df: pd.DataFrame, path: Path, label: str) -> None:
    df.to_csv(path, index=False)
    print(f"  [ok]   {label}: {len(df)} rows -> {path}")


# ── fetchers ──────────────────────────────────────────────────────────────────

def fetch_nyc_open_data(dataset_id: str, out_path: Path, label: str, limit: int = 5000) -> None:
    """Socrata REST endpoint — no API key required."""
    url = f"https://data.cityofnewyork.us/resource/{dataset_id}.json"
    try:
        r = requests.get(url, params={"$limit": limit}, timeout=30)
        r.raise_for_status()
        _save(pd.DataFrame(r.json()), out_path, label)
    except Exception as exc:
        print(f"  [err]  {label}: {exc}")


def fetch_airnow(out_path: Path) -> None:
    """Current AQI observations for five representative NYC ZIP codes."""
    if not AIRNOW_KEY:
        print("  [skip] AirNow: AIRNOW_API_KEY not set")
        return

    url  = "https://www.airnowapi.org/aq/observation/zipCode/current/"
    rows: list[dict] = []

    for zip_code in NYC_ZIPS:
        try:
            r = requests.get(url, params={
                "format":   "application/json",
                "zipCode":  zip_code,
                "distance": 25,
                "API_KEY":  AIRNOW_KEY,
            }, timeout=15)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                for obs in data:
                    obs["query_zip"] = zip_code
                rows.extend(data)
        except Exception as exc:
            print(f"    [warn] AirNow zip {zip_code}: {exc}")

    if not rows:
        print("  [err]  AirNow: no data returned — check key or try again later")
        return

    df = pd.DataFrame(rows)

    # Flatten the nested Category dict ({"Number": 2, "Name": "Moderate"})
    if "Category" in df.columns:
        cat_cols = df["Category"].apply(pd.Series).add_prefix("category_")
        df = pd.concat([df.drop(columns=["Category"]), cat_cols], axis=1)

    _save(df.drop_duplicates(), out_path, "AirNow")


def fetch_purpleair(out_path: Path) -> None:
    """Outdoor PM2.5 sensors inside the NYC bounding box — PurpleAir API v1."""
    if not PURPLEAIR_KEY:
        print("  [skip] PurpleAir: PURPLEAIR_API_KEY not set")
        return

    fields = (
        "name,latitude,longitude,"
        "pm2.5,pm2.5_10minute,pm2.5_30minute,pm2.5_60minute,"
        "humidity,temperature,last_seen"
    )

    try:
        r = requests.get(
            "https://api.purpleair.com/v1/sensors",
            headers={"X-API-Key": PURPLEAIR_KEY},
            params={"fields": fields, "location_type": 0, **NYC_BBOX},
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        # The response 'fields' list includes sensor_index as the first column
        df = pd.DataFrame(payload["data"], columns=payload["fields"])
        _save(df, out_path, "PurpleAir")
    except Exception as exc:
        print(f"  [err]  PurpleAir: {exc}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Data ingestion — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    fetch_nyc_open_data(
        "c3uy-2p5r",
        DATA_DIR / "air_quality_health.csv",
        "NYC Air Quality & Health",
        limit=25_000,  # dataset has ~19,827 rows; 25k ensures we get all of them
    )
    fetch_nyc_open_data(
        "ebe7-6eah",
        DATA_DIR / "asthma_ed_pm25.csv",
        "NYC Asthma ED PM2.5",
    )
    fetch_airnow(DATA_DIR / "airnow_aqi.csv")
    fetch_purpleair(DATA_DIR / "purpleair_pm25.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
