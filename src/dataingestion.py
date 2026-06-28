"""
dataingestion.py
Pulls data from 4 sources and saves each as a CSV in the Data/ folder:

  1. NYC Open Data    - Air Quality & Health Impacts (c3uy-2p5r)
                       NO key required
  2. NYC Open Data    - PM2.5 Attributable Asthma ED Visits (ebe7-6eah)
                       NO key required
  3. EPA AirNow API  - Real-time AQI by NYC zip code
                       Requires AIRNOW_API_KEY in .env
  4. PurpleAir API   - Community sensor network filling Brooklyn
                       and Manhattan monitoring gaps
                       Requires PURPLEAIR_API_KEY in .env

Run this script FIRST before opening the notebook:
    python src/dataingestion.py

Your .env file must contain:
    AIRNOW_API_KEY=your_airnow_key
    PURPLEAIR_API_KEY=your_purpleair_read_key
"""

import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv()
AIRNOW_KEY    = os.getenv("AIRNOW_API_KEY")
PURPLEAIR_KEY = os.getenv("PURPLEAIR_API_KEY")

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
os.makedirs(DATA_DIR, exist_ok=True)

NYC_BOROUGHS = ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]

TARGET_INDICATORS = [
    "Fine particles (PM 2.5)",
    "Nitrogen dioxide (NO2)",
    "Ozone (O3)",
    "Asthma emergency department visits due to PM2.5",
    "Asthma emergency departments visits due to Ozone",
    "Asthma hospitalizations due to Ozone",
    "Cardiovascular hospitalizations due to PM2.5 (age 40+)",
    "Respiratory hospitalizations due to PM2.5 (age 20+)",
    "Cardiac and respiratory deaths due to Ozone",
    "Deaths due to PM2.5",
    "Annual vehicle miles traveled",
    "Annual vehicle miles traveled (trucks)",
]


# ── Source 1: NYC Open Data — Air Quality & Health Impacts ─────────────────────
def fetch_nyc_air_quality():
    """
    Pulls borough-level air quality and health impact indicators
    from NYC Open Data dataset c3uy-2p5r.
    Covers PM2.5, NO2, Ozone, asthma ER visits, cardiovascular
    hospitalizations, respiratory hospitalizations, deaths, and
    truck traffic — all in one dataset, 2005 to present.
    """
    print("\n[1/4] Fetching NYC Air Quality & Health Impacts (c3uy-2p5r)...")
    url = "https://data.cityofnewyork.us/resource/c3uy-2p5r.json"
    params = {
        "$limit":  50000,
        "$where":  "geo_type_name='Borough'",
        "$order":  "start_date DESC",
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("error"):
            raise ValueError(f"API error: {data.get('message')}")

        df = pd.DataFrame(data)
        df = df[df["name"].isin(TARGET_INDICATORS)].copy()
        df = df[df["geo_place_name"].isin(NYC_BOROUGHS)].copy()
        df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

        out_path = os.path.join(DATA_DIR, "nyc_air_quality_health.csv")
        df.to_csv(out_path, index=False)
        print(f"    ✓ Saved {len(df)} rows → nyc_air_quality_health.csv")
        return df

    except Exception as e:
        print(f"    ✗ NYC Open Data fetch failed: {e}")
        print("      Falling back to local CSV...")
        fallback = os.path.join(DATA_DIR, "Air_Quality_and_Health_Impacts.csv")
        if os.path.exists(fallback):
            df = pd.read_csv(fallback)
            df = df[df["Geo Type Name"] == "Borough"].copy()
            df = df[df["Geo Place Name"].isin(NYC_BOROUGHS)].copy()
            print(f"    ✓ Loaded {len(df)} rows from local fallback CSV")
            return df
        raise


# ── Source 2: NYC Open Data — PM2.5 Attributable Asthma ED Visits ─────────────
def fetch_asthma_ed_visits():
    """
    Pulls PM2.5-attributable asthma emergency department visit data
    from NYC Open Data dataset ebe7-6eah.
    Provides neighborhood-level asthma ED visit rates specifically
    linked to PM2.5 exposure — key dependent variable for analysis.
    """
    print("\n[2/4] Fetching PM2.5 Attributable Asthma ED Visits (ebe7-6eah)...")
    url = "https://data.cityofnewyork.us/resource/ebe7-6eah.json"
    params = {"$limit": 50000}

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("error"):
            raise ValueError(f"API error: {data.get('message')}")

        df = pd.DataFrame(data)
        out_path = os.path.join(DATA_DIR, "asthma_ed_pm25.csv")
        df.to_csv(out_path, index=False)
        print(f"    ✓ Saved {len(df)} rows → asthma_ed_pm25.csv")
        return df

    except Exception as e:
        print(f"    ✗ Asthma ED fetch failed: {e}")
        return pd.DataFrame()


# ── Source 3: EPA AirNow API — Real-Time AQI by NYC Zip Code ──────────────────
def fetch_airnow_aqi():
    """
    Pulls current AQI readings from the EPA AirNow API for NYC zip codes.
    Uses borough-specific search radii to keep data geographically honest:
    - Manhattan:     5 miles  (narrow, 13 miles long x 2 miles wide)
    - Bronx:         8 miles  (compact borough)
    - Brooklyn:      10 miles (wider geography)
    - Queens:        12 miles (largest borough by area)
    - Staten Island: 10 miles (isolated, needs slightly wider net)
    """
    print("\n[3/4] Fetching AirNow real-time AQI (EPA AirNow API)...")

    if not AIRNOW_KEY:
        print("    ✗ AIRNOW_API_KEY not found in .env — skipping")
        return pd.DataFrame()

    borough_distances = {
        "Manhattan":     5,
        "Bronx":         8,
        "Brooklyn":      10,
        "Queens":        12,
        "Staten Island": 10,
    }

    all_nyc_zips = [
        # MANHATTAN
        ("10001", "Manhattan", "Chelsea — Lincoln Tunnel, Javits trucks"),
        ("10002", "Manhattan", "Lower East Side — FDR Drive"),
        ("10003", "Manhattan", "East Village — FDR Drive"),
        ("10004", "Manhattan", "Battery Park — Brooklyn Battery Tunnel"),
        ("10007", "Manhattan", "City Hall — Brooklyn Bridge trucks"),
        ("10009", "Manhattan", "Alphabet City — FDR Drive"),
        ("10011", "Manhattan", "Chelsea waterfront — West Side Hwy"),
        ("10013", "Manhattan", "Tribeca — Holland Tunnel approach"),
        ("10014", "Manhattan", "West Village — West Side Hwy"),
        ("10018", "Manhattan", "Hell's Kitchen — Lincoln Tunnel"),
        ("10019", "Manhattan", "Midtown West — Lincoln Tunnel, Javits"),
        ("10026", "Manhattan", "Harlem — FDR Drive, truck corridor"),
        ("10029", "Manhattan", "East Harlem — FDR Drive"),
        ("10031", "Manhattan", "Hamilton Heights — West Side Hwy"),
        ("10032", "Manhattan", "Washington Heights — GWB approach"),
        ("10034", "Manhattan", "Inwood — GWB, I-95 approach"),
        ("10035", "Manhattan", "East Harlem — RFK Bridge, FDR"),
        ("10036", "Manhattan", "Hell's Kitchen — Lincoln Tunnel, Javits"),
        ("10040", "Manhattan", "Inwood — GWB approach"),
        ("10044", "Manhattan", "Roosevelt Island — low traffic comparison"),
        ("10280", "Manhattan", "Battery Park City — West Side Hwy"),
        ("10282", "Manhattan", "Battery Park City — West Side Hwy, tunnel"),
        # BRONX
        ("10451", "Bronx", "Mott Haven — Cross Bronx Expwy I-278"),
        ("10452", "Bronx", "Highbridge — Major Deegan I-87"),
        ("10453", "Bronx", "Morris Heights — Cross Bronx Expwy"),
        ("10454", "Bronx", "Port Morris — Bruckner Expwy, waterway industrial"),
        ("10455", "Bronx", "Longwood — Bruckner Expressway I-278"),
        ("10456", "Bronx", "Morrisania — Major Deegan I-87"),
        ("10457", "Bronx", "East Tremont — Cross Bronx Expwy"),
        ("10458", "Bronx", "Belmont — Bronx River Pkwy"),
        ("10459", "Bronx", "Longwood — truck routes"),
        ("10460", "Bronx", "West Farms — I-895 Sheridan Expwy"),
        ("10461", "Bronx", "Parkchester — Bruckner Expwy"),
        ("10462", "Bronx", "Westchester Sq — Bruckner Expwy"),
        ("10463", "Bronx", "Kingsbridge — Henry Hudson Pkwy"),
        ("10464", "Bronx", "City Island — coastal, low traffic comparison"),
        ("10465", "Bronx", "Throgs Neck — Throgs Neck Bridge"),
        ("10468", "Bronx", "University Heights — Cross Bronx Expwy"),
        ("10471", "Bronx", "Riverdale — Henry Hudson Pkwy, lower density"),
        ("10472", "Bronx", "Soundview — Bruckner Expwy"),
        ("10473", "Bronx", "Soundview — Bruckner Expwy, industrial"),
        ("10474", "Bronx", "Hunts Point — major truck terminal, food hub"),
        ("10475", "Bronx", "Co-op City — I-95 New England Thruway"),
        # BROOKLYN
        ("11201", "Brooklyn", "Brooklyn Heights — Brooklyn Bridge, BQE"),
        ("11205", "Brooklyn", "Clinton Hill — BQE I-278"),
        ("11206", "Brooklyn", "Williamsburg — BQE, truck routes"),
        ("11207", "Brooklyn", "East New York — I-278, industrial trucks"),
        ("11208", "Brooklyn", "East New York — I-278, Spring Creek"),
        ("11209", "Brooklyn", "Bay Ridge — Verrazzano Bridge approach"),
        ("11211", "Brooklyn", "Williamsburg — BQE, industrial"),
        ("11212", "Brooklyn", "Brownsville — I-278, highest asthma rates"),
        ("11215", "Brooklyn", "Park Slope — Gowanus Superfund site"),
        ("11217", "Brooklyn", "Boerum Hill — Gowanus Canal Superfund"),
        ("11218", "Brooklyn", "Kensington — Prospect Expwy, BQE"),
        ("11219", "Brooklyn", "Borough Park — Prospect Expwy, Home Depot trucks"),
        ("11220", "Brooklyn", "Sunset Park — Gowanus Expwy, port industrial"),
        ("11221", "Brooklyn", "Bushwick — Broadway Junction, radioactive site"),
        ("11222", "Brooklyn", "Greenpoint — Newtown Creek Superfund"),
        ("11223", "Brooklyn", "Gravesend — Belt Pkwy"),
        ("11224", "Brooklyn", "Coney Island — Belt Pkwy"),
        ("11231", "Brooklyn", "Red Hook — BQE, Gowanus Expwy, port"),
        ("11232", "Brooklyn", "Sunset Park — Gowanus Expwy, industrial"),
        ("11233", "Brooklyn", "Broadway Junction — radioactive remediation"),
        ("11235", "Brooklyn", "Sheepshead Bay — Belt Pkwy"),
        ("11237", "Brooklyn", "Bushwick — BQE, truck routes"),
        ("11238", "Brooklyn", "Prospect Heights — BQE"),
        # QUEENS
        ("11101", "Queens", "LIC — Queens Midtown Tunnel, I-495"),
        ("11102", "Queens", "Astoria — RFK Bridge approach"),
        ("11106", "Queens", "Astoria — RFK/Triborough Bridge"),
        ("11354", "Queens", "Flushing — bus depot, truck routes"),
        ("11355", "Queens", "Flushing — heavy commercial traffic"),
        ("11359", "Queens", "Bayside — low traffic, comparison area"),
        ("11362", "Queens", "Little Neck — low traffic, comparison"),
        ("11368", "Queens", "Corona — I-678, LaGuardia trucks"),
        ("11369", "Queens", "East Elmhurst — Grand Central Pkwy, LaGuardia"),
        ("11378", "Queens", "Maspeth — Newtown Creek, industrial trucks"),
        ("11385", "Queens", "Ridgewood — Jackie Robinson Pkwy, BQE"),
        ("11413", "Queens", "Springfield Gardens — I-678, JFK trucks"),
        ("11416", "Queens", "Ozone Park — Van Wyck Expwy"),
        ("11420", "Queens", "South Ozone Park — Van Wyck, JFK"),
        ("11430", "Queens", "JFK Airport — max truck & aviation traffic"),
        ("11434", "Queens", "Jamaica — Van Wyck I-678, JFK trucks"),
        ("11691", "Queens", "Far Rockaway — coastal, low traffic"),
        ("11697", "Queens", "Breezy Point — coastal, lowest traffic"),
        # STATEN ISLAND
        ("10301", "Staten Island", "St. George — Ferry, Bayonne Bridge approach"),
        ("10302", "Staten Island", "Port Richmond — Goethals Bridge, industrial"),
        ("10303", "Staten Island", "Mariners Harbor — Bayonne Bridge, trucks"),
        ("10304", "Staten Island", "Stapleton — I-278 expressway"),
        ("10305", "Staten Island", "Rosebank — Verrazzano Bridge approach"),
        ("10307", "Staten Island", "Tottenville — low traffic, comparison area"),
        ("10308", "Staten Island", "Great Kills — low traffic"),
        ("10310", "Staten Island", "West Brighton — I-278 truck corridor"),
        ("10312", "Staten Island", "Eltingville — low traffic, comparison area"),
        ("10314", "Staten Island", "Heartland Village — I-278 expressway"),
    ]

    today   = datetime.now().strftime("%Y-%m-%dT%H")
    records = []
    total   = len(all_nyc_zips)

    for i, (zipcode, borough, notes) in enumerate(all_nyc_zips, 1):
        distance = borough_distances.get(borough, 10)
        url = "https://www.airnowapi.org/aq/observation/zipCode/current/"
        params = {
            "format":   "application/json",
            "zipCode":  zipcode,
            "distance": distance,
            "API_KEY":  AIRNOW_KEY,
        }
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data:
                for obs in data:
                    obs["borough"]       = borough
                    obs["zip_code"]      = zipcode
                    obs["location_note"] = notes
                    obs["search_radius"] = distance
                    obs["pulled_at"]     = today
                    records.append(obs)
                print(f"    [{i}/{total}] ✓ {zipcode} — {borough} "
                      f"({distance}mi) — {len(data)} reading(s)")
            else:
                print(f"    [{i}/{total}] ⚠ {zipcode} — no monitor "
                      f"within {distance}mi")
        except Exception as e:
            print(f"    [{i}/{total}] ✗ {zipcode} failed: {e}")

    if records:
        df = pd.DataFrame(records)
        if "AQI" in df.columns:
            df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
        out_path = os.path.join(DATA_DIR, "airnow_realtime_aqi.csv")
        df.to_csv(out_path, index=False)
        print(f"\n    ✓ Saved {len(df)} rows → airnow_realtime_aqi.csv")
        return df

    print("    ⚠ No AirNow data retrieved")
    return pd.DataFrame()


# ── Source 4: PurpleAir API — Community Sensors Filling Borough Gaps ──────────
def fetch_purpleair():
    """
    Pulls real-time PM2.5 readings from PurpleAir community sensors
    across all 5 NYC boroughs.

    PurpleAir fills the critical monitoring gaps that AirNow misses:
    - Brooklyn: Gowanus, Newtown Creek, Red Hook, Brownsville
    - Manhattan: Lower East Side, Midtown, Harlem, Washington Heights
    - Queens:  Flushing, Jamaica, Astoria neighborhoods
    - Bronx:   Additional neighborhood-level coverage
    - Staten Island: Supplemental readings

    Uses a bounding box covering all of NYC:
      NW corner: -74.26 lng, 40.92 lat
      SE corner: -73.70 lng, 40.48 lat

    Data updates every 2 minutes vs AirNow's hourly updates.
    Sensors are community-owned and street-level — capturing
    hyperlocal pollution near highways, warehouses, and truck routes.
    """
    print("\n[4/4] Fetching PurpleAir community sensor data...")

    if not PURPLEAIR_KEY:
        print("    ✗ PURPLEAIR_API_KEY not found in .env — skipping")
        print("      Add PURPLEAIR_API_KEY=your_read_key to your .env file")
        return pd.DataFrame()

    url = "https://api.purpleair.com/v1/sensors"
    params = {
        # NYC bounding box
        "nwlng": -74.26,   # northwest longitude
        "nwlat":  40.92,   # northwest latitude
        "selng": -73.70,   # southeast longitude
        "selat":  40.48,   # southeast latitude
        # Fields we want back
        "fields": ",".join([
            "name",
            "location_type",
            "latitude",
            "longitude",
            "pm2.5",
            "pm2.5_10minute",
            "pm2.5_30minute",
            "pm2.5_60minute",
            "pm2.5_24hour",
            "humidity",
            "temperature",
            "pressure",
            "last_seen",
        ]),
    }
    headers = {"X-API-Key": PURPLEAIR_KEY}

    try:
        response = requests.get(url, params=params,
                                headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "data" not in data:
            raise ValueError(f"Unexpected response: {data}")

        # Build DataFrame from fields + data arrays
        fields = data["fields"]
        rows   = data["data"]
        df     = pd.DataFrame(rows, columns=fields)

        # Clean up
        df["pm2.5"]         = pd.to_numeric(df.get("pm2.5"),         errors="coerce")
        df["pm2.5_24hour"]  = pd.to_numeric(df.get("pm2.5_24hour"),  errors="coerce")
        df["latitude"]      = pd.to_numeric(df.get("latitude"),       errors="coerce")
        df["longitude"]     = pd.to_numeric(df.get("longitude"),      errors="coerce")
        df["pulled_at"]     = datetime.now().strftime("%Y-%m-%dT%H:%M")

        # Assign borough based on coordinates
        def assign_borough(lat, lng):
            if lat is None or lng is None:
                return "Unknown"
            if lng > -74.05 and lat > 40.70 and lat < 40.88 and lng < -73.90:
                if lat > 40.80:
                    return "Manhattan"
                return "Manhattan"
            if lat > 40.78 and lng < -73.90:
                return "Bronx"
            if lng < -73.95 and lat < 40.74:
                return "Brooklyn"
            if lng > -73.95 and lat < 40.76:
                return "Queens"
            if lng < -74.05:
                return "Staten Island"
            return "NYC"

        df["borough"] = df.apply(
            lambda r: assign_borough(r.get("latitude"), r.get("longitude")),
            axis=1
        )

        # Filter out outdoor sensors only (location_type 0 = outside)
        if "location_type" in df.columns:
            df = df[df["location_type"] == 0].copy()

        # Drop sensors with no recent PM2.5 reading
        df = df.dropna(subset=["pm2.5"]).copy()

        # Filter to reasonable PM2.5 range (remove faulty sensors)
        df = df[(df["pm2.5"] >= 0) & (df["pm2.5"] <= 500)].copy()

        out_path = os.path.join(DATA_DIR, "purpleair_pm25.csv")
        df.to_csv(out_path, index=False)

        print(f"    ✓ Saved {len(df)} outdoor sensors → purpleair_pm25.csv")
        print(f"\n    Coverage by borough:")
        for boro, count in df["borough"].value_counts().items():
            print(f"      {boro:<15} {count} sensors")

        return df

    except Exception as e:
        print(f"    ✗ PurpleAir fetch failed: {e}")
        return pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  NYC Pollution & Disease — Data Ingestion")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n  Sources:")
    print("  1. NYC Open Data c3uy-2p5r  — Air Quality & Health Impacts")
    print("  2. NYC Open Data ebe7-6eah  — PM2.5 Asthma ED Visits")
    print("  3. EPA AirNow API           — Real-time AQI by zip code")
    print("  4. PurpleAir API            — Community sensors all 5 boroughs")
    print("=" * 60)

    df_air      = fetch_nyc_air_quality()
    df_asthma   = fetch_asthma_ed_visits()
    df_airnow   = fetch_airnow_aqi()
    df_purple   = fetch_purpleair()

    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Source 1 — NYC Air Quality rows  : {len(df_air)}")
    print(f"  Source 2 — Asthma ED rows        : {len(df_asthma)}")
    print(f"  Source 3 — AirNow rows           : {len(df_airnow)}")
    print(f"  Source 4 — PurpleAir sensors     : {len(df_purple)}")
    print("\n  Files saved to Data/ folder:")
    print("    nyc_air_quality_health.csv")
    print("    asthma_ed_pm25.csv")
    print("    airnow_realtime_aqi.csv")
    print("    purpleair_pm25.csv")
    print("=" * 60)