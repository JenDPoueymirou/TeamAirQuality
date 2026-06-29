"""
dataingestion.py
Pulls data from 3 sources and saves each as a CSV in the Data/ folder:
  1. NYC Open Data  - Air Quality & Health Impacts (c3uy-2p5r)
  2. NYC Open Data  - DOHMH Environment & Health indicators (dde9-2xgj)
  3. AirNow API     - Real-time AQI for the 5 NYC boroughs

Run this script first before opening the notebook.
Your AirNow API key must be in a .env file in the project root:
    AIRNOW_API_KEY=your_key_here
"""

import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv()
AIRNOW_KEY = os.getenv("AIRNOW_API_KEY")

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
os.makedirs(DATA_DIR, exist_ok=True)

NYC_BOROUGHS = ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]

# Indicators we care about (all live inside c3uy-2p5r)
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
    Pulls all health + pollution indicators from NYC Open Data c3uy-2p5r.
    Filters to Borough-level geography only (the 5 boroughs).
    Saves to Data/nyc_air_quality_health.csv
    """
    print("\n[1/3] Fetching NYC Air Quality & Health Impacts...")

    url = "https://data.cityofnewyork.us/resource/c3uy-2p5r.json"
    params = {
        "$limit": 50000,
        "$where": "geo_type_name='Borough'",   # borough-level rows only
        "$order": "start_date DESC",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Check for API error response
        if isinstance(data, dict) and data.get("error"):
            raise ValueError(f"API error: {data.get('message')}")

        df = pd.DataFrame(data)

        # Filter to only our target indicators
        df = df[df["name"].isin(TARGET_INDICATORS)].copy()

        # Filter to 5 boroughs explicitly
        df = df[df["geo_place_name"].isin(NYC_BOROUGHS)].copy()

        # Clean up data types
        df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

        out_path = os.path.join(DATA_DIR, "nyc_air_quality_health.csv")
        df.to_csv(out_path, index=False)
        print(f"    ✓ Saved {len(df)} rows → {out_path}")
        return df

    except Exception as e:
        print(f"    ✗ NYC Open Data fetch failed: {e}")
        print("      Falling back to local CSV if available...")
        fallback = os.path.join(DATA_DIR, "Air_Quality_and_Health_Impacts.csv")
        if os.path.exists(fallback):
            df = pd.read_csv(fallback)
            df = df[df["Geo Type Name"] == "Borough"].copy()
            df = df[df["Geo Place Name"].isin(NYC_BOROUGHS)].copy()
            print(f"    ✓ Loaded {len(df)} rows from fallback CSV")
            return df
        raise


# ── Source 2: NYC DOHMH — Environment & Health Data Portal ────────────────────
def fetch_dohmh_indicators():
    """
    Pulls neighborhood-level environmental health indicators from DOHMH.
    Dataset: dde9-2xgj — includes asthma, poverty, housing by UHF neighborhood.
    Saves to Data/dohmh_env_health.csv
    """
    print("\n[2/3] Fetching DOHMH Environment & Health indicators...")

    url = "https://data.cityofnewyork.us/resource/dde9-2xgj.json"
    params = {
        "$limit": 50000,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("error"):
            raise ValueError(f"API error: {data.get('message')}")

        df = pd.DataFrame(data)
        out_path = os.path.join(DATA_DIR, "dohmh_env_health.csv")
        df.to_csv(out_path, index=False)
        print(f"    ✓ Saved {len(df)} rows → {out_path}")
        return df

    except Exception as e:
        print(f"    ✗ DOHMH fetch failed: {e}")
        return pd.DataFrame()  # return empty df, non-fatal


# ── Source 3: AirNow API — Real-time AQI for ALL NYC Zip Codes ────────────────
def fetch_airnow_aqi():
    """
    Pulls current AQI readings from AirNow for ALL NYC zip codes.
    Full coverage lets us discover unexpected pollution hotspots beyond
    known highway corridors — e.g. Gowanus Superfund site, Newtown Creek,
    Broadway Junction radioactive remediation area, bus depots, big-box
    retail truck delivery zones, and waterway-adjacent industrial areas.

    Zip codes are annotated with borough and known environmental context
    so the heatmap can surface correlations we didn't anticipate.

    Saves to Data/airnow_realtime_aqi.csv
    """
    print("\n[3/3] Fetching AirNow real-time AQI for ALL NYC zip codes...")

    if not AIRNOW_KEY:
        print("    ✗ AIRNOW_API_KEY not found in .env — skipping")
        return pd.DataFrame()

    # ── All NYC zip codes with borough and environmental context ───────────────
    # Format: (zipcode, borough, notes)
    all_nyc_zips = [
        # MANHATTAN
        ("10001", "Manhattan", "Chelsea — Lincoln Tunnel, Javits trucks"),
        ("10002", "Manhattan", "Lower East Side — FDR Drive"),
        ("10003", "Manhattan", "East Village — FDR Drive"),
        ("10004", "Manhattan", "Battery Park — Brooklyn Battery Tunnel"),
        ("10005", "Manhattan", "Financial District — tunnel approaches"),
        ("10006", "Manhattan", "Financial District — tunnel approaches"),
        ("10007", "Manhattan", "City Hall — Brooklyn Bridge trucks"),
        ("10009", "Manhattan", "Alphabet City — FDR Drive"),
        ("10010", "Manhattan", "Gramercy — truck routes"),
        ("10011", "Manhattan", "Chelsea waterfront — West Side Hwy"),
        ("10012", "Manhattan", "SoHo — truck deliveries"),
        ("10013", "Manhattan", "Tribeca — Holland Tunnel approach"),
        ("10014", "Manhattan", "West Village — West Side Hwy trucks"),
        ("10016", "Manhattan", "Murray Hill — Midtown truck routes"),
        ("10017", "Manhattan", "Midtown East — high traffic density"),
        ("10018", "Manhattan", "Hell's Kitchen — Lincoln Tunnel approach"),
        ("10019", "Manhattan", "Midtown West — Lincoln Tunnel, Javits"),
        ("10020", "Manhattan", "Rockefeller Center — dense traffic"),
        ("10021", "Manhattan", "Upper East Side — FDR Drive"),
        ("10022", "Manhattan", "Midtown East — Queens Midtown Tunnel"),
        ("10023", "Manhattan", "Upper West Side — West Side Hwy"),
        ("10024", "Manhattan", "Upper West Side — West Side Hwy"),
        ("10025", "Manhattan", "Morningside Heights — West Side Hwy"),
        ("10026", "Manhattan", "Harlem — FDR Drive, truck corridor"),
        ("10027", "Manhattan", "Harlem — major truck routes"),
        ("10028", "Manhattan", "Upper East Side — FDR Drive"),
        ("10029", "Manhattan", "East Harlem — FDR Drive"),
        ("10030", "Manhattan", "Harlem — truck routes"),
        ("10031", "Manhattan", "Hamilton Heights — West Side Hwy"),
        ("10032", "Manhattan", "Washington Heights — GWB approach"),
        ("10033", "Manhattan", "Washington Heights — GWB trucks"),
        ("10034", "Manhattan", "Inwood — GWB, I-95 approach"),
        ("10035", "Manhattan", "East Harlem — RFK Bridge, FDR"),
        ("10036", "Manhattan", "Hell's Kitchen — Lincoln Tunnel, Javits"),
        ("10037", "Manhattan", "Harlem — elevated highway"),
        ("10038", "Manhattan", "Fulton — Brooklyn Bridge trucks"),
        ("10039", "Manhattan", "Harlem — truck routes"),
        ("10040", "Manhattan", "Inwood — GWB approach, I-95"),
        ("10044", "Manhattan", "Roosevelt Island"),
        ("10065", "Manhattan", "Upper East Side — FDR Drive"),
        ("10069", "Manhattan", "Upper West Side"),
        ("10075", "Manhattan", "Upper East Side — FDR Drive"),
        ("10128", "Manhattan", "Yorkville — FDR Drive"),
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
        ("10464", "Bronx", "City Island — coastal"),
        ("10465", "Bronx", "Throgs Neck — Throgs Neck Bridge"),
        ("10466", "Bronx", "Wakefield — I-87 New England Thruway"),
        ("10467", "Bronx", "Norwood — Mosholu Pkwy"),
        ("10468", "Bronx", "University Heights — Cross Bronx Expwy"),
        ("10469", "Bronx", "Pelham Gardens — Hutchinson River Pkwy"),
        ("10470", "Bronx", "Bronxwood — I-87"),
        ("10471", "Bronx", "Riverdale — Henry Hudson Pkwy"),
        ("10472", "Bronx", "Soundview — Bruckner Expwy"),
        ("10473", "Bronx", "Soundview — Bruckner Expwy, industrial"),
        ("10474", "Bronx", "Hunts Point — major truck terminal, food hub"),
        ("10475", "Bronx", "Co-op City — I-95 New England Thruway"),

        # BROOKLYN
        ("11201", "Brooklyn", "Brooklyn Heights — Brooklyn Bridge, BQE"),
        ("11203", "Brooklyn", "East Flatbush — truck routes"),
        ("11204", "Brooklyn", "Borough Park — truck routes"),
        ("11205", "Brooklyn", "Clinton Hill — BQE I-278"),
        ("11206", "Brooklyn", "Williamsburg — BQE, truck routes"),
        ("11207", "Brooklyn", "East New York — I-278, industrial trucks"),
        ("11208", "Brooklyn", "East New York — I-278, Spring Creek"),
        ("11209", "Brooklyn", "Bay Ridge — Verrazzano Bridge approach"),
        ("11210", "Brooklyn", "Flatbush — truck routes"),
        ("11211", "Brooklyn", "Williamsburg — BQE, industrial"),
        ("11212", "Brooklyn", "Brownsville — I-278, high asthma rates"),
        ("11213", "Brooklyn", "Crown Heights — truck routes"),
        ("11214", "Brooklyn", "Bensonhurst — Belt Pkwy"),
        ("11215", "Brooklyn", "Park Slope — Gowanus Superfund site"),
        ("11216", "Brooklyn", "Bedford-Stuyvesant — truck routes"),
        ("11217", "Brooklyn", "Boerum Hill — Gowanus Canal Superfund"),
        ("11218", "Brooklyn", "Kensington — Prospect Expwy, BQE"),
        ("11219", "Brooklyn", "Borough Park — Prospect Expwy trucks"),
        ("11220", "Brooklyn", "Sunset Park — Gowanus Expwy, port industrial"),
        ("11221", "Brooklyn", "Bushwick — Broadway Junction, radioactive site"),
        ("11222", "Brooklyn", "Greenpoint — Newtown Creek Superfund"),
        ("11223", "Brooklyn", "Gravesend — Belt Pkwy"),
        ("11224", "Brooklyn", "Coney Island — Belt Pkwy"),
        ("11225", "Brooklyn", "Flatbush — Prospect Expwy"),
        ("11226", "Brooklyn", "Flatbush — Prospect Expwy, truck routes"),
        ("11228", "Brooklyn", "Dyker Heights — Verrazzano approach"),
        ("11229", "Brooklyn", "Marine Park — Belt Pkwy"),
        ("11230", "Brooklyn", "Midwood — truck routes"),
        ("11231", "Brooklyn", "Red Hook — BQE, Gowanus Expwy, port"),
        ("11232", "Brooklyn", "Sunset Park — Gowanus Expwy, industrial"),
        ("11233", "Brooklyn", "Broadway Junction — radioactive remediation"),
        ("11234", "Brooklyn", "Mill Basin — Belt Pkwy"),
        ("11235", "Brooklyn", "Sheepshead Bay — Belt Pkwy"),
        ("11236", "Brooklyn", "Canarsie — Belt Pkwy, industrial"),
        ("11237", "Brooklyn", "Bushwick — BQE, truck routes"),
        ("11238", "Brooklyn", "Prospect Heights — BQE"),
        ("11239", "Brooklyn", "Georgetown — Belt Pkwy, industrial"),

        # QUEENS
        ("11101", "Queens", "Long Island City — Queens Midtown Tunnel, I-495"),
        ("11102", "Queens", "Astoria — RFK Bridge approach"),
        ("11103", "Queens", "Astoria — truck routes"),
        ("11104", "Queens", "Sunnyside — I-495 LIE"),
        ("11105", "Queens", "Astoria — Grand Central Pkwy"),
        ("11106", "Queens", "Astoria — RFK/Triborough Bridge"),
        ("11201", "Queens", "note: Brooklyn border area"),
        ("11354", "Queens", "Flushing — bus depot, truck routes"),
        ("11355", "Queens", "Flushing — heavy commercial traffic"),
        ("11356", "Queens", "College Point — industrial waterfront"),
        ("11357", "Queens", "Whitestone — Whitestone Bridge"),
        ("11358", "Queens", "Flushing — commercial trucks"),
        ("11359", "Queens", "Bayside — low traffic, comparison area"),
        ("11360", "Queens", "Bayside — low traffic, comparison area"),
        ("11361", "Queens", "Bayside — low traffic"),
        ("11362", "Queens", "Little Neck — low traffic, comparison"),
        ("11363", "Queens", "Little Neck — low traffic"),
        ("11364", "Queens", "Oakland Gardens — low traffic"),
        ("11365", "Queens", "Fresh Meadows — I-495 LIE"),
        ("11366", "Queens", "Fresh Meadows — I-495 LIE"),
        ("11367", "Queens", "Kew Gardens Hills — I-495 LIE"),
        ("11368", "Queens", "Corona — I-678, LaGuardia trucks"),
        ("11369", "Queens", "East Elmhurst — Grand Central Pkwy, LaGuardia"),
        ("11370", "Queens", "East Elmhurst — LaGuardia flight path"),
        ("11371", "Queens", "Flushing — LaGuardia Airport trucks"),
        ("11372", "Queens", "Jackson Heights — truck routes"),
        ("11373", "Queens", "Elmhurst — I-495 LIE, commercial"),
        ("11374", "Queens", "Rego Park — I-495 LIE"),
        ("11375", "Queens", "Forest Hills — I-495 LIE"),
        ("11377", "Queens", "Woodside — I-495 LIE, truck routes"),
        ("11378", "Queens", "Maspeth — Newtown Creek, industrial trucks"),
        ("11379", "Queens", "Middle Village — I-495 LIE"),
        ("11385", "Queens", "Ridgewood — Jackie Robinson Pkwy, BQE"),
        ("11411", "Queens", "Cambria Heights — I-678"),
        ("11412", "Queens", "St. Albans — I-678"),
        ("11413", "Queens", "Springfield Gardens — I-678, JFK trucks"),
        ("11414", "Queens", "Howard Beach — Belt Pkwy, JFK trucks"),
        ("11415", "Queens", "Kew Gardens — I-678, Van Wyck"),
        ("11416", "Queens", "Ozone Park — Van Wyck Expwy"),
        ("11417", "Queens", "Ozone Park — Van Wyck, Belt Pkwy"),
        ("11418", "Queens", "Richmond Hill — Van Wyck Expwy"),
        ("11419", "Queens", "South Richmond Hill — Van Wyck"),
        ("11420", "Queens", "South Ozone Park — Van Wyck, JFK"),
        ("11421", "Queens", "Woodhaven — Jackie Robinson Pkwy"),
        ("11422", "Queens", "Rosedale — I-678 truck corridor"),
        ("11423", "Queens", "Hollis — I-678"),
        ("11424", "Queens", "Jamaica — Van Wyck Expwy I-678"),
        ("11425", "Queens", "Jamaica — Van Wyck Expwy"),
        ("11426", "Queens", "Bellerose — low traffic comparison"),
        ("11427", "Queens", "Queens Village — I-495"),
        ("11428", "Queens", "Queens Village — I-495"),
        ("11429", "Queens", "Queens Village — I-495"),
        ("11430", "Queens", "JFK Airport — max truck/aviation traffic"),
        ("11432", "Queens", "Jamaica — Van Wyck, JFK trucks"),
        ("11433", "Queens", "Jamaica — Van Wyck Expwy"),
        ("11434", "Queens", "Jamaica — Van Wyck I-678, JFK trucks"),
        ("11435", "Queens", "Jamaica — Van Wyck, high commercial"),
        ("11436", "Queens", "Jamaica — Van Wyck Expwy"),
        ("11691", "Queens", "Far Rockaway — coastal, low traffic"),
        ("11692", "Queens", "Arverne — coastal"),
        ("11693", "Queens", "Rockaway Park — coastal"),
        ("11694", "Queens", "Rockaway Park — coastal"),
        ("11697", "Queens", "Breezy Point — coastal, low traffic"),

        # STATEN ISLAND
        ("10301", "Staten Island", "St. George — Ferry, Bayonne Bridge approach"),
        ("10302", "Staten Island", "Port Richmond — Goethals Bridge, industrial"),
        ("10303", "Staten Island", "Mariners Harbor — Bayonne Bridge, trucks"),
        ("10304", "Staten Island", "Stapleton — I-278 expressway"),
        ("10305", "Staten Island", "Rosebank — Verrazzano Bridge approach"),
        ("10306", "Staten Island", "New Dorp — I-278"),
        ("10307", "Staten Island", "Tottenville — low traffic, comparison"),
        ("10308", "Staten Island", "Great Kills — low traffic"),
        ("10309", "Staten Island", "Charleston — I-278, low traffic"),
        ("10310", "Staten Island", "West Brighton — I-278 truck corridor"),
        ("10311", "Staten Island", "Travis — I-278, industrial"),
        ("10312", "Staten Island", "Eltingville — low traffic, comparison"),
        ("10314", "Staten Island", "Heartland Village — I-278 expressway"),
    ]

    today = datetime.now().strftime("%Y-%m-%dT%H")
    records = []
    total_zips = len(all_nyc_zips)

    for i, (zipcode, borough, notes) in enumerate(all_nyc_zips, 1):
        url = "https://www.airnowapi.org/aq/observation/zipCode/current/"
        params = {
            "format":   "application/json",
            "zipCode":  zipcode,
            "distance": 5,    # tight radius — local readings only
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
                    obs["pulled_at"]     = today
                    records.append(obs)
                print(f"    [{i}/{total_zips}] ✓ {zipcode} — {borough} — {len(data)} reading(s)")
            else:
                print(f"    [{i}/{total_zips}] ⚠ {zipcode} — no data")

        except Exception as e:
            print(f"    [{i}/{total_zips}] ✗ {zipcode} failed: {e}")

    if records:
        df = pd.DataFrame(records)
        df["AQI"] = pd.to_numeric(df.get("AQI", pd.Series(dtype=float)), errors="coerce")
        out_path = os.path.join(DATA_DIR, "airnow_realtime_aqi.csv")
        df.to_csv(out_path, index=False)
        print(f"\n    ✓ Saved {len(df)} total readings from {total_zips} zip codes → {out_path}")
        return df

    print("    ⚠ No AirNow data retrieved")
    return pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  NYC Pollution & Disease — Data Ingestion")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    df_air    = fetch_nyc_air_quality()
    df_dohmh  = fetch_dohmh_indicators()
    df_airnow = fetch_airnow_aqi()

    print("\n" + "=" * 60)
    print("  Ingestion complete. Files saved to Data/ folder.")
    print(f"  NYC Air Quality rows : {len(df_air)}")
    print(f"  DOHMH rows           : {len(df_dohmh)}")
    print(f"  AirNow rows          : {len(df_airnow)}")
    print("=" * 60)
