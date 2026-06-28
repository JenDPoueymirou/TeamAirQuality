"""
src/dataingestion.py
--------------------
Pulls raw air quality and health data from five sources and saves each CSV to /data/.
Run from backend/: python src/dataingestion.py

Sources
-------
1. NYC Open Data (Socrata) — air quality & health indicators (c3uy-2p5r)
2. NYC Open Data (Socrata) — asthma ED + PM2.5 (ebe7-6eah)
3. EPA AirNow            — real-time AQI by ZIP
4. PurpleAir             — community PM2.5 sensors
5. NYC DOHMH Epiquery    — neighborhood ED visit rates (primary)
   SPARCS via Open Data  — ED visits by ZIP / ICD code (fallback, gnzp-ekau)
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

# ── ED visits lookup tables (Issue #11) ───────────────────────────────────────

# ZIP → UHF42 canonical neighborhood name (matches geo_place_name in merged CSV)
_ZIP_TO_UHF = {
    # Bronx
    "10454": "Hunts Point - Mott Haven", "10455": "Hunts Point - Mott Haven",
    "10459": "Hunts Point - Mott Haven", "10474": "Hunts Point - Mott Haven",
    "10453": "Crotona - Tremont",        "10456": "Crotona - Tremont",
    "10457": "Crotona - Tremont",
    "10452": "High Bridge - Morrisania", "10460": "High Bridge - Morrisania",
    "10458": "Fordham - Bronx Pk",       "10468": "Fordham - Bronx Pk",
    "10461": "Pelham - Throgs Neck",     "10462": "Pelham - Throgs Neck",
    "10464": "Pelham - Throgs Neck",     "10465": "Pelham - Throgs Neck",
    "10472": "Pelham - Throgs Neck",     "10473": "Pelham - Throgs Neck",
    "10463": "Kingsbridge - Riverdale",  "10471": "Kingsbridge - Riverdale",
    "10466": "Northeast Bronx",          "10467": "Northeast Bronx",
    "10469": "Northeast Bronx",          "10470": "Northeast Bronx",
    "10475": "Northeast Bronx",
    # Brooklyn
    "11211": "Greenpoint",               "11222": "Greenpoint",
    "11206": "Bedford Stuyvesant - Crown Heights",
    "11216": "Bedford Stuyvesant - Crown Heights",
    "11221": "Bedford Stuyvesant - Crown Heights",
    "11233": "Bedford Stuyvesant - Crown Heights",
    "11207": "East New York",            "11208": "East New York",
    "11212": "East New York",
    "11218": "Borough Park",             "11219": "Borough Park",
    "11230": "Borough Park",
    "11220": "Sunset Park",              "11232": "Sunset Park",
    "11203": "East Flatbush - Flatbush", "11210": "East Flatbush - Flatbush",
    "11226": "East Flatbush - Flatbush",
    "11234": "Canarsie - Flatlands",     "11236": "Canarsie - Flatlands",
    "11239": "Canarsie - Flatlands",
    "11209": "Bensonhurst - Bay Ridge",  "11214": "Bensonhurst - Bay Ridge",
    "11228": "Bensonhurst - Bay Ridge",
    "11223": "Coney Island - Sheepshead Bay",
    "11224": "Coney Island - Sheepshead Bay",
    "11235": "Coney Island - Sheepshead Bay",
    "11204": "Coney Island - Sheepshead Bay",
    "11201": "Downtown - Heights - Slope",
    "11215": "Downtown - Heights - Slope",
    "11217": "Downtown - Heights - Slope",
    "11231": "Downtown - Heights - Slope",
    # Manhattan
    "10026": "Central Harlem - Morningside Heights",
    "10027": "Central Harlem - Morningside Heights",
    "10031": "Central Harlem - Morningside Heights",
    "10037": "Central Harlem - Morningside Heights",
    "10039": "Central Harlem - Morningside Heights",
    "10029": "East Harlem",              "10035": "East Harlem",
    "10023": "Upper West Side",          "10024": "Upper West Side",
    "10025": "Upper West Side",
    "10021": "Upper East Side",          "10028": "Upper East Side",
    "10065": "Upper East Side",          "10075": "Upper East Side",
    "10128": "Upper East Side",
    "10001": "Chelsea - Clinton",        "10011": "Chelsea - Clinton",
    "10018": "Chelsea - Clinton",        "10019": "Chelsea - Clinton",
    "10036": "Chelsea - Clinton",
    "10010": "Gramercy Park - Murray Hill",
    "10016": "Gramercy Park - Murray Hill",
    "10017": "Gramercy Park - Murray Hill",
    "10022": "Gramercy Park - Murray Hill",
    "10003": "Greenwich Village - SoHo", "10012": "Greenwich Village - SoHo",
    "10013": "Greenwich Village - SoHo", "10014": "Greenwich Village - SoHo",
    "10002": "Union Square - Lower East Side",
    "10009": "Union Square - Lower East Side",
    "10004": "Lower Manhattan",          "10005": "Lower Manhattan",
    "10006": "Lower Manhattan",          "10007": "Lower Manhattan",
    "10038": "Lower Manhattan",
    "10032": "Washington Heights",       "10033": "Washington Heights",
    "10034": "Washington Heights",       "10040": "Washington Heights",
    # Queens
    "11101": "Long Island City - Astoria", "11102": "Long Island City - Astoria",
    "11103": "Long Island City - Astoria", "11104": "Long Island City - Astoria",
    "11105": "Long Island City - Astoria", "11106": "Long Island City - Astoria",
    "11368": "West Queens",              "11372": "West Queens",
    "11373": "West Queens",              "11374": "West Queens",
    "11377": "West Queens",              "11378": "West Queens",
    "11354": "Flushing - Clearview",     "11355": "Flushing - Clearview",
    "11356": "Flushing - Clearview",     "11357": "Flushing - Clearview",
    "11358": "Flushing - Clearview",     "11360": "Flushing - Clearview",
    "11361": "Bayside - Little Neck",    "11362": "Bayside - Little Neck",
    "11363": "Bayside - Little Neck",    "11364": "Bayside - Little Neck",
    "11365": "Fresh Meadows",            "11366": "Fresh Meadows",
    "11367": "Fresh Meadows",            "11375": "Fresh Meadows",
    "11432": "Jamaica",                  "11433": "Jamaica",
    "11434": "Jamaica",                  "11435": "Jamaica",
    "11436": "Jamaica",
    "11411": "Southeast Queens",         "11412": "Southeast Queens",
    "11413": "Southeast Queens",         "11422": "Southeast Queens",
    "11423": "Southeast Queens",         "11428": "Southeast Queens",
    "11429": "Southeast Queens",
    "11414": "Southwest Queens",         "11415": "Southwest Queens",
    "11416": "Southwest Queens",         "11417": "Southwest Queens",
    "11419": "Southwest Queens",         "11420": "Southwest Queens",
    "11421": "Southwest Queens",
    "11385": "Ridgewood - Forest Hills", "11379": "Ridgewood - Forest Hills",
    "11418": "Ridgewood - Forest Hills",
    "11691": "Rockaways",                "11692": "Rockaways",
    "11693": "Rockaways",                "11694": "Rockaways",
    "11695": "Rockaways",                "11697": "Rockaways",
    # Staten Island
    "10302": "Port Richmond",            "10303": "Port Richmond",
    "10310": "Port Richmond",
    "10301": "Stapleton - St. George",   "10304": "Stapleton - St. George",
    "10305": "Stapleton - St. George",
    "10306": "South Beach - Tottenville", "10307": "South Beach - Tottenville",
    "10308": "South Beach - Tottenville", "10309": "South Beach - Tottenville",
    "10312": "South Beach - Tottenville",
    "10311": "Willowbrook",              "10314": "Willowbrook",
    "10313": "Southern SI",
}

_UHF_BOROUGH = {
    "Hunts Point - Mott Haven":             "Bronx",
    "Crotona - Tremont":                    "Bronx",
    "High Bridge - Morrisania":             "Bronx",
    "Fordham - Bronx Pk":                   "Bronx",
    "Pelham - Throgs Neck":                 "Bronx",
    "Kingsbridge - Riverdale":              "Bronx",
    "Northeast Bronx":                      "Bronx",
    "Greenpoint":                           "Brooklyn",
    "Bedford Stuyvesant - Crown Heights":   "Brooklyn",
    "East New York":                        "Brooklyn",
    "Borough Park":                         "Brooklyn",
    "East Flatbush - Flatbush":             "Brooklyn",
    "Canarsie - Flatlands":                 "Brooklyn",
    "Bensonhurst - Bay Ridge":              "Brooklyn",
    "Coney Island - Sheepshead Bay":        "Brooklyn",
    "Sunset Park":                          "Brooklyn",
    "Downtown - Heights - Slope":           "Brooklyn",
    "Central Harlem - Morningside Heights": "Manhattan",
    "East Harlem":                          "Manhattan",
    "Upper West Side":                      "Manhattan",
    "Upper East Side":                      "Manhattan",
    "Chelsea - Clinton":                    "Manhattan",
    "Gramercy Park - Murray Hill":          "Manhattan",
    "Greenwich Village - SoHo":             "Manhattan",
    "Union Square - Lower East Side":       "Manhattan",
    "Lower Manhattan":                      "Manhattan",
    "Washington Heights":                   "Manhattan",
    "Long Island City - Astoria":           "Queens",
    "West Queens":                          "Queens",
    "Flushing - Clearview":                 "Queens",
    "Bayside - Little Neck":                "Queens",
    "Fresh Meadows":                        "Queens",
    "Jamaica":                              "Queens",
    "Southwest Queens":                     "Queens",
    "Southeast Queens":                     "Queens",
    "Ridgewood - Forest Hills":             "Queens",
    "Rockaways":                            "Queens",
    "Port Richmond":                        "Staten Island",
    "Stapleton - St. George":               "Staten Island",
    "South Beach - Tottenville":            "Staten Island",
    "Willowbrook":                          "Staten Island",
    "Southern SI":                          "Staten Island",
}

# ACS 2015-2019 five-year population estimates — denominator for per-100k rates
_UHF_POP = {
    "Hunts Point - Mott Haven":              92_400,
    "Crotona - Tremont":                    114_500,
    "High Bridge - Morrisania":             140_000,
    "Fordham - Bronx Pk":                   191_000,
    "Pelham - Throgs Neck":                 166_000,
    "Kingsbridge - Riverdale":              138_000,
    "Northeast Bronx":                      170_000,
    "Greenpoint":                           162_000,
    "Bedford Stuyvesant - Crown Heights":   269_000,
    "East New York":                        173_000,
    "Borough Park":                         217_000,
    "East Flatbush - Flatbush":             210_000,
    "Canarsie - Flatlands":                 148_000,
    "Bensonhurst - Bay Ridge":              174_000,
    "Coney Island - Sheepshead Bay":        196_000,
    "Sunset Park":                          133_000,
    "Downtown - Heights - Slope":           203_000,
    "Central Harlem - Morningside Heights": 136_000,
    "East Harlem":                          114_000,
    "Upper West Side":                      205_000,
    "Upper East Side":                      209_000,
    "Chelsea - Clinton":                    153_000,
    "Gramercy Park - Murray Hill":          176_000,
    "Greenwich Village - SoHo":             101_000,
    "Union Square - Lower East Side":       165_000,
    "Lower Manhattan":                       53_000,
    "Washington Heights":                   222_000,
    "Long Island City - Astoria":           175_000,
    "West Queens":                          244_000,
    "Flushing - Clearview":                 165_000,
    "Bayside - Little Neck":                127_000,
    "Fresh Meadows":                        107_000,
    "Jamaica":                              174_000,
    "Southwest Queens":                     232_000,
    "Southeast Queens":                     205_000,
    "Ridgewood - Forest Hills":             181_000,
    "Rockaways":                             99_000,
    "Port Richmond":                         76_000,
    "Stapleton - St. George":               138_000,
    "South Beach - Tottenville":            141_000,
    "Willowbrook":                           83_000,
    "Southern SI":                           55_000,
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


# ── ED visit fetchers (Issue #11) ─────────────────────────────────────────────

def _try_epiquery() -> pd.DataFrame | None:
    """
    Attempt NYC DOHMH Epiquery REST endpoint.
    Returns a normalized DataFrame on success, None if the endpoint is unavailable.
    The Epiquery system is a web app; failure here is expected and handled upstream.
    """
    candidate_urls = [
        "https://a816-dbcs.nyc.gov/api/epiquery/v1/EpiQuery",
        "https://a816-dbcs.nyc.gov/bvars/EPIQUERY/ByNeighborhood",
        "https://a816-dohbesp.nyc.gov/IndicatorPublic/api/v1/data/GetIndicatorData",
    ]
    for url in candidate_urls:
        try:
            r = requests.get(url, params={"format": "json", "$limit": 1}, timeout=5)
            if r.status_code != 200:
                continue
            payload = r.json()
            if not isinstance(payload, list) or len(payload) == 0:
                continue
            df = pd.DataFrame(payload)
            low = {c.lower() for c in df.columns}
            # Require at least a neighborhood/geo column and a value column
            if any("neighborhood" in c or "uhf" in c or "geo" in c for c in low):
                print(f"    Epiquery responded at {url}")
                return df
        except Exception:
            continue
    return None


def _try_sparcs(out_path: Path) -> bool:
    """
    Pull ED visit data from SPARCS (gnzp-ekau) via NYC Open Data / Socrata.

    Strategy:
    - Fetch visit counts filtered to respiratory (ICD J*) and cardiovascular (ICD I*)
    - Map patient ZIP → UHF42 neighborhood via _ZIP_TO_UHF
    - Aggregate counts per neighborhood per year
    - Compute per-100k rates using _UHF_POP population estimates
    - Save to out_path

    Returns True if the output file was written with at least one row.
    """
    base = "https://data.cityofnewyork.us/resource/gnzp-ekau.json"

    # ── Schema probe ──────────────────────────────────────────────────────────
    try:
        r = requests.get(base, params={"$limit": 1}, timeout=20)
        r.raise_for_status()
        sample = r.json()
    except Exception as exc:
        print(f"  [err]  SPARCS schema probe: {exc}")
        return False

    if not sample:
        print("  [warn] SPARCS: returned empty dataset")
        return False

    cols = list(sample[0].keys())

    # Detect column names — SPARCS schema changes between NYC Open Data releases
    def _find(cols: list[str], *keywords: str) -> str | None:
        for kw in keywords:
            match = next((c for c in cols if kw in c.lower()), None)
            if match:
                return match
        return None

    year_col  = _find(cols, "year", "yr")
    zip_col   = _find(cols, "zip")
    diag_col  = _find(cols, "diag", "icd", "dx", "primary_dx", "indication", "category")
    count_col = _find(cols, "discharge", "visit", "count", "total")

    if not zip_col:
        print(f"  [warn] SPARCS: no ZIP column detected — columns: {cols[:12]}")
        return False

    # ── Fetch by diagnosis category ───────────────────────────────────────────
    # ICD-10: J = respiratory (J00-J99), I = cardiovascular (I00-I99)
    # ICD-9:  respiratory 460-519, cardiovascular 390-459
    # We query for letter prefixes — works for ICD-10; ICD-9 rows are excluded
    # by the absence of a letter prefix in the range filter.
    categories = {
        "asthma":         ("J45", "starts_with"),   # asthma-specific
        "respiratory":    ("J",   "starts_with"),
        "cardiovascular": ("I",   "starts_with"),
    }

    agg: dict[tuple, dict] = {}   # (neighborhood, year) → {category: count}

    for cat_name, (icd_prefix, _) in categories.items():
        where_clause = (
            f"starts_with({diag_col}, '{icd_prefix}')"
            if diag_col
            else "1=1"
        )
        params: dict = {
            "$limit": 50_000,
            "$where": where_clause,
        }
        if year_col:
            params["$order"] = year_col

        try:
            r = requests.get(base, params=params, timeout=60)
            r.raise_for_status()
            rows = r.json()
        except Exception as exc:
            print(f"  [warn] SPARCS fetch ({cat_name}): {exc}")
            continue

        for row in rows:
            raw_zip = str(row.get(zip_col, "")).strip().zfill(5)[:5]
            neighborhood = _ZIP_TO_UHF.get(raw_zip)
            if not neighborhood:
                continue

            year_raw = row.get(year_col, "") if year_col else ""
            try:
                year = int(str(year_raw)[:4])
            except (ValueError, TypeError):
                continue

            count_raw = row.get(count_col, 0) if count_col else 1
            try:
                count = int(float(str(count_raw)))
            except (ValueError, TypeError):
                count = 1

            key = (neighborhood, year)
            if key not in agg:
                agg[key] = {"asthma": 0, "respiratory": 0, "cardiovascular": 0}
            agg[key][cat_name] += count

    if not agg:
        print("  [warn] SPARCS: no rows matched ZIP → UHF mapping")
        return False

    # ── Build output DataFrame ─────────────────────────────────────────────────
    records = []
    for (neighborhood, year), counts in sorted(agg.items()):
        pop = _UHF_POP.get(neighborhood, 100_000)
        records.append({
            "neighborhood_name":     neighborhood,
            "borough":               _UHF_BOROUGH.get(neighborhood, ""),
            "year":                  year,
            "asthma_ed_rate":        round(counts["asthma"]         / pop * 100_000, 1),
            "cardiovascular_ed_rate":round(counts["cardiovascular"]  / pop * 100_000, 1),
            "respiratory_ed_rate":   round(counts["respiratory"]     / pop * 100_000, 1),
            "total_ed_visits":       counts["respiratory"] + counts["cardiovascular"],
            "population":            pop,
        })

    df = pd.DataFrame(records)
    _save(df, out_path, "ED visits (SPARCS fallback)")
    return True


def pull_er_visits(out_path: Path) -> None:
    """
    Pull neighborhood-level ED visit rates (asthma, cardiovascular, respiratory).

    Primary:  NYC DOHMH Epiquery (a816-dbcs.nyc.gov)
    Fallback: SPARCS via NYC Open Data (gnzp-ekau), aggregated ZIP → UHF42

    Output columns: neighborhood_name, borough, year, asthma_ed_rate,
    cardiovascular_ed_rate, respiratory_ed_rate, total_ed_visits, population
    """
    print("  [try]  Epiquery (a816-dbcs.nyc.gov)...")
    eq_df = _try_epiquery()
    if eq_df is not None:
        # Normalize column names to the required output schema
        col_map = {}
        for col in eq_df.columns:
            lc = col.lower()
            if "neighborhood" in lc or "geo_place" in lc:
                col_map[col] = "neighborhood_name"
            elif "borough" in lc:
                col_map[col] = "borough"
            elif "year" in lc:
                col_map[col] = "year"
            elif "asthma" in lc and ("ed" in lc or "emergency" in lc):
                col_map[col] = "asthma_ed_rate"
            elif "cardio" in lc and ("ed" in lc or "emergency" in lc):
                col_map[col] = "cardiovascular_ed_rate"
            elif "resp" in lc and ("ed" in lc or "emergency" in lc):
                col_map[col] = "respiratory_ed_rate"
            elif "pop" in lc:
                col_map[col] = "population"
        eq_df = eq_df.rename(columns=col_map)
        required = {"neighborhood_name", "asthma_ed_rate"}
        if required.issubset(eq_df.columns):
            _save(eq_df, out_path, "ED visits (Epiquery)")
            return
        print("  [warn] Epiquery response missing expected columns — falling back")

    print("  [try]  SPARCS fallback (NYC Open Data gnzp-ekau)...")
    if _try_sparcs(out_path):
        return

    print("  [skip] ED visits: both Epiquery and SPARCS unavailable")
    print("         Run the server without er_visits_neighborhood.csv —")
    print("         asthma_ed_rate columns will be absent until this step completes.")


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
    pull_er_visits(DATA_DIR / "er_visits_neighborhood.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
