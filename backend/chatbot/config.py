"""
config.py
---------
Single source of truth for all environment variables and constants.
Restructured to use Google Gemini instead of OpenRouter.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API keys ──────────────────────────────────────────────────────────────────
# All loaded via os.getenv() — never hardcoded, never printed as values.
# GEMINI_API_KEY  : required for /chat (LLM calls)
# AIRNOW_API_KEY  : required for src/dataingestion.py (AirNow AQI fetch)
# PURPLEAIR_API_KEY: required for src/dataingestion.py (PurpleAir fetch)
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
AIRNOW_API_KEY    = os.getenv("AIRNOW_API_KEY")
PURPLEAIR_API_KEY = os.getenv("PURPLEAIR_API_KEY")

# Gemini 2.0 Flash — free tier, 1M token context, 1500 req/day
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# ── Embedding model (Gemini API — no local model, no PyTorch) ─────────────────
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")  # 1,500 req/day free tier

# ── File paths ─────────────────────────────────────────────────────────────────
CSV_PATH        = os.getenv("CSV_PATH",   "data/merged_final.csv")
CHROMA_DIR      = os.getenv("CHROMA_DIR", "data/.chroma")
COLLECTION_NAME = "nyc_pollution"

# ── Retrieval settings ─────────────────────────────────────────────────────────
TOP_K    = int(os.getenv("TOP_K", "8"))
STRUCT_K = 3   # rows from pandas structured filter
VECTOR_K = 6   # rows from ChromaDB semantic search

# ── Gemini free tier limits ────────────────────────────────────────────────────
DAILY_REQUEST_LIMIT          = 1500   # Gemini 2.0 Flash free tier
RATE_LIMIT_WARNING_THRESHOLD = 1400   # warn when close to cap
MINUTE_REQUEST_LIMIT         = 15     # requests per minute on free tier

# ── Known NYC boroughs ─────────────────────────────────────────────────────────
BOROUGHS = [
    "Bronx",
    "Brooklyn", 
    "Manhattan",
    "Queens",
    "Staten Island",
]

# ── UHF42 neighborhood list ────────────────────────────────────────────────────
UHF_NEIGHBORHOODS = [
    "hunts point", "mott haven", "south bronx", "crotona", "tremont",
    "highbridge", "fordham", "pelham", "williamsbridge", "riverdale",
    "greenpoint", "williamsburg", "bushwick", "east new york", "brownsville",
    "flatbush", "canarsie", "flatlands", "bensonhurst", "bay ridge",
    "borough park", "sunset park", "crown heights", "bedford stuyvesant",
    "bed stuy", "east harlem", "central harlem", "washington heights",
    "inwood", "upper west side", "upper east side", "chelsea", "clinton",
    "gramercy park", "lower east side", "chinatown", "downtown",
    "gowanus", "park slope", "jamaica", "flushing", "bayside",
    "fresh meadows", "ridgewood", "forest hills", "astoria",
    "long island city", "stapleton", "port richmond", "willowbrook",
]

# ── Neighborhood → borough lookup ─────────────────────────────────────────────
# Lets extract_filters() infer borough when only a neighborhood name appears in
# the query (e.g. "Hunts Point" → Bronx). Covers every entry in UHF_NEIGHBORHOODS.
UHF_TO_BOROUGH: dict[str, str] = {
    # Bronx
    "hunts point":       "Bronx",
    "mott haven":        "Bronx",
    "south bronx":       "Bronx",
    "crotona":           "Bronx",
    "tremont":           "Bronx",
    "highbridge":        "Bronx",
    "fordham":           "Bronx",
    "pelham":            "Bronx",
    "williamsbridge":    "Bronx",
    "riverdale":         "Bronx",
    # Brooklyn
    "greenpoint":        "Brooklyn",
    "williamsburg":      "Brooklyn",
    "bushwick":          "Brooklyn",
    "east new york":     "Brooklyn",
    "brownsville":       "Brooklyn",
    "flatbush":          "Brooklyn",
    "canarsie":          "Brooklyn",
    "flatlands":         "Brooklyn",
    "bensonhurst":       "Brooklyn",
    "bay ridge":         "Brooklyn",
    "borough park":      "Brooklyn",
    "sunset park":       "Brooklyn",
    "crown heights":     "Brooklyn",
    "bedford stuyvesant":"Brooklyn",
    "bed stuy":          "Brooklyn",
    "gowanus":           "Brooklyn",
    "park slope":        "Brooklyn",
    # Manhattan
    "east harlem":       "Manhattan",
    "central harlem":    "Manhattan",
    "washington heights":"Manhattan",
    "inwood":            "Manhattan",
    "upper west side":   "Manhattan",
    "upper east side":   "Manhattan",
    "chelsea":           "Manhattan",
    "clinton":           "Manhattan",
    "gramercy park":     "Manhattan",
    "lower east side":   "Manhattan",
    "chinatown":         "Manhattan",
    "downtown":          "Manhattan",
    # Queens
    "jamaica":           "Queens",
    "flushing":          "Queens",
    "bayside":           "Queens",
    "fresh meadows":     "Queens",
    "ridgewood":         "Queens",
    "forest hills":      "Queens",
    "astoria":           "Queens",
    "long island city":  "Queens",
    # Staten Island
    "stapleton":         "Staten Island",
    "port richmond":     "Staten Island",
    "willowbrook":       "Staten Island",
}


# ── Key presence summary ───────────────────────────────────────────────────────

def safe_config_summary() -> dict[str, bool]:
    """
    Return which API keys are configured — True/False presence flags only.
    Never includes actual key values. Safe to expose in /health and logs.
    """
    return {
        "GEMINI_API_KEY":    bool(GEMINI_API_KEY),
        "AIRNOW_API_KEY":    bool(AIRNOW_API_KEY),
        "PURPLEAIR_API_KEY": bool(PURPLEAIR_API_KEY),
    }