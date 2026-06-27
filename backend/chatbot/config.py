"""
chatbot/config.py
-----------------
Single source of truth for all environment variables and
app-wide constants. Every other module imports from here.
Nothing else should call os.getenv() directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── OpenRouter (free LLM) ──────────────────────────────────────────────────────

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

LLM_MODEL: str    = os.getenv("LLM_MODEL",    "meta-llama/llama-3.1-8b-instruct:free")
LLM_FALLBACK: str = os.getenv("LLM_FALLBACK", "mistralai/mistral-7b-instruct:free")
LLM_MAX_TOKENS: int = 512


# ── Data paths ─────────────────────────────────────────────────────────────────

CSV_PATH:   str = os.getenv("CSV_PATH",   "data/merged_final.csv")
CHROMA_DIR: str = os.getenv("CHROMA_DIR", "data/.chroma")
LOGS_PATH:  str = os.getenv("LOGS_PATH",  "logs/usage.csv")


# ── Embedding model (local, free) ──────────────────────────────────────────────

EMBED_MODEL: str  = "all-MiniLM-L6-v2"
COLLECTION_NAME: str = "nyc_pollution"


# ── Retrieval settings ─────────────────────────────────────────────────────────

TOP_K: int = 8          # max rows to inject into every LLM prompt
STRUCT_K: int = 3       # rows from pandas filter
VECTOR_K: int = 6       # rows from Chroma semantic search


# ── Rate limit tracking ────────────────────────────────────────────────────────

DAILY_REQUEST_LIMIT: int = 200    # OpenRouter free tier cap
RATE_WARN_THRESHOLD: int = 180    # warn the team at 180 req/day


# ── Known NYC boroughs and UHF42 neighborhoods ─────────────────────────────────

BOROUGHS: list[str] = [
    "Bronx",
    "Brooklyn",
    "Manhattan",
    "Queens",
    "Staten Island",
]

# UHF42 neighborhoods — used by intent extractor to detect neighborhood-level queries
UHF_NEIGHBORHOODS: list[str] = [
    "hunts point", "mott haven", "south bronx", "high bridge", "morrisania",
    "fordham", "pelham", "williamsbridge", "riverdale", "northeast bronx",
    "greenpoint", "williamsburg", "bedford stuyvesant", "crown heights",
    "east new york", "brownsville", "flatbush", "borough park", "bensonhurst",
    "bay ridge", "sunset park", "coney island", "east flatbush", "canarsie",
    "washington heights", "harlem", "east harlem", "upper west side",
    "upper east side", "chelsea", "gramercy", "greenwich village", "lower east side",
    "financial district", "jamaica", "flushing", "ridgewood", "bayside",
    "rockaway", "stapleton", "south beach", "willowbrook", "port richmond",
    "gowanus", "carroll gardens",
]




def validate_config() -> list[str]:
    """
    Call this at startup. Returns a list of warning strings for any
    missing or suspicious config values. Does not raise — lets the
    app boot with degraded functionality and surface warnings via /health.
    """
    warnings = []
    if not OPENROUTER_API_KEY:
        warnings.append("OPENROUTER_API_KEY is not set — /chat will fail")
    if not os.path.exists(CSV_PATH):
        warnings.append(f"CSV not found at {CSV_PATH} — run src/datamerge.py first")
    if not os.path.exists(CHROMA_DIR):
        warnings.append(f"Chroma store not found at {CHROMA_DIR} — run src/ingest.py first")
    return warnings