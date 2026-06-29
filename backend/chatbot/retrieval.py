"""
chatbot/retrieval.py
--------------------
Hybrid retrieval layer — runs before every LLM call.

Two search strategies run on every query and results are merged:
  1. Structured filter  — keyword + regex scan -> pandas DataFrame filter
  2. Semantic search    — local embedding -> ChromaDB vector search

This is what makes the chatbot grounded. The LLM never sees the full
dataset — only the 6-8 most relevant rows for the current question.
"""

import re
import logging
import pandas as pd

from chatbot.config import (
    BOROUGHS,
    GEMINI_EMBED_MODEL,
    STRUCT_K,
    TOP_K,
    UHF_NEIGHBORHOODS,
    UHF_TO_BOROUGH,
    VECTOR_K,
)

log = logging.getLogger(__name__)

# Module-level state — populated once at startup by init_retrieval().
_df: pd.DataFrame = pd.DataFrame()
_collection = None
_gemini_client = None       # chat client passed from main.py (kept for availability check)
_gemini_embed_client = None  # separate v1 client for text-embedding-004


def _embed_query(query: str) -> list[float]:
    """Embed a single query string via the Gemini embedContent REST endpoint.
    Uses requests directly to avoid google-genai SDK version quirks."""
    import requests as _req
    from chatbot.config import GEMINI_API_KEY as _key
    url = (
        f"https://generativelanguage.googleapis.com"
        f"/v1beta/models/{GEMINI_EMBED_MODEL}:embedContent"
    )
    r = _req.post(
        url,
        json={"content": {"parts": [{"text": query}]}},
        headers={"x-goog-api-key": _key},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["embedding"]["values"]


def init_retrieval(df: pd.DataFrame, collection, gemini_client) -> None:
    """
    Store data dependencies in module state.
    Called once from main.py lifespan so retrieve() needs no arguments
    beyond the query string.
    """
    global _df, _collection, _gemini_client
    _df = df
    _collection = collection
    _gemini_client = gemini_client


# ── Intent extraction ──────────────────────────────────────────────────────────

def extract_filters(query: str) -> dict:
    """
    Scan the user's question for structured clues without an LLM.
    Returns only keys actually detected — empty dict if nothing found.

    Detects:
      borough      — one or more names from BOROUGHS (str if one, list if many)
      neighborhood — first match from UHF_NEIGHBORHOODS list
      year         — 4-digit year in range 2005-2024
      zip_code     — 5-digit NYC ZIP starting with 1
    """
    q = query.lower().strip()
    filters: dict = {}

    detected = [b for b in BOROUGHS if b.lower() in q]
    if len(detected) == 1:
        filters["borough"] = detected[0]
    elif len(detected) > 1:
        filters["borough"] = detected  # list — handled downstream with isin / $in

    for neighborhood in UHF_NEIGHBORHOODS:
        if neighborhood in q:
            filters["neighborhood"] = neighborhood
            break

    # Infer borough from neighborhood when borough isn't explicit in the query.
    # "What is the asthma rate in Hunts Point?" → borough: Bronx (no word "Bronx" present).
    # Inferred borough flows into the Chroma where= clause in vector_search(),
    # keeping semantic results geographically scoped to the right borough.
    if "neighborhood" in filters and "borough" not in filters:
        inferred = UHF_TO_BOROUGH.get(filters["neighborhood"])
        if inferred:
            filters["borough"] = inferred

    year_match = re.search(r"\b(20(?:0[5-9]|1\d|2[0-4]))\b", q)
    if year_match:
        filters["year"] = year_match.group(1)

    zip_match = re.search(r"\b(1\d{4})\b", q)
    if zip_match:
        filters["zip_code"] = zip_match.group(1)

    if filters:
        log.info("Filters extracted: %s", filters)
    else:
        log.info("No filters extracted — falling back to pure semantic search")

    return filters


# ── Structured filter ──────────────────────────────────────────────────────────

def structured_filter(filters: dict, top_n: int = STRUCT_K) -> list[dict]:
    """
    Filter the in-memory DataFrame using detected structured values.
    Returns [] if nothing matches or df is not loaded.
    """
    if _df.empty:
        return []

    result = _df.copy()

    if "borough" in filters and "borough" in result.columns:
        b = filters["borough"]
        if isinstance(b, list):
            result = result[result["borough"].isin(b)]
        else:
            result = result[result["borough"].str.lower() == b.lower()]

    if "neighborhood" in filters and "geo_place_name" in result.columns:
        result = result[
            result["geo_place_name"].str.lower().str.contains(
                filters["neighborhood"].lower(), na=False
            )
        ]

    if "zip_code" in filters and "zip_code" in result.columns:
        result = result[result["zip_code"].astype(str) == filters["zip_code"]]

    if "year" in filters and "time_period" in result.columns:
        result = result[
            result["time_period"].astype(str).str.contains(filters["year"], na=False)
        ]

    return result.head(top_n).to_dict(orient="records")


# ── Semantic search ────────────────────────────────────────────────────────────

def vector_search(query: str, filters: dict, top_k: int = VECTOR_K) -> list[dict]:
    """
    Embed the query with Gemini text-embedding-004, then query ChromaDB
    for the most semantically similar stored rows.

    Applies a borough pre-filter using Chroma's where= clause when one or
    more boroughs were detected. Supports both single ($eq) and multi ($in).

    Returns [] on any error — semantic search is non-fatal.
    """
    if _collection is None:
        log.warning("Semantic search unavailable — collection not loaded")
        return []

    try:
        query_vector = [_embed_query(query)]

        where = None
        if "borough" in filters:
            b = filters["borough"]
            if isinstance(b, list):
                where = {"borough": {"$in": b}}
            else:
                where = {"borough": {"$eq": b}}

        results = _collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            where=where,
            include=["metadatas"],
        )

        return results["metadatas"][0] if results["metadatas"] else []

    except Exception as e:
        log.warning("Semantic search failed: %s", e)
        return []


# ── Row formatter ──────────────────────────────────────────────────────────────

def format_row(row: dict, index: int) -> str:
    """
    Convert a raw data row into a labeled text chunk for the LLM prompt.
    Only fields that are present and non-null are included — no "None" in output.
    ED rate fields (Issues #11/#12) are silently skipped when not yet populated.
    """
    field_map = [
        ("Borough",           "borough"),
        ("Area",              "geo_place_name"),
        ("ZIP",               "zip_code"),
        ("Period",            "time_period"),
        ("PM2.5",             "pm25"),
        ("NO2",               "no2"),
        ("Ozone",             "ozone"),
        ("AQI",               "aqi"),
        ("PurpleAir PM2.5",   "purpleair_pm25"),
        ("Truck VMT",         "truck_vmt"),
        ("Asthma ER rate",    "asthma_er_rate"),
        ("Asthma ED rate",    "asthma_ed_rate"),
        ("Cardio hosp rate",  "cardiovascular_hosp_rate"),
        ("Cardio ED rate",    "cardiovascular_ed_rate"),
        ("Resp hosp rate",    "respiratory_hosp_rate"),
        ("Resp ED rate",      "respiratory_ed_rate"),
        ("PM2.5 deaths",      "pm25_deaths"),
    ]

    parts = []
    for label, key in field_map:
        val = row.get(key)
        if val is not None and str(val).strip() not in ("", "nan", "None"):
            parts.append(f"{label}: {val}")

    return f"[Row {index}] " + " | ".join(parts)


# ── Main retrieve function ─────────────────────────────────────────────────────

def retrieve(query: str) -> tuple[list[str], dict, int]:
    """
    Entry point for all retrieval. Called by the /chat endpoint before
    every LLM call.

    Returns:
      chunks    — formatted [Row N] strings ready to inject into the prompt
      filters   — detected structured filters (for logging / debug)
      row_count — number of unique rows returned (capped at TOP_K)
    """
    filters = extract_filters(query)

    struct_rows = structured_filter(filters)
    vector_rows = vector_search(query, filters)

    # Merge, deduplicating on (geo_place_name + time_period).
    # Structured rows go first — higher confidence than semantic matches.
    seen: set = set()
    combined: list[dict] = []

    for row in struct_rows + vector_rows:
        key = (
            str(row.get("geo_place_name", "")).lower()
            + "_"
            + str(row.get("time_period", ""))
        )
        if key not in seen:
            seen.add(key)
            combined.append(row)

    top = combined[:TOP_K]
    chunks = [format_row(row, i + 1) for i, row in enumerate(top)]

    log.info(
        "Retrieved %d chunks (%d structured, %d semantic) for query: '%s...'",
        len(chunks),
        min(len(struct_rows), TOP_K),
        min(len(vector_rows), TOP_K),
        query[:60],
    )

    return chunks, filters, len(chunks)
