"""
chatbot/retrieval.py
--------------------
Hybrid retrieval layer — runs before every LLM call.

Two search strategies run in parallel and their results are merged:
  1. Structured filter  — keyword + regex scan → pandas DataFrame filter
  2. Semantic search    — local embedding → ChromaDB vector search

This is what makes the chatbot grounded. The LLM never sees the full
dataset — only the 6-8 most relevant rows for the current question.
"""

import re
import logging
import pandas as pd

from chatbot.config import (
    BOROUGHS,
    STRUCT_K,
    TOP_K,
    UHF_NEIGHBORHOODS,
    VECTOR_K,
)

log = logging.getLogger(__name__)

# Module-level state — populated once at startup by init_retrieval().
_df: pd.DataFrame = pd.DataFrame()
_collection = None
_embed_model = None


def init_retrieval(df: pd.DataFrame, collection, embed_model) -> None:
    """
    Store the data dependencies in module state.
    Called once from main.py lifespan so the chat endpoint can call
    retrieve(query) without threading these objects through every layer.
    """
    global _df, _collection, _embed_model
    _df = df
    _collection = collection
    _embed_model = embed_model


# ── Intent extraction ──────────────────────────────────────────────────────────

def extract_filters(query: str) -> dict:
    """
    Scan the user's question for structured clues without an LLM.
    Returns only keys that were actually detected — empty dict if nothing found.

    Detects:
      - borough name (from known list, case-insensitive)
      - neighborhood name (from UHF42 list, case-insensitive)
      - 4-digit year (2000–2029)
      - 5-digit NYC ZIP code (starts with 1)
    """
    q = query.lower().strip()
    filters: dict = {}

    # Borough detection
    for borough in BOROUGHS:
        if borough.lower() in q:
            filters["borough"] = borough
            break

    # UHF neighborhood detection
    for neighborhood in UHF_NEIGHBORHOODS:
        if neighborhood in q:
            filters["neighborhood"] = neighborhood
            break

    # Year detection
    year_match = re.search(r"\b(20[0-2]\d)\b", q)
    if year_match:
        filters["year"] = year_match.group(1)

    # ZIP code detection (NYC ZIPs start with 1)
    zip_match = re.search(r"\b(1\d{4})\b", q)
    if zip_match:
        filters["zip_code"] = zip_match.group(1)

    if filters:
        log.info("Filters extracted: %s", filters)
    else:
        log.info("No filters extracted — falling back to pure semantic search")

    return filters


# ── Structured filter ──────────────────────────────────────────────────────────

def apply_df_filters(
    df: pd.DataFrame,
    filters: dict,
    top_n: int = STRUCT_K,
) -> list[dict]:
    """
    Filter the in-memory DataFrame using detected structured values.
    Returns a list of row dicts — empty list if nothing matches or df is empty.
    """
    if df.empty:
        return []

    result = df.copy()

    if "borough" in filters and "borough" in result.columns:
        result = result[
            result["borough"].str.lower() == filters["borough"].lower()
        ]

    if "neighborhood" in filters and "geo_place_name" in result.columns:
        result = result[
            result["geo_place_name"].str.lower().str.contains(
                filters["neighborhood"].lower(), na=False
            )
        ]

    if "zip_code" in filters and "zip_code" in result.columns:
        result = result[
            result["zip_code"].astype(str) == filters["zip_code"]
        ]

    if "year" in filters and "time_period" in result.columns:
        result = result[
            result["time_period"].astype(str).str.contains(
                filters["year"], na=False
            )
        ]

    return result.head(top_n).to_dict(orient="records")


# ── Semantic search ────────────────────────────────────────────────────────────

def vector_search(
    query: str,
    collection,
    embed_model,
    borough_filter: str | None = None,
    top_k: int = VECTOR_K,
) -> list[dict]:
    """
    Embed the query locally with sentence-transformers, then query ChromaDB
    for the most semantically similar stored rows.

    Optionally applies a borough pre-filter using Chroma's where= clause
    so semantic search stays scoped to the right geography.

    Returns a list of metadata dicts (one per row).
    """
    if collection is None or embed_model is None:
        log.warning("Semantic search unavailable — collection or model not loaded")
        return []

    try:
        query_vector = embed_model.encode([query]).tolist()

        where = {}
        if borough_filter:
            where = {"borough": {"$eq": borough_filter}}

        results = collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            where=where if where else None,
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
    Only includes fields that are present and non-null.

    Example output:
      [Row 3] Borough: Bronx | Area: Hunts Point | ZIP: 10474 |
              Period: 2019 | PM2.5: 18.2 | Asthma ER rate: 210.0 |
              Cardio hosp rate: 145.3
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
        ("Cardio hosp rate",  "cardiovascular_hosp_rate"),
        ("Resp hosp rate",    "respiratory_hosp_rate"),
        ("PM2.5 deaths",      "pm25_deaths"),
    ]

    parts = []
    for label, key in field_map:
        val = row.get(key)
        if val is not None and str(val).strip() not in ("", "nan", "None"):
            parts.append(f"{label}: {val}")

    return f"[Row {index}] " + " | ".join(parts)


# ── Main retrieve function ─────────────────────────────────────────────────────

def retrieve(
    query: str,
    df: pd.DataFrame,
    collection,
    embed_model,
) -> tuple[list[str], dict]:
    """
    Entry point for all retrieval. Called by the /chat endpoint before
    every LLM call.

    Returns:
      chunks  — list of formatted [Row N] strings ready to inject into the prompt
      filters — the structured filters that were detected (for logging)
    """
    filters = extract_filters(query)

    # Structured rows from CSV
    struct_rows = apply_df_filters(df, filters, top_n=STRUCT_K)

    # Semantic rows from ChromaDB
    vector_rows = vector_search(
        query=query,
        collection=collection,
        embed_model=embed_model,
        borough_filter=filters.get("borough"),
        top_k=VECTOR_K,
    )

    # Merge and deduplicate on (geo_place_name + time_period)
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

    # Cap at TOP_K and format
    chunks = [
        format_row(row, i + 1)
        for i, row in enumerate(combined[:TOP_K])
    ]

    log.info(
        "Retrieved %d chunks (%d structured, %d semantic) for query: '%s...'",
        len(chunks),
        min(len(struct_rows), TOP_K),
        min(len(vector_rows), TOP_K),
        query[:60],
    )

    return chunks, filters