"""
src/ingest.py
-------------
Reads data/merged_final.csv, converts each row to a natural-language
sentence, embeds it with a free local model, and upserts the vectors
to ChromaDB on disk.

Run from backend/: python src/ingest.py
Idempotent: re-running produces the same vector count (stable IDs).
"""

import hashlib
import math
import sys
import time
from pathlib import Path

import pandas as pd

# Import shared constants so this stays in sync with the chatbot layer.
sys.path.insert(0, str(Path(__file__).parent.parent))
from chatbot.config import CHROMA_DIR, COLLECTION_NAME, CSV_PATH, GEMINI_API_KEY, GEMINI_EMBED_MODEL

CHROMA_BATCH = 500   # rows per ChromaDB upsert call
EMBED_BATCH  = 100   # texts per batchEmbedContents call (Gemini free tier: 100 RPM)


# ── text conversion ───────────────────────────────────────────────────────────

def _present(row: dict, key: str) -> bool:
    """True if the field exists and has a meaningful value (not null/nan/empty)."""
    v = row.get(key)
    return v is not None and str(v).strip() not in ("", "nan", "None", "NaN")


def row_to_text(row: dict) -> str:
    """
    Convert a CSV row into a readable sentence for embedding.
    Every non-null field is included. Future ER visit columns added by
    Issues #11 and #12 (asthma_ed_rate, cardiovascular_ed_rate,
    respiratory_ed_rate) are silently skipped when null.
    """
    parts: list[str] = []

    # Geography header
    geo: list[str] = []
    if _present(row, "borough"):
        geo.append(f"borough {row['borough']}")
    if _present(row, "geo_place_name"):
        geo.append(f"neighborhood {row['geo_place_name']}")
    if _present(row, "zip_code"):
        geo.append(f"ZIP {row['zip_code']}")
    if _present(row, "time_period"):
        geo.append(f"period {row['time_period']}")
    if geo:
        parts.append(", ".join(geo) + ".")

    # Air quality measurements
    if _present(row, "pm25"):
        parts.append(f"PM2.5 {float(row['pm25']):.2f} mcg per cubic meter.")
    if _present(row, "no2"):
        parts.append(f"NO2 {float(row['no2']):.2f} ppb.")
    if _present(row, "ozone"):
        parts.append(f"Ozone {float(row['ozone']):.2f} ppb.")
    if _present(row, "aqi"):
        parts.append(f"AQI {int(float(row['aqi']))}.")
    if _present(row, "purpleair_pm25"):
        parts.append(f"PurpleAir PM2.5 {float(row['purpleair_pm25']):.2f} mcg per cubic meter.")
    if _present(row, "truck_vmt"):
        parts.append(f"Truck route traffic {float(row['truck_vmt']):.2f} million miles.")

    # Health outcomes (current)
    if _present(row, "asthma_er_rate"):
        parts.append(f"Asthma ER rate {float(row['asthma_er_rate']):.1f} per 100,000.")
    if _present(row, "cardiovascular_hosp_rate"):
        parts.append(f"Cardiovascular hospitalization rate {float(row['cardiovascular_hosp_rate']):.1f} per 100,000.")
    if _present(row, "respiratory_hosp_rate"):
        parts.append(f"Respiratory hospitalization rate {float(row['respiratory_hosp_rate']):.1f} per 100,000.")
    if _present(row, "pm25_deaths"):
        parts.append(f"PM2.5 attributable death rate {float(row['pm25_deaths']):.2f} per 100,000.")

    # Future ER visit columns — silently included when populated (Issues #11/#12)
    if _present(row, "asthma_ed_rate"):
        parts.append(f"Asthma ED rate {float(row['asthma_ed_rate']):.1f} per 100,000.")
    if _present(row, "cardiovascular_ed_rate"):
        parts.append(f"Cardiovascular ED rate {float(row['cardiovascular_ed_rate']):.1f} per 100,000.")
    if _present(row, "respiratory_ed_rate"):
        parts.append(f"Respiratory ED rate {float(row['respiratory_ed_rate']):.1f} per 100,000.")

    return " ".join(parts)


def stable_id(row: dict) -> str:
    """
    MD5 of the four identity fields. Re-running the script upserts the same
    IDs, so the vector count stays constant after the first run.
    """
    key = "|".join([
        str(row.get("borough", "") or ""),
        str(row.get("geo_place_name", "") or ""),
        str(row.get("time_period", "") or ""),
        str(row.get("zip_code", "") or ""),
    ])
    return hashlib.md5(key.encode()).hexdigest()


def _build_metadata(row: dict) -> dict:
    """
    Build a ChromaDB-compatible metadata dict (str/int/float values only).
    Null fields are omitted — retrieval.py accesses them with .get() / None default.
    """
    meta: dict = {}
    for key, val in row.items():
        if val is None:
            continue
        if isinstance(val, float) and math.isnan(val):
            continue
        meta[key] = val if isinstance(val, (int, float, bool)) else str(val)
    return meta


# ── main ──────────────────────────────────────────────────────────────────────

def _embed_batch(texts: list[str], api_key: str, model: str, retries: int = 5) -> list[list[float]]:
    """
    Embed multiple texts in one batchEmbedContents call.
    Retries automatically on 429 using the delay suggested by the API.
    One batch call = up to 100 texts, so 2,267 rows needs only ~23 API calls.
    """
    import re as _re
    import requests as _req
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in .env")
    url = (
        f"https://generativelanguage.googleapis.com"
        f"/v1beta/models/{model}:batchEmbedContents"
    )
    body = {
        "requests": [
            {"model": f"models/{model}", "content": {"parts": [{"text": t}]}}
            for t in texts
        ]
    }
    for attempt in range(retries):
        r = _req.post(url, json=body, headers={"x-goog-api-key": api_key}, timeout=60)
        if r.status_code == 429:
            m = _re.search(r"retry in (\d+(?:\.\d+)?)s", r.text)
            delay = max(float(m.group(1)) + 5, 65) if m else 65
            print(f"  [rate limit] sleeping {delay:.0f}s before retry {attempt + 1}/{retries}…")
            time.sleep(delay)
            continue
        if not r.ok:
            print(f"\n  [API ERROR] {r.status_code} — {r.text[:500]}")
        r.raise_for_status()
        return [e["values"] for e in r.json()["embeddings"]]
    raise RuntimeError(f"Embedding failed after {retries} retries")


def main() -> None:
    try:
        import chromadb
    except ImportError as exc:
        print(f"[error] Missing dependency: {exc}")
        sys.exit(1)

    if not GEMINI_API_KEY:
        print("[error] GEMINI_API_KEY not set — required for embedding")
        sys.exit(1)

    print("Vector ingest\n")

    # load CSV
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        print(f"[error] {csv_path} not found — run src/datamerge.py first")
        sys.exit(1)

    df = pd.read_csv(csv_path).drop_duplicates()
    print(f"  [ok]   Loaded {len(df):,} rows from {csv_path}")

    rows = df.where(df.notna(), other=None).to_dict(orient="records")
    print(f"  [ok]   Embedding model: {GEMINI_EMBED_MODEL} (via REST API)")

    # ChromaDB persistent store — always recreate so dimension stays consistent
    chroma_dir = Path(CHROMA_DIR)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        chroma.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma.create_collection(name=COLLECTION_NAME)
    print(f"  [ok]   ChromaDB collection {COLLECTION_NAME!r} at {chroma_dir}\n")

    # Build text/id/metadata lists first (no API calls yet)
    total = len(rows)
    all_texts:      list[str]         = []
    all_ids:        list[str]         = []
    all_metadatas:  list[dict]        = []
    all_embeddings: list[list[float]] = []

    for row in rows:
        all_texts.append(row_to_text(row))
        all_ids.append(stable_id(row))
        all_metadatas.append(_build_metadata(row))

    # Batch-embed: up to 100 texts per API call (~23 calls for 2,267 rows)
    n_embed_batches = math.ceil(total / EMBED_BATCH)
    print(f"Embedding {total:,} rows in {n_embed_batches} batch calls via Gemini REST API...\n")

    for i in range(n_embed_batches):
        start = i * EMBED_BATCH
        end   = min(start + EMBED_BATCH, total)
        batch_embeddings = _embed_batch(
            all_texts[start:end], GEMINI_API_KEY, GEMINI_EMBED_MODEL
        )
        all_embeddings.extend(batch_embeddings)
        print(f"  {end:>5,}/{total:,} rows embedded")

    # Upsert to ChromaDB in batches of CHROMA_BATCH
    n_chroma_batches = math.ceil(total / CHROMA_BATCH)
    for i in range(n_chroma_batches):
        start = i * CHROMA_BATCH
        end   = min(start + CHROMA_BATCH, total)
        collection.upsert(
            ids=all_ids[start:end],
            documents=all_texts[start:end],
            embeddings=all_embeddings[start:end],
            metadatas=all_metadatas[start:end],
        )

    print(f"\n  Total vectors in collection: {collection.count():,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
