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
from pathlib import Path

import pandas as pd

# Import shared constants so this stays in sync with the chatbot layer.
sys.path.insert(0, str(Path(__file__).parent.parent))
from chatbot.config import CHROMA_DIR, COLLECTION_NAME, CSV_PATH, EMBED_MODEL

BATCH_SIZE = 128


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

def main() -> None:
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(f"[error] Missing dependency: {exc}")
        print("  Run: pip install sentence-transformers chromadb")
        sys.exit(1)

    print("Vector ingest\n")

    # load CSV
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        print(f"[error] {csv_path} not found — run src/datamerge.py first")
        sys.exit(1)

    df = pd.read_csv(csv_path).drop_duplicates()
    print(f"  [ok]   Loaded {len(df):,} rows from {csv_path}")

    # convert NaN → None so _present() and _build_metadata() work cleanly
    rows = df.where(df.notna(), other=None).to_dict(orient="records")

    # embedding model (~80 MB download on first run, cached to ~/.cache after)
    print(f"  Loading embedding model {EMBED_MODEL!r}...")
    model = SentenceTransformer(EMBED_MODEL)
    print("  [ok]   Model loaded")

    # ChromaDB persistent store
    chroma_dir = Path(CHROMA_DIR)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"  [ok]   ChromaDB collection {COLLECTION_NAME!r} at {chroma_dir}\n")

    # batch embed + upsert
    total = len(rows)
    n_batches = math.ceil(total / BATCH_SIZE)
    print(f"Embedding {total:,} rows in {n_batches} batches of {BATCH_SIZE}...\n")

    for i in range(n_batches):
        batch = rows[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

        texts      = [row_to_text(r) for r in batch]
        ids        = [stable_id(r) for r in batch]
        metadatas  = [_build_metadata(r) for r in batch]
        embeddings = model.encode(
            texts, batch_size=BATCH_SIZE, show_progress_bar=False
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        done = min((i + 1) * BATCH_SIZE, total)
        print(f"  batch {i + 1:>3}/{n_batches}  [{done:>5,}/{total:,} rows]")

    print(f"\n  Total vectors in collection: {collection.count():,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
