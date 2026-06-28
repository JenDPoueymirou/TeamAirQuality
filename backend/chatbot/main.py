import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import chromadb
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from chatbot.chat import router as chat_router
from chatbot.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    CSV_PATH,
    DAILY_REQUEST_LIMIT,
    EMBED_MODEL,
    LLM_FALLBACK,
    LLM_MODEL,
    LOGS_PATH,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    validate_config,
)
from chatbot.retrieval import init_retrieval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Module-level state (populated by lifespan) ─────────────────────────────────

df: pd.DataFrame = pd.DataFrame()
collection: chromadb.Collection | None = None
embed_model: SentenceTransformer | None = None
oai_client: OpenAI | None = None


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, collection, embed_model, oai_client

    log.info("=" * 55)
    log.info("  NYC Pollution Chatbot — starting up")
    log.info("=" * 55)

    # surface missing config early (non-fatal warnings)
    for w in validate_config():
        log.warning("CONFIG: %s", w)

    # 1. Load CSV — hard failure if pipeline hasn't been run
    if not Path(CSV_PATH).exists():
        raise RuntimeError(
            f"CSV not found at {CSV_PATH!r}. "
            "Run src/datamerge.py before starting the server."
        )
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    log.info("CSV loaded — %d rows from %s", len(df), CSV_PATH)

    # 2. Connect ChromaDB — hard failure if ingest hasn't been run
    if not Path(CHROMA_DIR).exists():
        raise RuntimeError(
            f"Chroma store not found at {CHROMA_DIR!r}. "
            "Run src/ingest.py before starting the server."
        )
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    log.info("ChromaDB loaded — %d vectors", collection.count())

    # 3. Load embedding model (~80 MB on first run, cached after)
    try:
        log.info("Loading embedding model %r ...", EMBED_MODEL)
        embed_model = SentenceTransformer(EMBED_MODEL)
        log.info("Embedding model ready")
    except Exception as exc:
        log.warning("Embedding model failed: %s — semantic search disabled", exc)

    # 4. Wire retrieval module with all three dependencies
    init_retrieval(df, collection, embed_model)

    # 5. Initialize OpenRouter / OpenAI client (used by Issue #7 chat endpoint)
    if OPENROUTER_API_KEY:
        oai_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        log.info("OpenAI client pointed at %s", OPENROUTER_BASE_URL)
    else:
        log.warning("OPENROUTER_API_KEY not set — /chat will be disabled until key is added")

    # 6. Ensure logs directory and usage file exist
    Path(LOGS_PATH).parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 55)
    log.info("  Startup complete — http://localhost:8000/docs")
    log.info("=" * 55)

    yield  # app runs here

    log.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NYC Pollution Chatbot API",
    description=(
        "Grounded AI chatbot over the NYC Air Pollution & Disease dataset "
        "(2005–2024, 5 boroughs, 2,171 rows). Free stack: OpenRouter + "
        "sentence-transformers + ChromaDB."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten when deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


# ── Meta endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health():
    """
    Dependency health check. Returns status 'ok' when CSV and ChromaDB
    are both loaded with data. Check this first if something feels wrong.
    """
    requests_today = 0
    if Path(LOGS_PATH).exists():
        try:
            log_df = pd.read_csv(LOGS_PATH)
            if "timestamp" in log_df.columns:
                log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
                requests_today = int((log_df["timestamp"].dt.date == date.today()).sum())
        except Exception:
            pass

    config_warnings = validate_config()
    return {
        "status": "ok" if not config_warnings else "degraded",
        "config_warnings": config_warnings,
        "csv_rows": len(df),
        "csv_columns": list(df.columns),
        "chroma_vectors": collection.count() if collection else 0,
        "embed_model": EMBED_MODEL,
        "llm_primary": LLM_MODEL,
        "llm_fallback": LLM_FALLBACK,
        "llm_provider": "OpenRouter (free tier)",
        "requests_today": requests_today,
        "daily_limit": DAILY_REQUEST_LIMIT,
    }


@app.get("/usage/summary", tags=["Meta"])
async def usage_summary():
    """Request count and token usage from logs/usage.csv."""
    if not Path(LOGS_PATH).exists():
        return {
            "total_requests": 0,
            "requests_today": 0,
            "remaining_today": DAILY_REQUEST_LIMIT,
            "message": "No requests logged yet.",
        }
    try:
        log_df = pd.read_csv(LOGS_PATH)
        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
        today_df = log_df[log_df["timestamp"].dt.date == date.today()]
        return {
            "total_requests_all_time": len(log_df),
            "requests_today": len(today_df),
            "remaining_today": max(0, DAILY_REQUEST_LIMIT - len(today_df)),
            "avg_input_tokens": round(log_df["input_tokens"].mean(), 1)
                if "input_tokens" in log_df.columns else None,
            "avg_output_tokens": round(log_df["output_tokens"].mean(), 1)
                if "output_tokens" in log_df.columns else None,
        }
    except Exception as exc:
        return {"error": f"Could not read usage log: {exc}"}


# ── Data endpoints ─────────────────────────────────────────────────────────────

@app.get("/boroughs", tags=["Data"])
async def list_boroughs():
    """Unique boroughs present in the dataset."""
    if df.empty or "borough" not in df.columns:
        return {"boroughs": []}
    return {"boroughs": sorted(df["borough"].dropna().unique().tolist())}


@app.get("/stats/borough", tags=["Data"])
async def borough_stats():
    """
    Per-borough mean for every pollution and health metric.
    Pure pandas — no LLM. Columns that don't exist yet (asthma_ed_rate,
    cardiovascular_ed_rate) are silently skipped until Issues #11/#12.
    """
    if df.empty:
        return {"error": "Dataset not loaded. Run src/datamerge.py first."}

    metric_cols = [
        c for c in [
            "pm25", "no2", "ozone", "aqi",
            "asthma_er_rate", "asthma_ed_rate",
            "cardiovascular_hosp_rate", "cardiovascular_ed_rate",
            "respiratory_hosp_rate", "pm25_deaths",
        ]
        if c in df.columns
    ]

    if "borough" not in df.columns or not metric_cols:
        return {"error": "Expected columns not found in dataset."}

    borough_means = (
        df.groupby("borough")[metric_cols]
        .mean()
        .round(2)
        .to_dict()
    )
    # JSON can't encode float NaN — replace with None (→ null)
    summary = {
        col: {b: (v if pd.notna(v) else None) for b, v in vals.items()}
        for col, vals in borough_means.items()
    }
    return {"borough_averages": summary}


@app.get("/stats/hotspots", tags=["Data"])
async def hotspot_stats():
    """Top 10 neighborhoods by asthma ER visit rate. Pure pandas."""
    if df.empty:
        return {"error": "Dataset not loaded."}
    if "asthma_er_rate" not in df.columns or "geo_place_name" not in df.columns:
        return {"error": "Required columns not found."}

    top10_df = (
        df[["geo_place_name", "borough", "asthma_er_rate", "pm25"]]
        .dropna(subset=["asthma_er_rate"])
        .sort_values("asthma_er_rate", ascending=False)
        .drop_duplicates(subset=["geo_place_name"])
        .head(10)
    )
    top10 = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in top10_df.to_dict(orient="records")
    ]
    return {"top_10_by_asthma_er_rate": top10}


@app.get("/stats/correlations", tags=["Data"])
async def correlations():
    """
    Pearson r-values between pollution and health metrics.

    Method: cross-sectional averages per neighborhood.
    PM2.5 data (annual) and health outcomes (3-year ranges) live on
    different rows in the dataset, so we average each metric per
    geo_place_name then join for pairwise complete observations.
    This produces one data point per neighborhood — the correlation
    captures geographic variation, not time-series variation.

    Returns citywide correlations plus per-borough breakdowns.
    """
    if df.empty:
        return {"error": "Dataset not loaded."}

    pollutant_cols = [c for c in ("pm25", "no2", "ozone", "truck_vmt") if c in df.columns]
    outcome_cols   = [c for c in (
        "asthma_er_rate", "cardiovascular_hosp_rate",
        "respiratory_hosp_rate", "pm25_deaths",
    ) if c in df.columns]

    if not pollutant_cols or not outcome_cols:
        return {"error": "Not enough metric columns for correlation."}

    # average each metric per neighborhood (collapses time)
    poll_avg = df.groupby("geo_place_name")[pollutant_cols].mean()
    out_avg  = df.groupby("geo_place_name")[outcome_cols].mean()
    combined = poll_avg.join(out_avg, how="inner")

    # pairwise complete observations (pandas default for .corr())
    result: dict = {
        "method": "cross-sectional mean per neighborhood, Pearson r",
        "citywide": combined.corr().round(3).to_dict(),
    }

    # per-borough: filter to only that borough's neighborhoods
    borough_lookup = (
        df[["geo_place_name", "borough"]]
        .dropna()
        .drop_duplicates()
        .set_index("geo_place_name")["borough"]
    )
    combined["borough"] = combined.index.map(borough_lookup)

    for borough in df["borough"].dropna().unique():
        subset = combined[combined["borough"] == borough].drop(columns=["borough"])
        if len(subset) >= 3:
            result[borough] = subset.corr().round(3).to_dict()

    return {"correlations": result}
