import logging
import os
from contextlib import asynccontextmanager
from datetime import date
import chromadb
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer


from chatbot.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    CSV_PATH,
    DAILY_REQUEST_LIMIT,
    EMBED_MODEL,
    LOGS_PATH,
    validate_config,
)
from chatbot.chat import router as chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log=logging.getLogger(__name__)


df: pd.DataFrame = pd.DataFrame()
collection: chromadb.Collection = None
embed_model: SentenceTransformer = None


# ── Lifespan (startup + shutdown) ──────────────────────────────────────────────
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Everything in the 'before yield' block runs on startup.
    Everything after yield runs on shutdown.
    FastAPI's lifespan replaces the old @app.on_event("startup") pattern.
    """
    global df, collection, embed_model
 
    log.info("=" * 55)
    log.info("  NYC Pollution Chatbot — starting up")
    log.info("=" * 55)
 
    # 1. Validate config — surface missing keys early
    warnings = validate_config()
    for w in warnings:
        log.warning("CONFIG WARNING: %s", w)
 
    # 2. Load CSV
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        log.info("CSV loaded — %d rows from %s", len(df), CSV_PATH)
    else:
        log.warning("CSV not found at %s — data endpoints will return empty", CSV_PATH)
 
    # 3. Load ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        log.info("ChromaDB loaded — %d vectors", collection.count())
    except Exception as e:
        log.warning("ChromaDB failed to load: %s — semantic search disabled", e)
 
    # 4. Load embedding model (downloads ~80MB on first run, then cached)
    try:
        log.info("Loading embedding model '%s' ...", EMBED_MODEL)
        embed_model = SentenceTransformer(EMBED_MODEL)
        log.info("Embedding model ready")
    except Exception as e:
        log.warning("Embedding model failed to load: %s — semantic search disabled", e)
 
    # 5. Ensure logs directory exists
    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
 
    log.info("=" * 55)
    log.info("  Startup complete — http://localhost:8000/docs")
    log.info("=" * 55)
 
    yield  # ← app runs here
 
    log.info("Shutting down.")
 
 
# ── App ────────────────────────────────────────────────────────────────────────
 
app = FastAPI(
    title="NYC Pollution Chatbot API",
    description=(
        "Grounded AI chatbot over the NYC Air Pollution & Disease dataset "
        "(2005–2024, 5 boroughs, ~19k rows). Free stack: OpenRouter + "
        "sentence-transformers + ChromaDB."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten this when you deploy
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Register routers
app.include_router(chat_router)

@app.get("/health", tags=["Meta"])
async def health():
    """
    Confirms the app is running and surfaces the state of every
    dependency. Check this first whenever something feels wrong.
    """
    config_warnings = validate_config()
 
    # Count today's requests from usage log
    requests_today = 0
    if os.path.exists(LOGS_PATH):
        try:
            log_df = pd.read_csv(LOGS_PATH)
            if "timestamp" in log_df.columns:
                log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
                requests_today = int(
                    (log_df["timestamp"].dt.date == date.today()).sum()
                )
        except Exception:
            pass
 
    return {
        "status": "ok" if not config_warnings else "degraded",
        "config_warnings": config_warnings,
        "data": {
            "csv_rows": len(df),
            "chroma_vectors": collection.count() if collection else 0,
        },
        "models": {
            "embed_model": EMBED_MODEL,
            "llm": "meta-llama/llama-3.1-8b-instruct:free",
            "llm_provider": "OpenRouter (free tier)",
        },
        "rate_limits": {
            "requests_today": requests_today,
            "daily_limit": DAILY_REQUEST_LIMIT,
            "remaining_today": max(0, DAILY_REQUEST_LIMIT - requests_today),
        },
    }
@app.get("/boroughs", tags=["Data"])
async def list_boroughs():
    """Returns the list of boroughs present in the dataset."""
    if df.empty or "borough" not in df.columns:
        return {"boroughs": []}
    return {
        "boroughs": sorted(df["borough"].dropna().unique().tolist())
    }
 
 
@app.get("/stats/borough", tags=["Data"])
async def borough_stats():
    """
    Per-borough averages for key pollution and health metrics.
    Pure pandas — no LLM involved. Fast and free.
    """
    if df.empty:
        return {"error": "Dataset not loaded. Run src/datamerge.py first."}
 
    metric_cols = [
        c for c in [
            "pm25", "no2", "ozone", "aqi",
            "asthma_er_rate", "cardiovascular_hosp_rate",
            "respiratory_hosp_rate", "pm25_deaths",
        ]
        if c in df.columns
    ]
 
    if "borough" not in df.columns or not metric_cols:
        return {"error": "Expected columns not found in dataset."}
 
    summary = (
        df.groupby("borough")[metric_cols]
        .mean()
        .round(2)
        .to_dict()
    )
    return {"borough_averages": summary}
 
 
@app.get("/stats/hotspots", tags=["Data"])
async def hotspot_stats():
    """
    Top 10 neighborhoods by asthma ER visit rate.
    Pure pandas — no LLM involved.
    """
    if df.empty:
        return {"error": "Dataset not loaded."}
    if "asthma_er_rate" not in df.columns or "geo_place_name" not in df.columns:
        return {"error": "Required columns not found."}
 
    top10 = (
        df[["geo_place_name", "borough", "asthma_er_rate", "pm25"]]
        .dropna(subset=["asthma_er_rate"])
        .sort_values("asthma_er_rate", ascending=False)
        .drop_duplicates(subset=["geo_place_name"])
        .head(10)
        .to_dict(orient="records")
    )
    return {"top_10_by_asthma_er_rate": top10}
 
 
@app.get("/stats/correlations", tags=["Data"])
async def correlations():
    """
    Precomputed Pearson correlation matrix per borough between
    pollution metrics and health outcomes. This is the core research
    finding of the original dataset — surfaces it without any LLM call.
    """
    if df.empty:
        return {"error": "Dataset not loaded."}
 
    metric_cols = [
        c for c in [
            "pm25", "no2", "truck_vmt",
            "asthma_er_rate", "cardiovascular_hosp_rate",
            "respiratory_hosp_rate",
        ]
        if c in df.columns
    ]
 
    result = {}
    for borough in df["borough"].dropna().unique():
        subset = df[df["borough"] == borough][metric_cols].dropna()
        if len(subset) > 2:
            result[borough] = subset.corr().round(3).to_dict()
 
    return {"correlations_by_borough": result}
 
 
@app.get("/usage/summary", tags=["Meta"])
async def usage_summary():
    """
    Request count and token usage summary from logs/usage.csv.
    Helps the team track OpenRouter free tier consumption.
    """
    if not os.path.exists(LOGS_PATH):
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
            "most_common_borough_filter": (
                log_df["filters_applied"]
                .dropna()
                .mode()[0]
                if "filters_applied" in log_df.columns and len(log_df) > 0
                else None
            ),
        }
    except Exception as e:
        return {"error": f"Could not read usage log: {e}"}