import csv
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

import chromadb
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

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
    RATE_WARN_THRESHOLD,
    validate_config,
)
from chatbot.prompt import build_messages, build_system_prompt
from chatbot.retrieval import init_retrieval, retrieve

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
    log.info("  NYC Pollution Chatbot -- starting up")
    log.info("=" * 55)

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
    log.info("CSV loaded -- %d rows from %s", len(df), CSV_PATH)

    # 2. Connect ChromaDB — hard failure if ingest hasn't been run
    if not Path(CHROMA_DIR).exists():
        raise RuntimeError(
            f"Chroma store not found at {CHROMA_DIR!r}. "
            "Run src/ingest.py before starting the server."
        )
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    log.info("ChromaDB loaded -- %d vectors", collection.count())

    # 3. Load embedding model (~80 MB on first run, cached after)
    try:
        log.info("Loading embedding model %r ...", EMBED_MODEL)
        embed_model = SentenceTransformer(EMBED_MODEL)
        log.info("Embedding model ready")
    except Exception as exc:
        log.warning("Embedding model failed: %s -- semantic search disabled", exc)

    # 4. Wire retrieval module with all three dependencies
    init_retrieval(df, collection, embed_model)

    # 5. Initialize OpenRouter / OpenAI client (used by /chat)
    if OPENROUTER_API_KEY:
        oai_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        log.info("OpenAI client pointed at %s", OPENROUTER_BASE_URL)
    else:
        log.warning("OPENROUTER_API_KEY not set -- /chat will be disabled until key is added")

    # 6. Ensure logs directory exists
    Path(LOGS_PATH).parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 55)
    log.info("  Startup complete -- http://localhost:8000/docs")
    log.info("=" * 55)

    yield

    log.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NYC Pollution Chatbot API",
    description=(
        "Grounded AI chatbot over the NYC Air Pollution & Disease dataset "
        "(2005-2024, 5 boroughs, 2,171 rows). Free stack: OpenRouter + "
        "sentence-transformers + ChromaDB."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    history: list[dict] = Field(default_factory=list, max_length=10)


class ChatResponse(BaseModel):
    answer: str
    model_used: str
    filters_applied: dict
    rows_retrieved: int


# ── LLM helpers ───────────────────────────────────────────────────────────────

def call_llm(messages: list[dict]) -> tuple[str, str, dict]:
    """
    Try primary model first; on 429 or 503 retry once with the fallback model.
    Raises HTTP 503 if both models fail.

    Note: OpenRouter free models sometimes omit usage counts in their response.
    Treat missing prompt_tokens / completion_tokens as 0 rather than crashing —
    known limitation of the free tier.
    """
    if oai_client is None:
        raise HTTPException(
            status_code=503,
            detail="LLM not configured -- set OPENROUTER_API_KEY in .env",
        )

    last_exc: Exception | None = None

    for model in (LLM_MODEL, LLM_FALLBACK):
        try:
            resp = oai_client.chat.completions.create(
                model=model,
                messages=messages,
            )
            answer = resp.choices[0].message.content or ""
            usage = {
                "prompt_tokens":     (resp.usage.prompt_tokens     or 0) if resp.usage else 0,
                "completion_tokens": (resp.usage.completion_tokens or 0) if resp.usage else 0,
            }
            log.info("LLM responded via %s", model)
            return answer, model, usage
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in (429, 503):
                log.warning("Model %s returned %s -- trying fallback", model, status)
                last_exc = exc
                continue
            log.error("LLM call failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

    raise HTTPException(
        status_code=503,
        detail="Both LLM models are unavailable. Try again later.",
    )


def log_request(
    question_length: int,
    model_used: str,
    input_tokens: int,
    output_tokens: int,
    rows_retrieved: int,
    filters_applied: dict,
) -> None:
    log_path = Path(LOGS_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp":       datetime.now().isoformat(),
        "question_length": question_length,
        "model_used":      model_used,
        "input_tokens":    input_tokens,
        "output_tokens":   output_tokens,
        "rows_retrieved":  rows_retrieved,
        "filters_applied": str(filters_applied),
    }

    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    try:
        log_df = pd.read_csv(log_path)
        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
        today_count = int((log_df["timestamp"].dt.date == date.today()).sum())
        if today_count >= RATE_WARN_THRESHOLD:
            log.warning(
                "Daily request count (%d) approaching limit of %d",
                today_count, DAILY_REQUEST_LIMIT,
            )
    except Exception:
        pass


# ── Meta endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health():
    """Dependency health check. Returns status 'ok' when CSV and ChromaDB are loaded."""
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
        "status":          "ok" if not config_warnings else "degraded",
        "config_warnings": config_warnings,
        "csv_rows":        len(df),
        "csv_columns":     list(df.columns),
        "chroma_vectors":  collection.count() if collection else 0,
        "embed_model":     EMBED_MODEL,
        "llm_primary":     LLM_MODEL,
        "llm_fallback":    LLM_FALLBACK,
        "llm_provider":    "OpenRouter (free tier)",
        "requests_today":  requests_today,
        "daily_limit":     DAILY_REQUEST_LIMIT,
    }


@app.get("/usage/summary", tags=["Meta"])
async def usage_summary():
    """Request count and token usage from logs/usage.csv."""
    if not Path(LOGS_PATH).exists():
        return {
            "total_requests":      0,
            "requests_today":      0,
            "requests_remaining":  DAILY_REQUEST_LIMIT,
            "daily_limit":         DAILY_REQUEST_LIMIT,
            "total_input_tokens":  0,
            "total_output_tokens": 0,
            "avg_input_tokens":    None,
            "avg_output_tokens":   None,
            "warning_threshold":   RATE_WARN_THRESHOLD,
        }
    try:
        log_df = pd.read_csv(LOGS_PATH)
        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
        today_df = log_df[log_df["timestamp"].dt.date == date.today()]
        requests_today = len(today_df)

        has_in  = "input_tokens"  in log_df.columns
        has_out = "output_tokens" in log_df.columns

        return {
            "total_requests":      len(log_df),
            "requests_today":      requests_today,
            "requests_remaining":  max(0, DAILY_REQUEST_LIMIT - requests_today),
            "daily_limit":         DAILY_REQUEST_LIMIT,
            "total_input_tokens":  int(log_df["input_tokens"].sum())  if has_in  else 0,
            "total_output_tokens": int(log_df["output_tokens"].sum()) if has_out else 0,
            "avg_input_tokens":    round(log_df["input_tokens"].mean(), 1)  if has_in  else None,
            "avg_output_tokens":   round(log_df["output_tokens"].mean(), 1) if has_out else None,
            "warning_threshold":   RATE_WARN_THRESHOLD,
        }
    except Exception as exc:
        return {"error": f"Could not read usage log: {exc}"}


# ── Chat endpoint ──────────────────────────────────────────────────────────────

@app.post("/chat", tags=["Chat"], response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Grounded LLM chat. Retrieves relevant dataset rows, injects them into
    the system prompt, and returns a cited answer. The model only sees the
    rows this function retrieves -- it cannot access the full dataset.
    """
    chunks, filters, row_count = retrieve(req.message)
    system_prompt = build_system_prompt(chunks)
    messages = build_messages(system_prompt, req.history, req.message)

    answer, model_used, usage = call_llm(messages)

    log_request(
        question_length=len(req.message),
        model_used=model_used,
        input_tokens=usage.get("prompt_tokens", 0) or 0,
        output_tokens=usage.get("completion_tokens", 0) or 0,
        rows_retrieved=row_count,
        filters_applied=filters,
    )

    return ChatResponse(
        answer=answer,
        model_used=model_used,
        filters_applied=filters,
        rows_retrieved=row_count,
    )


# ── Data endpoints ─────────────────────────────────────────────────────────────

@app.get("/boroughs", tags=["Data"])
async def list_boroughs():
    """Unique boroughs present in the dataset."""
    if df.empty or "borough" not in df.columns:
        return {"boroughs": []}
    return {"boroughs": sorted(df["borough"].dropna().unique().tolist())}


@app.get("/stats/borough", tags=["Data"])
async def borough_stats():
    """Per-borough mean for every pollution and health metric."""
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
    # JSON cannot encode float NaN -- replace with None (-> null)
    summary = {
        col: {b: (v if pd.notna(v) else None) for b, v in vals.items()}
        for col, vals in borough_means.items()
    }
    return {"borough_averages": summary}


@app.get("/stats/hotspots", tags=["Data"])
async def hotspot_stats():
    """Top 10 neighborhoods by asthma ER visit rate."""
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

    poll_avg = df.groupby("geo_place_name")[pollutant_cols].mean()
    out_avg  = df.groupby("geo_place_name")[outcome_cols].mean()
    combined = poll_avg.join(out_avg, how="inner")

    result: dict = {
        "method": "cross-sectional mean per neighborhood, Pearson r",
        "citywide": combined.corr().round(3).to_dict(),
    }

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
