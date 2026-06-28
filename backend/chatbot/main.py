"""
main.py
-------
FastAPI application for the NYC Pollution Chatbot.
Restructured to use Google Gemini 2.0 Flash (free tier)
instead of OpenRouter.

Key difference from OpenRouter version:
  - Uses google-genai SDK instead of openai SDK
  - Gemini uses generate_content() not chat.completions.create()
  - System prompt passed as system_instruction parameter
  - History formatted as Content objects not role/content dicts
  - Free tier: 1,500 requests/day vs OpenRouter's 200/day
"""

import csv
import os
from datetime import date, datetime
from contextlib import asynccontextmanager

import pandas as pd
import chromadb
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MAX_TOKENS,
    EMBED_MODEL,
    CSV_PATH,
    CHROMA_DIR,
    COLLECTION_NAME,
    DAILY_REQUEST_LIMIT,
    RATE_LIMIT_WARNING_THRESHOLD,
)
from chatbot.retrieval import init_retrieval, retrieve
from chatbot.prompt import build_system_prompt


# ── Shared state ───────────────────────────────────────────────────────────────

df: pd.DataFrame | None = None
gemini_client: genai.Client | None = None
LOG_PATH = "logs/usage.csv"
LOG_HEADERS = [
    "timestamp", "question_length", "model_used",
    "input_tokens", "output_tokens",
    "rows_retrieved", "filters_applied",
]


# ── Startup ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, gemini_client

    print("\n── Starting NYC Pollution Chatbot (Gemini) ─────────────────")

    # 1. Load CSV
    print(f"[startup] Loading dataset from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        raise RuntimeError(
            f"CSV not found at {CSV_PATH}. "
            "Run src/dataingestion.py and src/datamerge.py first."
        )
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"[startup] Loaded {len(df)} rows, {len(df.columns)} columns")

    # 2. Connect to ChromaDB
    print(f"[startup] Connecting to ChromaDB at {CHROMA_DIR}...")
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            f"ChromaDB not found at {CHROMA_DIR}. "
            "Run src/ingest.py first."
        )
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"[startup] ChromaDB ready — {collection.count()} vectors")

    # 3. Load embedding model
    print(f"[startup] Loading embedding model '{EMBED_MODEL}'...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    print("[startup] Embedding model ready")

    # 4. Wire retrieval layer
    init_retrieval(df, collection, embed_model)
    print("[startup] Retrieval layer initialized")

    # 5. Initialize Gemini client
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not found. "
            "Get a free key at aistudio.google.com and add it to .env"
        )
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"[startup] Gemini client ready — model: {GEMINI_MODEL}")

    # 6. Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADERS)
    print(f"[startup] Usage log at {LOG_PATH}")

    print("── Ready. Server is accepting requests ─────────────────────\n")

    yield

    print("\n[shutdown] Server stopping.")


# ── App instance ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="NYC Pollution Chatbot API",
    description="Grounded AI chatbot over the NYC Air Pollution & Disease dataset",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    history: list[dict] = Field(default=[], max_length=10)


class ChatResponse(BaseModel):
    answer: str
    model_used: str
    filters_applied: dict
    rows_retrieved: int


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(
    system_prompt: str,
    history: list[dict],
    user_message: str,
) -> tuple[str, str, dict]:
    """
    Call Gemini 2.0 Flash with the grounded system prompt and conversation.

    Gemini SDK differences from OpenRouter/OpenAI:
      - System prompt goes in system_instruction, not as a message
      - History is formatted as types.Content objects
      - Response text is at response.text
      - Token counts at response.usage_metadata
    """
    # Convert history to Gemini Content format
    # Gemini uses "user" and "model" roles (not "assistant")
    gemini_history = []
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        # Map "assistant" → "model" for Gemini's role convention
        gemini_role = "model" if role == "assistant" else "user"
        gemini_history.append(
            types.Content(
                role=gemini_role,
                parts=[types.Part(text=content)]
            )
        )

    # Add current user message
    gemini_history.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        )
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=gemini_history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=LLM_MAX_TOKENS,
                temperature=0.2,   # lower = more factual, less creative
            ),
        )

        answer = response.text
        usage = {
            "input_tokens":  getattr(response.usage_metadata, "prompt_token_count",     0) or 0,
            "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
        }
        return answer, GEMINI_MODEL, usage

    except Exception as e:
        error_str = str(e).lower()
        if "quota" in error_str or "429" in error_str or "rate" in error_str:
            raise HTTPException(
                status_code=429,
                detail="Gemini free tier rate limit hit. "
                       "You have 15 requests/minute and 1,500/day. "
                       "Wait a moment and try again."
            )
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")


# ── Logging helpers ────────────────────────────────────────────────────────────

def get_today_count() -> int:
    today = date.today().isoformat()
    count = 0
    try:
        with open(LOG_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("timestamp", "").startswith(today):
                    count += 1
    except FileNotFoundError:
        pass
    return count


def log_request(
    question: str,
    model: str,
    usage: dict,
    rows_retrieved: int,
    filters: dict,
) -> None:
    try:
        today_count = get_today_count() + 1
        if today_count >= RATE_LIMIT_WARNING_THRESHOLD:
            print(
                f"[usage] WARNING: {today_count} requests today. "
                f"Approaching Gemini free tier limit of {DAILY_REQUEST_LIMIT}/day."
            )
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                len(question),
                model,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                rows_retrieved,
                str(filters),
            ])
    except Exception as e:
        print(f"[usage] Logging failed (non-fatal): {e}")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "csv_rows": len(df) if df is not None else 0,
        "csv_columns": len(df.columns) if df is not None else 0,
        "embed_model": EMBED_MODEL,
        "llm_model": GEMINI_MODEL,
        "llm_provider": "Google Gemini (free tier)",
        "requests_today": get_today_count(),
        "daily_limit": DAILY_REQUEST_LIMIT,
    }


@app.get("/boroughs")
async def list_boroughs():
    if df is None or "borough" not in df.columns:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    boroughs = sorted(df["borough"].dropna().unique().tolist())
    return {"boroughs": boroughs, "count": len(boroughs)}


@app.get("/stats/borough")
async def borough_stats():
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    target_cols = [
        "pm25", "no2", "ozone", "aqi",
        "asthma_er_rate", "asthma_ed_rate",
        "cardiovascular_hosp_rate", "cardiovascular_ed_rate",
        "respiratory_hosp_rate", "pm25_deaths",
    ]
    available = [c for c in target_cols if c in df.columns]
    summary = (
        df.groupby("borough")[available]
        .mean()
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )
    return {"boroughs": summary, "metrics": available}


@app.get("/stats/correlations")
async def correlations():
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    corr_cols = [
        "pm25", "no2", "truck_vmt",
        "asthma_er_rate", "asthma_ed_rate",
        "cardiovascular_hosp_rate", "cardiovascular_ed_rate",
        "respiratory_hosp_rate", "pm25_deaths",
    ]
    available = [c for c in corr_cols if c in df.columns]
    result = {}
    citywide = df[available].dropna()
    result["citywide"] = citywide.corr().round(3).to_dict()
    if "borough" in df.columns:
        for borough in df["borough"].dropna().unique():
            subset = df[df["borough"] == borough][available].dropna()
            if len(subset) > 5:
                result[borough] = subset.corr().round(3).to_dict()
    return result


@app.get("/usage/summary")
async def usage_summary():
    today = date.today().isoformat()
    today_count = 0
    total_count = 0
    total_input = 0
    total_output = 0
    try:
        with open(LOG_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_count += 1
                total_input  += int(row.get("input_tokens", 0) or 0)
                total_output += int(row.get("output_tokens", 0) or 0)
                if row.get("timestamp", "").startswith(today):
                    today_count += 1
    except FileNotFoundError:
        pass
    return {
        "requests_today":     today_count,
        "requests_remaining": max(0, DAILY_REQUEST_LIMIT - today_count),
        "daily_limit":        DAILY_REQUEST_LIMIT,
        "minute_limit":       15,
        "total_requests":     total_count,
        "total_input_tokens": total_input,
        "total_output_tokens":total_output,
        "avg_input_tokens":   round(total_input  / total_count, 1) if total_count else 0,
        "avg_output_tokens":  round(total_output / total_count, 1) if total_count else 0,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Retrieve relevant rows
    chunks, filters, row_count = retrieve(req.message)

    # Build grounded system prompt
    system_prompt = build_system_prompt(chunks)

    # Call Gemini — note: system prompt passed separately, not in messages
    answer, model_used, usage = call_llm(
        system_prompt=system_prompt,
        history=req.history,
        user_message=req.message,
    )

    # Log usage
    log_request(req.message, model_used, usage, row_count, filters)

    return ChatResponse(
        answer=answer,
        model_used=model_used,
        filters_applied=filters,
        rows_retrieved=row_count,
    )