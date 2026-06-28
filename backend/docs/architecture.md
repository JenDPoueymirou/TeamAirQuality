# Architecture

A draw.io diagram of the full request flow is in [`docs/architecture.drawio`](architecture.drawio) (open at [app.diagrams.net](https://app.diagrams.net)). The ASCII version below covers the same ground.

---

## System Diagram

```
User question
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  POST /chat  (FastAPI — chatbot/main.py)            │
│                                                     │
│  ChatRequest validation (Pydantic)                  │
│    message: str [1–500 chars]                       │
│    history: list[dict] [max 10 turns]               │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Hybrid Retrieval  (chatbot/retrieval.py)           │
│                                                     │
│  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │  Structured Filter   │  │   Semantic Search    │ │
│  │                      │  │                      │ │
│  │  extract_filters()   │  │  embed query with    │ │
│  │  ── borough name     │  │  all-MiniLM-L6-v2   │ │
│  │  ── neighborhood     │  │  (local, CPU, free)  │ │
│  │  ── year (regex)     │  │         │            │ │
│  │  ── ZIP (regex)      │  │         ▼            │ │
│  │         │            │  │    ChromaDB query    │ │
│  │         ▼            │  │  (top-6 results;     │ │
│  │  pandas DataFrame    │  │  borough pre-filter  │ │
│  │  filter (top-3)      │  │  if borough detected)│ │
│  └──────────┬───────────┘  └──────────┬───────────┘ │
│             │                         │             │
│             └────────────┬────────────┘             │
│                          ▼                          │
│              Merge + deduplicate                    │
│              on (geo_place_name + time_period)      │
│              Cap at TOP_K = 8 rows                  │
└──────────────────────────┬──────────────────────────┘
                           │  up to 8 [Row N] chunks
                           ▼
┌─────────────────────────────────────────────────────┐
│  Prompt Builder  (chatbot/prompt.py)                │
│                                                     │
│  build_system_prompt(chunks)                        │
│  ── injects retrieved rows into SYSTEM_TEMPLATE     │
│  ── grounding rules: cite every number as (Row N)  │
└──────────────────────────┬──────────────────────────┘
                           │  system_prompt (str)
                           ▼
┌─────────────────────────────────────────────────────┐
│  LLM  (call_llm — chatbot/main.py)                  │
│                                                     │
│  Google Gemini 2.0 Flash (free tier)                │
│  1,500 req/day · 15 req/min · 1M token context     │
│                                                     │
│  google-genai SDK                                   │
│  ── system_prompt → system_instruction parameter   │
│  ── history as types.Content objects               │
│     (role "user" / "model")                        │
│  ── response.text for the answer                   │
│                                                     │
│  On 429 → HTTP 429 with rate-limit message         │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
              log_request() → logs/usage.csv
                           │
                           ▼
                    ChatResponse
              answer, model_used,
              filters_applied, rows_retrieved
```

---

## Layers

### Data Pipeline (one-time setup, run from `backend/`)

| Script | Input | Output | Notes |
|--------|-------|--------|-------|
| `src/dataingestion.py` | 4 APIs | `data/*.csv` | Socrata (no key), AirNow, PurpleAir, NYC DOHMH |
| `src/datamerge.py` | `data/*.csv` | `data/merged_final.csv` | Long → wide pivot; 2,171 rows |
| `src/ingest.py` | `merged_final.csv` | `data/.chroma/` | 2,171 vectors, MD5 stable IDs, idempotent |

**Why the pipeline is separate from the API:** Data is fetched once and stored locally. The API never calls external data sources at request time — it only reads the pre-built CSV and vector store. This keeps latency low and avoids hitting rate limits during normal use.

### API Server (`chatbot/main.py`)

FastAPI with a lifespan context manager that loads all dependencies at startup:

1. `merged_final.csv` → pandas DataFrame (in-memory, ~300 KB)
2. ChromaDB PersistentClient → collection of 2,171 vectors (~40 MB on disk)
3. `all-MiniLM-L6-v2` SentenceTransformer → loaded into RAM (~80 MB, CPU)
4. `genai.Client` → Gemini 2.0 Flash client (raises `RuntimeError` if `GEMINI_API_KEY` missing)

The DataFrame and collection are module-level globals shared across all requests. Loading happens once at boot, not per-request.

### Retrieval (`chatbot/retrieval.py`)

Two strategies run on every query and their results are merged:

**Structured filter** — zero latency, high precision for known entities:
- Keyword scan for borough names (`BOROUGHS` list in `config.py`)
- Keyword scan for UHF42 neighborhood names (`UHF_NEIGHBORHOODS` list, ~50 entries)
- Regex for 4-digit years (2005–2024) and 5-digit NYC ZIP codes starting with `1`
- Multi-borough queries (e.g. "Compare Bronx and Manhattan") set `filters["borough"]` to a `list[str]`, which flows into `isin()` for the DataFrame and `{"$in": [...]}` for ChromaDB

**Semantic search** — handles paraphrasing, synonyms, and conceptual queries:
- Embed the question with the same model used at ingest time (identical embedding space)
- Query ChromaDB with `n_results=6`
- Optional `where=` pre-filter by borough to keep results geographically scoped

**Deduplication:** Results are merged on `(geo_place_name, time_period)` and capped at `TOP_K = 8`. Structured rows go first (higher confidence). Returns `(chunks, filters, row_count)`.

### Prompt (`chatbot/prompt.py`)

`build_system_prompt(chunks)` injects the retrieved rows into `SYSTEM_TEMPLATE` and returns a plain string. This string becomes the `system_instruction` for the Gemini call — separate from the message history.

Grounding rules embedded in the prompt:
1. Only state facts present in the retrieved rows
2. Cite `(Row N)` for every number
3. Say `"I don't have that data in my current context."` if rows don't cover the question
4. General science explanations are allowed but must be labelled as background context
5. Keep answers to 4–6 sentences

### Embeddings and Vector Store

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` — 80 MB, runs on CPU, no GPU required
- **Dimension:** 384
- **Store:** ChromaDB `PersistentClient` writing to `data/.chroma/`
- **IDs:** MD5 hash of `borough|geo_place_name|time_period|zip_code` — makes re-runs idempotent (upsert, not insert)
- **Metadata:** every non-null field from the CSV row stored as Chroma metadata, enabling `where=` filtering at query time

### Usage Logging

`log_request()` appends one row to `logs/usage.csv` after every successful `/chat` call:

```
timestamp, question_length, model_used, input_tokens, output_tokens, rows_retrieved, filters_applied
```

`GET /usage/summary` reads this file and returns daily counts and token totals. A WARNING log fires when `requests_today >= 1400` (the `RATE_LIMIT_WARNING_THRESHOLD`).

---

## Configuration (`chatbot/config.py`)

All environment variables and app-wide constants live in one place. Every other module imports from here; nothing else calls `os.getenv()` directly.

| Constant | Default | Meaning |
|----------|---------|---------|
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model ID |
| `LLM_MAX_TOKENS` | `512` | Max tokens in LLM response |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `TOP_K` | `8` | Max rows injected into each LLM prompt |
| `STRUCT_K` | `3` | Rows from structured pandas filter |
| `VECTOR_K` | `6` | Rows from ChromaDB semantic search |
| `DAILY_REQUEST_LIMIT` | `1500` | Gemini 2.0 Flash free tier daily cap |
| `RATE_LIMIT_WARNING_THRESHOLD` | `1400` | Log WARNING before hitting daily limit |
| `MINUTE_REQUEST_LIMIT` | `15` | Gemini free tier per-minute cap |

---

## Data Schema

`data/merged_final.csv` — 2,171 rows, 14 columns:

| Column | Source | Notes |
|--------|--------|-------|
| `borough` | Derived | Inferred from `geo_join_id` prefix |
| `geo_place_name` | NYC Open Data | UHF42 neighborhood or borough name |
| `zip_code` | AirNow / PurpleAir | Only populated for AirNow/PurpleAir rows |
| `time_period` | NYC Open Data | Annual (e.g. `2019`) or 3-year range (e.g. `2017-2019`) |
| `pm25` | NYC Open Data | Annual mean PM2.5 (µg/m³) |
| `no2` | NYC Open Data | Annual mean NO2 (ppb) |
| `ozone` | NYC Open Data | Summer mean ozone (ppb) |
| `aqi` | AirNow | Max daily AQI across parameters |
| `purpleair_pm25` | PurpleAir | Real-time sensor PM2.5 (µg/m³) |
| `truck_vmt` | NYC Open Data | Annual truck miles (millions) |
| `asthma_er_rate` | NYC Open Data | Asthma ER visits per 100,000 (under 18) |
| `cardiovascular_hosp_rate` | NYC Open Data | Cardiovascular hosp per 100,000 (40+) |
| `respiratory_hosp_rate` | NYC Open Data | Respiratory hosp per 100,000 (20+) |
| `pm25_deaths` | NYC Open Data | PM2.5-attributable deaths per 100,000 (30+) |

Rows sourced from UHF42 geography take priority over UHF34 when both cover the same area (UHF42 carries truck VMT data; UHF34 does not).
