# NYC Air Quality Chatbot

A grounded AI chatbot that answers questions about air pollution and public health outcomes across New York City's five boroughs. The model can only cite facts from the dataset — it cannot hallucinate statistics it hasn't seen.

Built for the Bloomberg Hackathon.

---

## Project Description

**Research question:** Which NYC communities bear a disproportionate pollution burden, and how does that burden correlate with measurable health outcomes?

The dataset covers 2005–2024 at UHF42 neighborhood granularity — the same geography used by NYC DOHMH for public health surveillance. Each row links air quality measurements (PM2.5, NO2, ozone, AQI) to health outcomes (asthma ER rates, cardiovascular hospitalization rates, respiratory hospitalization rates, PM2.5-attributable deaths) and traffic density (truck VMT).

**Environmental justice angle:** South Bronx neighborhoods like Hunts Point and Mott Haven, Brownsville in Brooklyn, and industrial waterfront areas like Greenpoint and Gowanus carry outsized pollution loads relative to the rest of the city. The chatbot makes this neighborhood-level disparity queryable in plain English.

---

## Free Stack — Zero Cost

| Tool                          | Role                | Cost  |
| ----------------------------- | ------------------- | ----- |
| Google Gemini 2.0 Flash       | LLM inference       | $0.00 |
| sentence-transformers         | Local embeddings    | $0.00 |
| ChromaDB                      | Local vector store  | $0.00 |
| NYC Open Data (Socrata)       | Pollution dataset   | $0.00 |
| NYC DOHMH Epiquery            | ER visit dataset    | $0.00 |
| EPA AirNow                    | Real-time AQI       | $0.00 |
| PurpleAir                     | Community sensors   | $0.00 |
| FastAPI + uvicorn             | Backend API         | $0.00 |

---

## Free Tier Limits

Gemini 2.0 Flash free tier: **1,500 requests/day · 15 requests/minute · 1 million token context window**

- Monitor usage: `GET /usage/summary`
- Warning log fires at: 1,400 requests/day
- No credit card required

---

## Prerequisites

- Python 3.11+
- **Gemini API key** (free):
  1. Go to [aistudio.google.com](https://aistudio.google.com)
  2. Sign in with any Google account
  3. Click **Get API Key** → **Create API key**
  4. Copy the key — you'll paste it into `.env` in step 3 below
- **AirNow API key** (free): [docs.airnowapi.org/account/request](https://docs.airnowapi.org/account/request)
- **PurpleAir API key** (free): [develop.purpleair.com](https://develop.purpleair.com)

---

## Setup

Two paths: **Docker** (recommended for demos) or **local Python** (recommended for development).

---

### Option A — Docker

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
# 1. Clone the repo
git clone <repo-url>
cd TeamAirQuality

# 2. Configure environment variables
cp backend/.env.example backend/.env
# Open backend/.env and fill in your three API keys:
#   GEMINI_API_KEY=AIza...   ← from aistudio.google.com
#   AIRNOW_API_KEY=...
#   PURPLEAIR_API_KEY=...

# 3. Generate the dataset — runs Python locally once to produce data/ files
#    (Docker bakes these into the image so the container needs no internet access)
cd backend
python src/dataingestion.py     # ~2 minutes — fetches from NYC Open Data + APIs
python src/datamerge.py         # <10 seconds — merges into merged_final.csv
python src/ingest.py            # ~3 minutes — embeds rows into ChromaDB
cd ..

# 4. Build the image
#    The builder stage pre-downloads the sentence-transformers model weights
#    so the first /chat request has no cold-start latency.
docker build -t nyc-air-quality-chatbot backend/

# 5. Run the container — secrets injected at runtime, never baked into the image
docker run --env-file backend/.env -p 8000:8000 nyc-air-quality-chatbot
```

The server starts at `http://localhost:8000`. Interactive API docs: `http://localhost:8000/docs`

To persist usage logs across container restarts:

```bash
docker run --env-file backend/.env \
  -p 8000:8000 \
  -v "$(pwd)/backend/logs:/app/logs" \
  nyc-air-quality-chatbot
```

---

### Option B — Local Python

All commands run from the `backend/` directory.

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd TeamAirQuality/backend

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and fill in your three API keys:
#   GEMINI_API_KEY=AIza...   ← from aistudio.google.com
#   AIRNOW_API_KEY=...
#   PURPLEAIR_API_KEY=...

# 5. Fetch raw data from all four sources (~2 minutes)
python src/dataingestion.py

# 6. Merge into a single wide-format CSV (< 10 seconds)
python src/datamerge.py

# 7. Embed all rows into ChromaDB (~3 minutes on first run, then cached)
python src/ingest.py

# 8. Start the API server
uvicorn chatbot.main:app --reload
```

The server starts at `http://localhost:8000`. Interactive API docs: `http://localhost:8000/docs`

---

## Verify Each Step Works

Run these in order to confirm the pipeline built correctly before sending your first chat.

```bash
# Confirm step 6 produced the merged CSV
python -c "import pandas as pd; df = pd.read_csv('data/merged_final.csv'); print(len(df), 'rows')"
# Expected: 2171 rows

# Confirm step 7 embedded all rows into ChromaDB
python -c "import chromadb; c = chromadb.PersistentClient('data/.chroma'); print(c.get_collection('nyc_pollution').count(), 'vectors')"
# Expected: 2171 vectors

# Confirm the server loaded all dependencies
curl http://localhost:8000/health
# Expected: {"status":"ok","csv_rows":2171,...,"llm_provider":"Google Gemini (free tier)"}

# Confirm data endpoints
curl http://localhost:8000/boroughs
curl http://localhost:8000/stats/borough
curl http://localhost:8000/stats/correlations

# Confirm the chatbot answers with grounded citations
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Which borough has the highest asthma rate?"}'
# Expected: answer mentions Bronx and contains at least one (Row N) citation
```

---

## Running Evaluations

The eval suite checks 15 questions with known correct answers derived from the dataset.

```bash
# Server must be running first (default port 8001 for evals)
python evals/run_evals.py

# Show full LLM answers for every question
python evals/run_evals.py --verbose

# Target a different server port
python evals/run_evals.py --url http://127.0.0.1:8000
```

MVP is considered working at **10/15 questions passed**.

---

## Common Issues

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `RuntimeError: GEMINI_API_KEY not found` | `.env` doesn't exist or key is blank | Run `cp .env.example .env`, fill in the key, restart the server |
| `RuntimeError: CSV not found` | Data pipeline hasn't run | Run steps 5 and 6 (`dataingestion.py` then `datamerge.py`) |
| `RuntimeError: ChromaDB not found` | Ingest hasn't run | Run step 7 (`src/ingest.py`) |
| HTTP 429 from `/chat` | Hit 15 req/min free-tier limit | Wait 60 seconds and retry |
| `ModuleNotFoundError` on startup | Virtual environment not activated | Run `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux) |
| Docker: `RuntimeError: CSV not found` | `data/` wasn't built before `docker build` | Run `src/dataingestion.py`, `datamerge.py`, `ingest.py` locally first, then rebuild |
| Docker: `RuntimeError: GEMINI_API_KEY not found` | `--env-file` flag missing from `docker run` | Use `docker run --env-file backend/.env ...` — never copy `.env` into the image |
| Docker: port already in use | Something else is on 8000 | Use `-p 8001:8000` and hit `localhost:8001` instead |
| Docker: stale data after re-ingestion | Image still has the old `data/` baked in | Re-run `docker build` after regenerating data files |

---

## Project Structure

```
TeamAirQuality/
└── backend/
    ├── src/
    │   ├── dataingestion.py   # Fetch raw data from 4 sources → data/*.csv
    │   ├── datamerge.py       # Join CSVs → data/merged_final.csv (2,171 rows)
    │   └── ingest.py          # Embed rows → data/.chroma/ (2,171 vectors)
    ├── chatbot/
    │   ├── config.py          # All env vars and constants
    │   ├── main.py            # FastAPI app, endpoints, call_llm, log_request
    │   ├── retrieval.py       # Hybrid retrieval (structured filter + semantic)
    │   └── prompt.py          # System prompt template and grounding rules
    ├── evals/
    │   ├── golden_set.json    # 15 questions with expected answers
    │   └── run_evals.py       # Eval runner — prints score out of 15
    ├── docs/
    │   └── architecture.md    # System diagram and layer descriptions
    ├── data/                  # Generated — not committed to git
    │   ├── merged_final.csv
    │   └── .chroma/
    ├── logs/                  # Generated — usage.csv written by /chat
    ├── .env.example           # Copy to .env and fill in keys
    ├── requirements.txt
    ├── Dockerfile             # Two-stage build: deps + pre-downloaded model
    └── .dockerignore          # Excludes .env, logs/, __pycache__, evals/
```

---

## API Reference

| Method | Path                  | Description                                                  |
| ------ | --------------------- | ------------------------------------------------------------ |
| GET    | `/health`             | Dependency check — csv_rows, chroma_vectors, llm_provider   |
| GET    | `/boroughs`           | List of boroughs in the dataset                              |
| GET    | `/stats/borough`      | Per-borough mean for all pollution and health metrics        |
| GET    | `/stats/correlations` | Pearson r between pollutants and health outcomes             |
| GET    | `/stats/hotspots`     | Top 10 neighborhoods by asthma ER rate                       |
| GET    | `/usage/summary`      | Today's request count, token totals, remaining quota         |
| POST   | `/chat`               | Grounded LLM answer with (Row N) citations                   |

Full interactive docs: `http://localhost:8000/docs`

---

## How the Chatbot Works

Every `/chat` request runs this pipeline before calling the LLM:

1. **Intent extraction** — scan the question for borough names, UHF neighborhoods, years, and ZIP codes using keyword matching and regex. No LLM needed.
2. **Structured filter** — filter `merged_final.csv` in memory using the detected values. Fast and precise for known entities.
3. **Semantic search** — embed the question locally with `all-MiniLM-L6-v2` (~80 MB, runs on CPU), query ChromaDB for the most similar stored row vectors. Catches paraphrasing the keyword filter would miss.
4. **Merge and deduplicate** — combine both result sets, cap at 8 rows.
5. **Grounded prompt** — inject the rows directly into the system prompt. The model can only cite facts from these rows.
6. **LLM call** — send to Gemini 2.0 Flash with the grounded system prompt in `system_instruction`.
7. **Log** — append token counts and metadata to `logs/usage.csv`.

The model is instructed to cite every number as `(Row N)` and say `"I don't have that data in my current context."` if the retrieved rows don't cover the question.

See [docs/architecture.md](backend/docs/architecture.md) for the full system diagram.

---

## Key Dataset Findings

- **Manhattan** has the highest NO2 (25.3 ppb avg) and highest PM2.5 (9.5 µg/m³ avg) among boroughs, driven by tunnel density and highway traffic
- **Staten Island** has the lowest PM2.5 (7.3 µg/m³) and the lowest asthma ER rate (50.6 per 100,000)
- **PM2.5 has declined sharply** — from 11.1 µg/m³ citywide in 2009 to 6.1 µg/m³ in 2022
- **Central Harlem** has the highest neighborhood-level asthma ER rate (~260 per 100,000)
- **Respiratory and cardiovascular hospitalization rates** correlate at r = 0.87 — areas with one burden tend to have the other
- **PM2.5 and NO2** correlate at r = 0.96 across neighborhoods — same traffic sources drive both
