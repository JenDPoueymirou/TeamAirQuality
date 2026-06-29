# ── Build stage: install deps, download model, pre-build ChromaDB index ─────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download sentence-transformers model weights into the install prefix so
# they are baked into the image and first-request latency is zero.
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source needed to run the data pipeline at build time
COPY src/     src/
COPY chatbot/ chatbot/

# Fetch public Socrata data (no API keys needed — AirNow/PurpleAir are skipped
# automatically when their keys are absent), merge, then embed into ChromaDB.
# This runs the memory-heavy embedding step once at build time instead of on
# every container start, keeping runtime well under the 512 MB free-tier limit.
RUN PYTHONPATH=/install/lib/python3.11/site-packages python src/dataingestion.py
RUN PYTHONPATH=/install/lib/python3.11/site-packages python src/datamerge.py
RUN PYTHONPATH=/install/lib/python3.11/site-packages python src/ingest.py


# ── Runtime stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Copy installed packages + pre-downloaded model weights from builder
COPY --from=builder /install /usr/local
# sentence-transformers caches models in ~/.cache/torch/sentence_transformers
COPY --from=builder /root/.cache /root/.cache

WORKDIR /app

# Application source
COPY chatbot/ chatbot/
COPY src/     src/
COPY start.sh .

# Pre-built dataset and ChromaDB vectors from builder — no embedding at startup
COPY --from=builder /build/data/ data/

# Logs directory — usage.csv written by /chat; mount a volume to persist across restarts
RUN mkdir -p logs && chmod +x start.sh

EXPOSE 8000

# API keys must be injected at runtime — never baked into the image.
# Run with:  docker run --env-file .env -p 8000:8000 nyc-air-quality-chatbot
# Or pass individually: -e GEMINI_API_KEY=... -e AIRNOW_API_KEY=... -e PURPLEAIR_API_KEY=...
CMD ["bash", "start.sh"]
