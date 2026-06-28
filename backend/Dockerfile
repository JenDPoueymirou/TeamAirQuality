# ── Build stage: install heavy deps and pre-download the embedding model ───────
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

# Pre-generated dataset and ChromaDB vector store.
# Build these locally before running docker build:
#   python src/dataingestion.py
#   python src/datamerge.py
#   python src/ingest.py
COPY data/ data/

# Logs directory — runtime writes go here; mount a volume to persist across restarts
RUN mkdir -p logs

EXPOSE 8000

# API keys must be injected at runtime — never baked into the image.
# Run with:  docker run --env-file .env -p 8000:8000 nyc-air-quality-chatbot
# Or pass individually: -e GEMINI_API_KEY=... -e AIRNOW_API_KEY=... -e PURPLEAIR_API_KEY=...
CMD ["uvicorn", "chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]
