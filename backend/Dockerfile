# ── Build stage: install dependencies only ───────────────────────────────────
# sentence-transformers and PyTorch are no longer needed at runtime.
# Embeddings are handled by the Gemini API (text-embedding-004).
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-fetch public Socrata data during the build (no API keys required).
# AirNow and PurpleAir are skipped automatically when their keys are absent.
# This shaves ~30 seconds off every cold start.
COPY src/     src/
COPY chatbot/ chatbot/
RUN PYTHONPATH=/install/lib/python3.11/site-packages python src/dataingestion.py
RUN PYTHONPATH=/install/lib/python3.11/site-packages python src/datamerge.py


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

COPY --from=builder /install /usr/local

WORKDIR /app

COPY chatbot/ chatbot/
COPY src/     src/
COPY start.sh .

# Pre-fetched Socrata CSV files from builder — saves time on cold start
COPY --from=builder /build/data/ data/

RUN mkdir -p logs && chmod +x start.sh

EXPOSE 8000

# API keys must be injected at runtime — never baked into the image.
CMD ["bash", "start.sh"]
